from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.datasets.transforms import PackPoseInputs
from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix

from .annotation_utils import (derive_keypoints_from_contours,
                               distance_transform_from_binary, flip_points,
                               rasterize_polygon, rasterize_polyline,
                               resample_semantic_side, transform_points)


def _should_swap_sides(direction: str) -> bool:
    return direction in {'horizontal', 'diagonal'}


@TRANSFORMS.register_module()
class ExpandToothBBox(BaseTransform):
    """Expand a root-based bbox into a larger single-tooth context box.

    The source annotations currently build bbox from the root polygon, which is
    too tight for anatomy-aware landmarks such as B/C. This transform expands
    the bbox asymmetrically to include crown, gingiva and alveolar context.
    """

    def __init__(self,
                 left_ratio: float = 0.35,
                 right_ratio: float = 0.35,
                 top_ratio: float = 0.70,
                 bottom_ratio: float = 0.30,
                 occlusal_shift_ratio: float = 0.0,
                 upper_height_scale: float = 1.0,
                 upper_post_shift_ratio: float = 0.0,
                 min_pad_x: float = 24.0,
                 min_pad_top: float = 72.0,
                 min_pad_bottom: float = 36.0,
                 use_tooth_id_templates: bool = True,
                 clip_border: bool = True) -> None:
        self.left_ratio = float(left_ratio)
        self.right_ratio = float(right_ratio)
        self.top_ratio = float(top_ratio)
        self.bottom_ratio = float(bottom_ratio)
        self.occlusal_shift_ratio = float(occlusal_shift_ratio)
        self.upper_height_scale = float(upper_height_scale)
        self.upper_post_shift_ratio = float(upper_post_shift_ratio)
        self.min_pad_x = float(min_pad_x)
        self.min_pad_top = float(min_pad_top)
        self.min_pad_bottom = float(min_pad_bottom)
        self.use_tooth_id_templates = bool(use_tooth_id_templates)
        self.clip_border = bool(clip_border)

    def _resolve_template(self, tooth_id: Optional[int]) -> dict:
        template = dict(
            left_ratio=self.left_ratio,
            right_ratio=self.right_ratio,
            top_ratio=self.top_ratio,
            bottom_ratio=self.bottom_ratio,
            occlusal_shift_ratio=self.occlusal_shift_ratio,
            upper_height_scale=self.upper_height_scale,
            upper_post_shift_ratio=self.upper_post_shift_ratio,
            min_pad_x=self.min_pad_x,
            min_pad_top=self.min_pad_top,
            min_pad_bottom=self.min_pad_bottom)

        if not self.use_tooth_id_templates or tooth_id is None:
            return template

        quadrant = int(tooth_id) // 10
        tooth_index = int(tooth_id) % 10

        # Narrower lateral context for incisors/canines, slightly wider for
        # posterior teeth. Use mild vertical bias by arch to reduce useless
        # context on the opposite side.
        if tooth_index <= 3:
            template.update(
                left_ratio=0.07,
                right_ratio=0.07,
                top_ratio=1.10,
                bottom_ratio=0.20,
                occlusal_shift_ratio=0.20,
                min_pad_x=8.0,
                min_pad_top=108.0,
                min_pad_bottom=20.0)
        elif tooth_index <= 5:
            template.update(
                left_ratio=0.10,
                right_ratio=0.10,
                top_ratio=0.95,
                bottom_ratio=0.24,
                occlusal_shift_ratio=0.15,
                min_pad_x=10.0,
                min_pad_top=96.0,
                min_pad_bottom=24.0)
        else:
            template.update(
                left_ratio=0.14,
                right_ratio=0.14,
                top_ratio=0.82,
                bottom_ratio=0.30,
                occlusal_shift_ratio=0.10,
                min_pad_x=14.0,
                min_pad_top=84.0,
                min_pad_bottom=28.0)

        if quadrant in {1, 2}:  # upper teeth: preserve more coronal context
            template['top_ratio'] += 0.10
            template['bottom_ratio'] = max(0.16,
                                           template['bottom_ratio'] - 0.04)
            template['min_pad_top'] += 12.0
            template['occlusal_shift_ratio'] += 0.10
            template['upper_height_scale'] = 2.0 / 3.0
            template['upper_post_shift_ratio'] = 0.25
        elif quadrant in {3, 4}:  # lower teeth: preserve more apical context
            template['top_ratio'] = max(0.60, template['top_ratio'] - 0.08)
            template['bottom_ratio'] += 0.10
            template['min_pad_bottom'] += 12.0

        return template

    def transform(self, results: Dict) -> Optional[dict]:
        bbox = np.asarray(results['bbox'], dtype=np.float32).reshape(-1, 4).copy()
        img_shape = results.get('img_shape', results.get('ori_shape', None))
        img_h, img_w = (img_shape[:2] if img_shape is not None else (None, None))
        tooth_id = results.get('tooth_id')
        template = self._resolve_template(tooth_id)

        widths = np.maximum(bbox[:, 2] - bbox[:, 0], 1.0)
        heights = np.maximum(bbox[:, 3] - bbox[:, 1], 1.0)

        pad_left = np.maximum(widths * template['left_ratio'],
                              template['min_pad_x'])
        pad_right = np.maximum(widths * template['right_ratio'],
                               template['min_pad_x'])
        pad_top = np.maximum(heights * template['top_ratio'],
                             template['min_pad_top'])
        pad_bottom = np.maximum(heights * template['bottom_ratio'],
                                template['min_pad_bottom'])
        shift_y = np.zeros_like(heights)
        if tooth_id is not None:
            quadrant = int(tooth_id) // 10
            shift_sign = 1.0 if quadrant in {1, 2} else -1.0
            shift_y = heights * template['occlusal_shift_ratio'] * shift_sign

        bbox[:, 0] = bbox[:, 0] - pad_left
        bbox[:, 1] = bbox[:, 1] - pad_top + shift_y
        bbox[:, 2] = bbox[:, 2] + pad_right
        bbox[:, 3] = bbox[:, 3] + pad_bottom + shift_y

        if tooth_id is not None and int(tooth_id) // 10 in {1, 2}:
            current_h = np.maximum(bbox[:, 3] - bbox[:, 1], 1.0)
            center_y = 0.5 * (bbox[:, 1] + bbox[:, 3])
            center_y = center_y + current_h * template['upper_post_shift_ratio']
            new_h = current_h * template['upper_height_scale']
            bbox[:, 1] = center_y - 0.5 * new_h
            bbox[:, 3] = center_y + 0.5 * new_h

        if self.clip_border and img_w is not None and img_h is not None:
            bbox[:, 0] = np.clip(bbox[:, 0], 0, img_w - 1)
            bbox[:, 1] = np.clip(bbox[:, 1], 0, img_h - 1)
            bbox[:, 2] = np.clip(bbox[:, 2], 0, img_w - 1)
            bbox[:, 3] = np.clip(bbox[:, 3], 0, img_h - 1)

        results['bbox'] = bbox.astype(np.float32)
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'left_ratio={self.left_ratio}, '
            f'right_ratio={self.right_ratio}, '
            f'top_ratio={self.top_ratio}, '
            f'bottom_ratio={self.bottom_ratio}, '
            f'occlusal_shift_ratio={self.occlusal_shift_ratio}, '
            f'upper_height_scale={self.upper_height_scale}, '
            f'upper_post_shift_ratio={self.upper_post_shift_ratio}, '
            f'min_pad_x={self.min_pad_x}, '
            f'min_pad_top={self.min_pad_top}, '
            f'min_pad_bottom={self.min_pad_bottom}, '
            f'use_tooth_id_templates={self.use_tooth_id_templates}, '
            f'clip_border={self.clip_border})')


@TRANSFORMS.register_module()
class GenerateStructuredToothTargets(BaseTransform):
    """Generate contour, boundary, mask and geometry targets."""

    def __init__(self,
                 num_contour_points: int = 16,
                 line_width: int = 2,
                 use_udp: bool = False) -> None:
        self.num_contour_points = int(num_contour_points)
        self.line_width = int(line_width)
        self.use_udp = bool(use_udp)

    def transform(self, results: Dict) -> Optional[dict]:
        polygon = np.array(
            results.get('root_polygon', []), dtype=np.float32).reshape(-1, 2)
        side_contours = results.get('side_contours', {})
        mesial_points = np.array(
            side_contours.get('M', []), dtype=np.float32).reshape(-1, 2)
        distal_points = np.array(
            side_contours.get('D', []), dtype=np.float32).reshape(-1, 2)

        if len(polygon) < 3 or len(mesial_points) < 2 or len(distal_points) < 2:
            return None

        if results.get('flip', False):
            direction = results.get('flip_direction', 'horizontal')
            polygon = flip_points(polygon, results['img_shape'], direction)
            mesial_points = flip_points(mesial_points, results['img_shape'],
                                        direction)
            distal_points = flip_points(distal_points, results['img_shape'],
                                        direction)
            if _should_swap_sides(direction):
                mesial_points, distal_points = distal_points, mesial_points

        center = np.array(results['input_center'], dtype=np.float32)
        scale = np.array(results['input_scale'], dtype=np.float32)
        input_width, input_height = results['input_size']
        rotation = results.get('bbox_rotation', np.array([0.0], dtype=np.float32))
        if isinstance(rotation, np.ndarray):
            rotation = float(rotation.reshape(-1)[0])
        else:
            rotation = float(rotation)

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rotation,
                output_size=(input_width, input_height))
        else:
            warp_mat = get_warp_matrix(
                center, scale, rotation,
                output_size=(input_width, input_height))

        polygon = transform_points(polygon, warp_mat)
        mesial_points = transform_points(mesial_points, warp_mat)
        distal_points = transform_points(distal_points, warp_mat)

        mesial_contour = resample_semantic_side(mesial_points,
                                                self.num_contour_points)
        distal_contour = resample_semantic_side(distal_points,
                                                self.num_contour_points)
        derived_keypoints, apex = derive_keypoints_from_contours(
            mesial_contour, distal_contour)

        root_mask = rasterize_polygon(
            polygon, width=int(input_width), height=int(input_height))
        mesial_boundary = rasterize_polyline(
            mesial_points,
            width=int(input_width),
            height=int(input_height),
            thickness=self.line_width)
        distal_boundary = rasterize_polyline(
            distal_points,
            width=int(input_width),
            height=int(input_height),
            thickness=self.line_width)

        normalize_by = float(max(input_width, input_height))
        mesial_distance = distance_transform_from_binary(mesial_boundary,
                                                        normalize_by)
        distal_distance = distance_transform_from_binary(distal_boundary,
                                                        normalize_by)

        transformed_keypoints = np.asarray(
            results.get('transformed_keypoints', np.zeros((1, 5, 2))),
            dtype=np.float32)
        if transformed_keypoints.ndim >= 3 and transformed_keypoints.size > 0:
            keypoint_target = transformed_keypoints[0, :, :2].astype(np.float32)
        else:
            keypoint_target = derived_keypoints

        keypoint_visible = results.get('keypoints_visible', None)
        if keypoint_visible is not None:
            keypoint_visible = np.asarray(keypoint_visible, dtype=np.float32)
            if keypoint_visible.ndim == 3:
                keypoint_weights = keypoint_visible[0, :, 0]
            elif keypoint_visible.ndim == 2:
                keypoint_weights = keypoint_visible[0]
            elif keypoint_visible.ndim == 1:
                keypoint_weights = keypoint_visible
            else:
                raise ValueError(
                    f'Unexpected keypoints_visible shape: '
                    f'{keypoint_visible.shape!r}')
            keypoint_weights = np.asarray(keypoint_weights, dtype=np.float32)
        else:
            keypoint_weights = np.ones((len(keypoint_target), ),
                                       dtype=np.float32)

        results['root_mask'] = root_mask[None, ...]
        results['mesial_boundary'] = mesial_boundary[None, ...]
        results['distal_boundary'] = distal_boundary[None, ...]
        results['mesial_distance'] = mesial_distance[None, ...]
        results['distal_distance'] = distal_distance[None, ...]
        results['mesial_contour'] = mesial_contour.astype(np.float32)
        results['distal_contour'] = distal_contour.astype(np.float32)
        results['keypoint_target'] = keypoint_target.astype(np.float32)
        results['keypoint_weights'] = keypoint_weights.astype(np.float32)
        results['apex_target'] = apex.astype(np.float32)
        results['structured_root_polygon'] = polygon.astype(np.float32)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_contour_points={self.num_contour_points}, '
                f'line_width={self.line_width}, use_udp={self.use_udp})')


@TRANSFORMS.register_module()
class PackStructuredToothInputs(PackPoseInputs):
    """Pack structured tooth targets for contour reconstruction training."""

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'category_id',
                            'ori_shape', 'img_shape', 'input_size',
                            'input_center', 'input_scale', 'flip',
                            'flip_direction', 'flip_indices', 'raw_ann_info',
                            'dataset_name', 'tooth_id'),
                 pack_transformed: bool = False):
        super().__init__(
            meta_keys=meta_keys, pack_transformed=pack_transformed)

    def transform(self, results: dict) -> dict:
        field_mapping = dict(self.field_mapping_table)
        field_mapping.update(results.get('field_mapping_table', {}))
        field_mapping.update(
            root_mask='root_mask',
            mesial_boundary='mesial_boundary',
            distal_boundary='distal_boundary',
            mesial_distance='mesial_distance',
            distal_distance='distal_distance')
        results['field_mapping_table'] = field_mapping

        label_mapping = dict(self.label_mapping_table)
        label_mapping.update(results.get('label_mapping_table', {}))
        label_mapping.update(
            keypoint_target='keypoint_targets',
            mesial_contour='mesial_contour',
            distal_contour='distal_contour',
            apex_target='apex_target')
        results['label_mapping_table'] = label_mapping

        # PackPoseInputs stores labels in InstanceData, which requires all
        # fields to share the same leading instance dimension.
        for key in ('keypoint_target', 'keypoint_weights', 'mesial_contour',
                    'distal_contour', 'apex_target'):
            if key in results:
                results[key] = np.expand_dims(
                    np.asarray(results[key], dtype=np.float32), axis=0)
        return super().transform(results)
