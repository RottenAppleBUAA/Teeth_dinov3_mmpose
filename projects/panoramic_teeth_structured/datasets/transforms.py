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
