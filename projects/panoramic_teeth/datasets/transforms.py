from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.datasets.transforms import PackPoseInputs
from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix


def _flip_points(points: np.ndarray, image_shape, direction: str) -> np.ndarray:
    flipped = points.copy()
    height, width = image_shape[:2]
    if direction == 'horizontal':
        flipped[:, 0] = width - 1 - flipped[:, 0]
    elif direction == 'vertical':
        flipped[:, 1] = height - 1 - flipped[:, 1]
    elif direction == 'diagonal':
        flipped[:, 0] = width - 1 - flipped[:, 0]
        flipped[:, 1] = height - 1 - flipped[:, 1]
    else:
        raise ValueError(f'Unsupported flip direction: {direction}')
    return flipped


@TRANSFORMS.register_module()
class GenerateRootMask(BaseTransform):
    """Rasterize the root polygon into the current topdown crop."""

    def __init__(self, use_udp: bool = False) -> None:
        self.use_udp = use_udp

    def transform(self, results: Dict) -> Optional[dict]:
        polygon = results.get('root_polygon', None)
        if polygon is None or len(polygon) < 3:
            return None

        polygon = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        if results.get('flip', False):
            polygon = _flip_points(polygon, results['img_shape'],
                                   results['flip_direction'])

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
                center, scale, rotation, output_size=(input_width, input_height))
        else:
            warp_mat = get_warp_matrix(
                center, scale, rotation, output_size=(input_width, input_height))

        transformed_polygon = cv2.transform(polygon[None, ...], warp_mat)[0]
        mask = np.zeros((int(input_height), int(input_width)), dtype=np.float32)
        contour = np.round(transformed_polygon).astype(np.int32)
        cv2.fillPoly(mask, [contour], 1.0)

        results['transformed_root_polygon'] = transformed_polygon
        results['root_mask'] = mask[None, ...]
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(use_udp={self.use_udp})'


@TRANSFORMS.register_module()
class PackTeethInputs(PackPoseInputs):
    """Pack keypoint targets together with root mask supervision."""

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
        field_mapping['root_mask'] = 'root_mask'
        results['field_mapping_table'] = field_mapping
        return super().transform(results)
