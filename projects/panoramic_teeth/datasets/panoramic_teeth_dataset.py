from __future__ import annotations

from typing import Optional

import numpy as np

from mmpose.datasets.datasets.base import BaseCocoStyleDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class PanoramicTeethRootDataset(BaseCocoStyleDataset):
    """COCO-style single-tooth dataset with root polygon supervision."""

    METAINFO: dict = dict()

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        data_info = super().parse_data_info(raw_data_info)
        if data_info is None:
            return None

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']
        segmentation = ann.get('segmentation', [])
        if segmentation:
            root_polygon = np.array(segmentation[0], dtype=np.float32).reshape(
                -1, 2)
        else:
            root_polygon = np.zeros((0, 2), dtype=np.float32)

        data_info.update(
            tooth_id=int(ann.get('tooth_id', -1)),
            side_contours=ann.get('side_contours', {}),
            root_polygon=root_polygon,
            img_shape=(img['height'], img['width']),
            ori_shape=(img['height'], img['width']),
        )
        return data_info
