from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmpose.registry import METRICS


def _extract_field(data_sample, field_name: str):
    if isinstance(data_sample, dict):
        return data_sample.get(field_name, None)
    return getattr(data_sample, field_name, None)


def _extract_mask(container, mask_name: str):
    if container is None:
        return None
    if isinstance(container, dict):
        value = container.get(mask_name, None)
    else:
        value = getattr(container, mask_name, None)
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@METRICS.register_module()
class RootMaskIoUMetric(BaseMetric):
    """Evaluate crop-space binary root masks with Dice and IoU."""

    default_prefix = 'mask'

    def __init__(self,
                 threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: str = 'mask'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold = float(threshold)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_fields = _extract_field(data_sample, 'pred_fields')
            gt_fields = _extract_field(data_sample, 'gt_fields')

            pred_mask = _extract_mask(pred_fields, 'root_mask')
            gt_mask = _extract_mask(gt_fields, 'root_mask')
            if pred_mask is None or gt_mask is None:
                continue

            if pred_mask.ndim == 3:
                pred_mask = pred_mask[0]
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[0]

            pred_binary = pred_mask >= self.threshold
            gt_binary = gt_mask >= 0.5
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            pred_area = pred_binary.sum()
            gt_area = gt_binary.sum()

            self.results.append(
                dict(
                    intersection=float(intersection),
                    union=float(union),
                    pred_area=float(pred_area),
                    gt_area=float(gt_area)))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if not results:
            return {'mIoU': 0.0, 'mDice': 0.0}

        intersection = np.array(
            [result['intersection'] for result in results], dtype=np.float32)
        union = np.array([result['union'] for result in results],
                         dtype=np.float32)
        pred_area = np.array(
            [result['pred_area'] for result in results], dtype=np.float32)
        gt_area = np.array(
            [result['gt_area'] for result in results], dtype=np.float32)

        iou = intersection / np.maximum(union, 1.0)
        dice = (2 * intersection) / np.maximum(pred_area + gt_area, 1.0)
        return {'mIoU': float(iou.mean()), 'mDice': float(dice.mean())}
