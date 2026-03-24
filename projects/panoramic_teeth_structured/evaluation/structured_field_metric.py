from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmpose.registry import METRICS


def _extract_field(data_sample, field_name: str):
    if isinstance(data_sample, dict):
        return data_sample.get(field_name, None)
    return getattr(data_sample, field_name, None)


def _extract_map(container, map_name: str):
    if container is None:
        return None
    if isinstance(container, dict):
        value = container.get(map_name, None)
    else:
        value = getattr(container, map_name, None)
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _compute_binary_overlap(pred_map: np.ndarray, gt_map: np.ndarray,
                            threshold: float) -> Dict[str, float]:
    if pred_map.ndim == 3:
        pred_map = pred_map[0]
    if gt_map.ndim == 3:
        gt_map = gt_map[0]

    pred_binary = pred_map >= threshold
    gt_binary = gt_map >= 0.5
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    pred_area = pred_binary.sum()
    gt_area = gt_binary.sum()
    return dict(
        intersection=float(intersection),
        union=float(union),
        pred_area=float(pred_area),
        gt_area=float(gt_area))


@METRICS.register_module()
class StructuredFieldMetric(BaseMetric):
    """Evaluate root-mask IoU and mesial/distal boundary Dice."""

    default_prefix = 'structured'

    def __init__(self,
                 threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: str = 'structured'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold = float(threshold)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_fields = _extract_field(data_sample, 'pred_fields')
            gt_fields = _extract_field(data_sample, 'gt_fields')
            if pred_fields is None or gt_fields is None:
                continue

            root_overlap = _compute_binary_overlap(
                _extract_map(pred_fields, 'root_mask'),
                _extract_map(gt_fields, 'root_mask'), self.threshold)
            mesial_overlap = _compute_binary_overlap(
                _extract_map(pred_fields, 'mesial_boundary'),
                _extract_map(gt_fields, 'mesial_boundary'), self.threshold)
            distal_overlap = _compute_binary_overlap(
                _extract_map(pred_fields, 'distal_boundary'),
                _extract_map(gt_fields, 'distal_boundary'), self.threshold)

            self.results.append(
                dict(
                    root=root_overlap,
                    mesial=mesial_overlap,
                    distal=distal_overlap))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if not results:
            return dict(root_mIoU=0.0, root_mDice=0.0, mesial_mDice=0.0,
                        distal_mDice=0.0, boundary_mDice=0.0)

        def _mean_metric(group: str, key: str) -> float:
            values = np.array(
                [item[group][key] for item in results], dtype=np.float32)
            return float(values.mean())

        root_i = np.array([item['root']['intersection'] for item in results],
                          dtype=np.float32)
        root_u = np.array([item['root']['union'] for item in results],
                          dtype=np.float32)
        root_p = np.array([item['root']['pred_area'] for item in results],
                          dtype=np.float32)
        root_g = np.array([item['root']['gt_area'] for item in results],
                          dtype=np.float32)
        root_iou = root_i / np.maximum(root_u, 1.0)
        root_dice = (2 * root_i) / np.maximum(root_p + root_g, 1.0)

        def _dice(group: str) -> float:
            intersection = np.array(
                [item[group]['intersection'] for item in results],
                dtype=np.float32)
            pred_area = np.array(
                [item[group]['pred_area'] for item in results],
                dtype=np.float32)
            gt_area = np.array(
                [item[group]['gt_area'] for item in results],
                dtype=np.float32)
            return float(
                ((2 * intersection) / np.maximum(pred_area + gt_area, 1.0))
                .mean())

        mesial_dice = _dice('mesial')
        distal_dice = _dice('distal')
        return dict(
            root_mIoU=float(root_iou.mean()),
            root_mDice=float(root_dice.mean()),
            mesial_mDice=mesial_dice,
            distal_mDice=distal_dice,
            boundary_mDice=float((mesial_dice + distal_dice) * 0.5))

