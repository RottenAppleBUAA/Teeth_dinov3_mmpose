from __future__ import annotations

from copy import deepcopy
from math import dist
from pathlib import Path
from typing import Iterable, Mapping, Sequence

TEETH_32_CLASSES = (
    '1',
    '2',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12',
    '13',
    '15',
    '16',
    '17',
    '19',
    '20',
    '21',
    '22',
    '23',
    '24',
    '25',
    '26',
    '27',
    '28',
    '30',
    '32',
    '18',
    '29',
    '3',
    '14',
    '31',
)

_SIDE_KEYPOINTS = {
    'mesial': ('M_C', 'M_B', 'A'),
    'distal': ('D_C', 'D_B', 'A'),
}


def class_id_to_tooth_id(class_id: int) -> int:
    if class_id < 0 or class_id >= len(TEETH_32_CLASSES):
        raise ValueError(f'Invalid class_id {class_id}, expected 0-31.')
    return int(TEETH_32_CLASSES[class_id])


def xyxy_to_xywh(box_xyxy: Sequence[float]) -> list[float]:
    if len(box_xyxy) != 4:
        raise ValueError(f'Expected 4 bbox values, got {len(box_xyxy)}.')
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def normalize_stage1_instance(instance: Mapping[str, object]) -> dict:
    bbox_xyxy = instance.get('bbox_xyxy')
    if not isinstance(bbox_xyxy, Sequence) or len(bbox_xyxy) != 4:
        raise ValueError('Stage1 instance is missing bbox_xyxy.')

    class_id = int(instance['class_id'])
    tooth_id = class_id_to_tooth_id(class_id)
    return dict(
        tooth_id=tooth_id,
        class_id=class_id,
        score=float(instance.get('score', 0.0)),
        bbox_xyxy=[float(v) for v in bbox_xyxy],
        bbox_xywh=xyxy_to_xywh(bbox_xyxy),
    )


def select_best_instances_by_tooth(
        instances: Iterable[Mapping[str, object]]) -> list[dict]:
    best_by_tooth: dict[int, dict] = {}

    for raw_instance in instances:
        instance = normalize_stage1_instance(raw_instance)
        tooth_id = int(instance['tooth_id'])
        previous = best_by_tooth.get(tooth_id)
        if previous is None or float(instance['score']) > float(
                previous['score']):
            best_by_tooth[tooth_id] = instance

    return [best_by_tooth[tooth_id] for tooth_id in sorted(best_by_tooth)]


def parse_stage1_predictions(predictions: Sequence[Mapping[str, object]]
                             ) -> dict[str, list[dict]]:
    parsed: dict[str, list[dict]] = {}
    for entry in predictions:
        image_name = Path(str(entry['image'])).name
        parsed[image_name] = select_best_instances_by_tooth(
            entry.get('instances', []))
    return parsed


def _build_keypoint_map(keypoints: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    keypoint_map: dict[str, dict] = {}
    for keypoint in keypoints:
        name = str(keypoint.get('name', '')).strip()
        if name:
            keypoint_map[name] = dict(keypoint)
    return keypoint_map


def _point_xy(point: Mapping[str, object]) -> tuple[float, float]:
    return float(point['x']), float(point['y'])


def _score_is_valid(point: Mapping[str, object], threshold: float) -> bool:
    score = point.get('score')
    return score is None or float(score) >= threshold


def compute_side_bone_loss(keypoints: Sequence[Mapping[str, object]],
                           side: str,
                           mm_per_pixel: float,
                           keypoint_score_thr: float = 0.0) -> dict:
    if side not in _SIDE_KEYPOINTS:
        raise ValueError(f'Unsupported side "{side}".')

    keypoint_map = _build_keypoint_map(keypoints)
    c_name, b_name, a_name = _SIDE_KEYPOINTS[side]
    missing = [
        name for name in (c_name, b_name, a_name) if name not in keypoint_map
    ]
    if missing:
        return dict(
            side=side,
            cb_mm=None,
            ca_mm=None,
            bone_loss_pct=None,
            valid=False,
            invalid_reason=f'missing_keypoints:{",".join(missing)}',
        )

    side_points = [keypoint_map[name] for name in (c_name, b_name, a_name)]
    if not all(_score_is_valid(point, keypoint_score_thr) for point in side_points):
        return dict(
            side=side,
            cb_mm=None,
            ca_mm=None,
            bone_loss_pct=None,
            valid=False,
            invalid_reason='low_keypoint_score',
        )

    c_point, b_point, a_point = side_points
    cb_mm = dist(_point_xy(c_point), _point_xy(b_point)) * float(mm_per_pixel)
    ca_mm = dist(_point_xy(c_point), _point_xy(a_point)) * float(mm_per_pixel)

    if ca_mm <= 2.0:
        return dict(
            side=side,
            cb_mm=cb_mm,
            ca_mm=ca_mm,
            bone_loss_pct=None,
            valid=False,
            invalid_reason='denominator_non_positive',
        )

    if cb_mm < 2.0:
        bone_loss_pct = 0.0
    else:
        denominator = ca_mm - 2.0
        if denominator <= 0.0:
            return dict(
                side=side,
                cb_mm=cb_mm,
                ca_mm=ca_mm,
                bone_loss_pct=None,
                valid=False,
                invalid_reason='denominator_non_positive',
            )
        bone_loss_pct = ((cb_mm - 2.0) / denominator) * 100.0

    return dict(
        side=side,
        cb_mm=cb_mm,
        ca_mm=ca_mm,
        bone_loss_pct=bone_loss_pct,
        valid=True,
        invalid_reason=None,
    )


def compute_tooth_bone_loss(keypoints: Sequence[Mapping[str, object]],
                            mm_per_pixel: float,
                            keypoint_score_thr: float = 0.0) -> dict:
    mesial = compute_side_bone_loss(
        keypoints=keypoints,
        side='mesial',
        mm_per_pixel=mm_per_pixel,
        keypoint_score_thr=keypoint_score_thr)
    distal = compute_side_bone_loss(
        keypoints=keypoints,
        side='distal',
        mm_per_pixel=mm_per_pixel,
        keypoint_score_thr=keypoint_score_thr)

    valid_sides = [
        side['bone_loss_pct'] for side in (mesial, distal)
        if side['bone_loss_pct'] is not None
    ]
    invalid_reasons = [
        f'{side["side"]}:{side["invalid_reason"]}' for side in (mesial, distal)
        if side['invalid_reason'] is not None
    ]
    mean_pct = (sum(valid_sides) / len(valid_sides)) if valid_sides else None

    return dict(
        bone_loss_mesial_pct=mesial['bone_loss_pct'],
        bone_loss_distal_pct=distal['bone_loss_pct'],
        bone_loss_mean_pct=mean_pct,
        mesial_cb_mm=mesial['cb_mm'],
        mesial_ca_mm=mesial['ca_mm'],
        distal_cb_mm=distal['cb_mm'],
        distal_ca_mm=distal['ca_mm'],
        bone_loss_valid=bool(valid_sides),
        bone_loss_invalid_reasons=invalid_reasons,
        bone_loss_mesial_invalid_reason=mesial['invalid_reason'],
        bone_loss_distal_invalid_reason=distal['invalid_reason'],
    )


def augment_panoramic_prediction_payload(payload: Mapping[str, object],
                                         mm_per_pixel: float,
                                         keypoint_score_thr: float = 0.0
                                         ) -> dict:
    augmented = deepcopy(dict(payload))
    panoramic_prediction = dict(augmented.get('panoramic_prediction', {}))
    teeth = list(panoramic_prediction.get('teeth', []))

    augmented_teeth = []
    for tooth in teeth:
        tooth_payload = dict(tooth)
        bone_loss = compute_tooth_bone_loss(
            keypoints=tooth_payload.get('keypoints', []),
            mm_per_pixel=mm_per_pixel,
            keypoint_score_thr=keypoint_score_thr)
        tooth_payload.update(bone_loss)
        augmented_teeth.append(tooth_payload)

    panoramic_prediction['teeth'] = augmented_teeth
    augmented['panoramic_prediction'] = panoramic_prediction
    augmented['bone_loss'] = dict(
        formula='Bone_Loss%=((CB-2)/(CA-2))*100%, if CB-2<0 then 0',
        mm_per_pixel=float(mm_per_pixel),
        keypoint_score_thr=float(keypoint_score_thr),
        aggregation='mean_of_valid_mesial_and_distal',
    )
    return augmented


def iter_bone_loss_rows(payload: Mapping[str, object]) -> Iterable[dict]:
    image = dict(payload.get('image', {}))
    panoramic_prediction = dict(payload.get('panoramic_prediction', {}))
    for tooth in panoramic_prediction.get('teeth', []):
        tooth_payload = dict(tooth)
        yield dict(
            image_name=image.get('file_name'),
            tooth_id=tooth_payload.get('tooth_id'),
            bone_loss_mesial_pct=tooth_payload.get('bone_loss_mesial_pct'),
            bone_loss_distal_pct=tooth_payload.get('bone_loss_distal_pct'),
            bone_loss_mean_pct=tooth_payload.get('bone_loss_mean_pct'),
            mesial_cb_mm=tooth_payload.get('mesial_cb_mm'),
            mesial_ca_mm=tooth_payload.get('mesial_ca_mm'),
            distal_cb_mm=tooth_payload.get('distal_cb_mm'),
            distal_ca_mm=tooth_payload.get('distal_ca_mm'),
            bone_loss_valid=tooth_payload.get('bone_loss_valid'),
            bone_loss_invalid_reasons=';'.join(
                tooth_payload.get('bone_loss_invalid_reasons', [])),
        )
