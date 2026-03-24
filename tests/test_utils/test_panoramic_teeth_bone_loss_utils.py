from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.panoramic_teeth.bone_loss_utils import (  # noqa: E402
    augment_panoramic_prediction_payload,
    class_id_to_tooth_id,
    compute_side_bone_loss,
    compute_tooth_bone_loss,
    iter_bone_loss_rows,
    parse_stage1_predictions,
)


def _keypoints():
    return [
        dict(name='M_C', x=0.0, y=0.0, score=0.9),
        dict(name='M_B', x=0.0, y=6.0, score=0.9),
        dict(name='A', x=0.0, y=10.0, score=0.9),
        dict(name='D_B', x=10.0, y=4.0, score=0.9),
        dict(name='D_C', x=10.0, y=0.0, score=0.9),
    ]


def test_class_id_to_tooth_id_uses_expected_order():
    assert class_id_to_tooth_id(0) == 1
    assert class_id_to_tooth_id(27) == 18
    assert class_id_to_tooth_id(31) == 31


def test_parse_stage1_predictions_keeps_best_instance_per_tooth():
    parsed = parse_stage1_predictions([
        dict(
            image='case_001.png',
            instances=[
                dict(bbox_xyxy=[0, 0, 10, 12], score=0.3, class_id=0),
                dict(bbox_xyxy=[1, 2, 11, 14], score=0.8, class_id=0),
                dict(bbox_xyxy=[20, 0, 28, 10], score=0.5, class_id=27),
            ])
    ])

    assert list(parsed) == ['case_001.png']
    instances = parsed['case_001.png']
    assert [item['tooth_id'] for item in instances] == [1, 18]
    assert instances[0]['score'] == 0.8
    assert instances[0]['bbox_xywh'] == [1.0, 2.0, 10.0, 12.0]


def test_compute_side_bone_loss_uses_formula_with_two_millimeter_offset():
    result = compute_side_bone_loss(
        keypoints=_keypoints(),
        side='mesial',
        mm_per_pixel=1.0,
    )

    assert result['valid'] is True
    assert math.isclose(result['cb_mm'], 6.0)
    assert math.isclose(result['ca_mm'], 10.0)
    assert math.isclose(result['bone_loss_pct'], 50.0)


def test_compute_side_bone_loss_returns_zero_when_cb_below_two():
    keypoints = _keypoints()
    keypoints[1] = dict(name='M_B', x=0.0, y=1.5, score=0.9)

    result = compute_side_bone_loss(
        keypoints=keypoints,
        side='mesial',
        mm_per_pixel=1.0,
    )

    assert result['valid'] is True
    assert math.isclose(result['cb_mm'], 1.5)
    assert result['bone_loss_pct'] == 0.0


def test_compute_side_bone_loss_marks_non_positive_denominator_invalid():
    keypoints = _keypoints()
    keypoints[2] = dict(name='A', x=0.0, y=2.0, score=0.9)

    result = compute_side_bone_loss(
        keypoints=keypoints,
        side='mesial',
        mm_per_pixel=1.0,
    )

    assert result['valid'] is False
    assert result['bone_loss_pct'] is None
    assert result['invalid_reason'] == 'denominator_non_positive'


def test_compute_tooth_bone_loss_averages_valid_mesial_and_distal():
    result = compute_tooth_bone_loss(
        keypoints=_keypoints(),
        mm_per_pixel=1.0,
    )

    assert math.isclose(result['bone_loss_mesial_pct'], 50.0)
    assert math.isclose(result['bone_loss_distal_pct'], 16.471566962990767)
    assert math.isclose(result['bone_loss_mean_pct'], 33.23578348149538)
    assert result['bone_loss_valid'] is True
    assert result['bone_loss_invalid_reasons'] == []


def test_compute_tooth_bone_loss_honors_keypoint_score_threshold():
    keypoints = _keypoints()
    keypoints[3] = dict(name='D_B', x=10.0, y=4.0, score=0.2)

    result = compute_tooth_bone_loss(
        keypoints=keypoints,
        mm_per_pixel=1.0,
        keypoint_score_thr=0.5,
    )

    assert math.isclose(result['bone_loss_mesial_pct'], 50.0)
    assert result['bone_loss_distal_pct'] is None
    assert math.isclose(result['bone_loss_mean_pct'], 50.0)
    assert result['bone_loss_invalid_reasons'] == ['distal:low_keypoint_score']


def test_augment_payload_and_iter_rows_emit_expected_fields():
    payload = dict(
        image=dict(file_name='case_001.png'),
        panoramic_prediction=dict(
            num_teeth=1,
            teeth=[
                dict(
                    tooth_id=11,
                    keypoints=_keypoints(),
                )
            ],
        ),
    )

    augmented = augment_panoramic_prediction_payload(
        payload=payload,
        mm_per_pixel=1.0,
        keypoint_score_thr=0.0,
    )
    rows = list(iter_bone_loss_rows(augmented))

    assert augmented['bone_loss']['mm_per_pixel'] == 1.0
    assert len(rows) == 1
    assert rows[0]['image_name'] == 'case_001.png'
    assert rows[0]['tooth_id'] == 11
    assert math.isclose(rows[0]['bone_loss_mean_pct'], 33.23578348149538)
