import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.panoramic_teeth.datasets.annotation_utils import (
    build_tooth_instance,
    dedupe_consecutive_points,
)


def test_dedupe_consecutive_points_keeps_order():
    points = [(1, 1), (1, 1), (2, 2), (2, 2), (3, 3)]
    assert dedupe_consecutive_points(points) == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test_build_tooth_instance_derives_keypoints_and_polygon():
    mesial = [(10, 10), (10, 18), (12, 28)]
    distal = [(26, 10), (26, 18), (24, 28)]

    annotation, debug = build_tooth_instance(
        tooth_id=11,
        mesial_points=mesial,
        distal_points=distal,
        image_id=0,
        annotation_id=0,
        image_width=100,
        image_height=100,
    )

    assert annotation is not None
    assert debug['skip_reason'] is None
    assert annotation['tooth_id'] == 11
    assert annotation['num_keypoints'] == 5
    assert len(annotation['segmentation'][0]) >= 8

    keypoints = annotation['keypoints']
    assert keypoints[:6] == [10.0, 10.0, 2.0, 10.0, 18.0, 2.0]
    assert math.isclose(keypoints[6], 18.0)
    assert math.isclose(keypoints[7], 28.0)
    assert keypoints[12:] == [26.0, 10.0, 2.0]

    bbox = annotation['bbox']
    assert bbox[0] <= 10.0
    assert bbox[1] <= 10.0
    assert bbox[2] > 0
    assert bbox[3] > 0
