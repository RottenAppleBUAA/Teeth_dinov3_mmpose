import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_ROOT = ROOT / 'tools' / 'dataset_converters'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import panoramic_teeth_annotation_quality_report as quality_report


def test_extract_blue_points_detects_blue_pixels(tmp_path):
    image_path = tmp_path / 'blue.png'
    image = np.full((6, 6, 3), 255, dtype=np.uint8)
    image[1, 2] = (20, 40, 230)
    image[4, 3] = (79, 79, 210)
    image[5, 5] = (81, 10, 220)
    Image.fromarray(image).save(image_path)

    points = quality_report.extract_blue_points(
        image_path,
        blue_threshold=100,
        red_max=80,
        green_max=80,
    )

    assert (2, 1) in points
    assert (3, 4) in points
    assert (5, 5) not in points
    assert len(points) == 2


def test_nearest_point_offset_returns_best_match():
    bucket_index = quality_report.build_bucket_index([(10, 10), (15, 14)], 8)

    match = quality_report.nearest_point_offset((13.0, 12.0), bucket_index, 8,
                                                10.0)

    assert match is not None
    assert round(match[0], 2) == 2.0
    assert round(match[1], 2) == 2.0
