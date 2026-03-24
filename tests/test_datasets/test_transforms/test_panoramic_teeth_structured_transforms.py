import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mmcv')
pytest.importorskip('mmengine')

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.panoramic_teeth_structured.datasets.transforms import (  # noqa: E402
    GenerateStructuredToothTargets, )


def test_generate_structured_tooth_targets_outputs_expected_fields():
    transform = GenerateStructuredToothTargets(
        num_contour_points=16, line_width=2, use_udp=False)

    results = dict(
        root_polygon=np.array([[25.0, 10.0], [20.0, 75.0], [40.0, 95.0],
                               [60.0, 95.0], [80.0, 75.0], [75.0, 10.0]],
                              dtype=np.float32),
        side_contours=dict(
            M=np.array([[25.0, 10.0], [22.0, 30.0], [24.0, 55.0], [40.0, 95.0]],
                       dtype=np.float32),
            D=np.array([[75.0, 10.0], [78.0, 30.0], [76.0, 55.0], [60.0, 95.0]],
                       dtype=np.float32)),
        img_shape=(100, 100),
        input_center=np.array([50.0, 50.0], dtype=np.float32),
        input_scale=np.array([100.0, 100.0], dtype=np.float32),
        input_size=(100, 100),
        bbox_rotation=np.array([0.0], dtype=np.float32),
        transformed_keypoints=np.array([[[25.0, 10.0], [22.0, 30.0],
                                         [50.0, 95.0], [78.0, 30.0],
                                         [75.0, 10.0]]],
                                       dtype=np.float32),
        keypoints_visible=np.ones((1, 5, 1), dtype=np.float32))

    output = transform(results)

    assert output is not None
    assert output['root_mask'].shape == (1, 100, 100)
    assert output['mesial_boundary'].shape == (1, 100, 100)
    assert output['distal_boundary'].shape == (1, 100, 100)
    assert output['mesial_distance'].shape == (1, 100, 100)
    assert output['distal_distance'].shape == (1, 100, 100)
    assert output['mesial_contour'].shape == (16, 2)
    assert output['distal_contour'].shape == (16, 2)
    assert output['keypoint_target'].shape == (5, 2)
    assert output['apex_target'].shape == (2, )
    np.testing.assert_allclose(output['keypoint_target'],
                               results['transformed_keypoints'][0],
                               atol=1e-5)
    np.testing.assert_allclose(
        output['apex_target'],
        0.5 *
        (output['mesial_contour'][-1] + output['distal_contour'][-1]),
        atol=1e-5)

