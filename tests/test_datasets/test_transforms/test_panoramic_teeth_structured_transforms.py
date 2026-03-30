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
    ExpandToothBBox, GenerateAnatomicalPointMaskTargets,
    GenerateAnatomicalToothTargets,
    GenerateStructuredToothTargets)


def test_expand_tooth_bbox_adds_anatomical_context():
    transform = ExpandToothBBox(
        left_ratio=0.25,
        right_ratio=0.25,
        top_ratio=0.50,
        bottom_ratio=0.20,
        min_pad_x=10.0,
        min_pad_top=20.0,
        min_pad_bottom=8.0)

    results = dict(
        bbox=np.array([[50.0, 60.0, 100.0, 180.0]], dtype=np.float32),
        img_shape=(256, 256))

    output = transform(results)

    np.testing.assert_allclose(
        output['bbox'][0],
        np.array([37.5, 0.0 + 0.0, 112.5, 204.0], dtype=np.float32),
        atol=1e-5)


def test_expand_tooth_bbox_uses_tooth_id_templates():
    transform = ExpandToothBBox(use_tooth_id_templates=True)

    anterior = transform(
        dict(
            bbox=np.array([[50.0, 60.0, 100.0, 180.0]], dtype=np.float32),
            tooth_id=11,
            img_shape=(512, 512)))
    molar = transform(
        dict(
            bbox=np.array([[50.0, 60.0, 100.0, 180.0]], dtype=np.float32),
            tooth_id=16,
            img_shape=(512, 512)))

    anterior_w = anterior['bbox'][0, 2] - anterior['bbox'][0, 0]
    molar_w = molar['bbox'][0, 2] - molar['bbox'][0, 0]

    assert anterior_w < molar_w


def test_expand_tooth_bbox_shifts_towards_occlusal_midline():
    transform = ExpandToothBBox(use_tooth_id_templates=True)
    base_bbox = np.array([[100.0, 400.0, 160.0, 540.0]], dtype=np.float32)
    height = float(base_bbox[0, 3] - base_bbox[0, 1])

    q1 = transform(
        dict(bbox=base_bbox.copy(), tooth_id=11, img_shape=(4096, 4096)))
    q2 = transform(
        dict(bbox=base_bbox.copy(), tooth_id=21, img_shape=(4096, 4096)))
    q3 = transform(
        dict(bbox=base_bbox.copy(), tooth_id=31, img_shape=(4096, 4096)))
    q4 = transform(
        dict(bbox=base_bbox.copy(), tooth_id=41, img_shape=(4096, 4096)))

    q1_center_y = float((q1['bbox'][0, 1] + q1['bbox'][0, 3]) * 0.5)
    q2_center_y = float((q2['bbox'][0, 1] + q2['bbox'][0, 3]) * 0.5)
    q3_center_y = float((q3['bbox'][0, 1] + q3['bbox'][0, 3]) * 0.5)
    q4_center_y = float((q4['bbox'][0, 1] + q4['bbox'][0, 3]) * 0.5)

    def expanded_center_y_without_shift(tooth_id: int) -> float:
        template = transform._resolve_template(tooth_id)
        pad_top = max(height * template['top_ratio'], template['min_pad_top'])
        pad_bottom = max(height * template['bottom_ratio'],
                         template['min_pad_bottom'])
        return float((base_bbox[0, 1] - pad_top + base_bbox[0, 3] + pad_bottom)
                     * 0.5)

    q1_base_center_y = expanded_center_y_without_shift(11)
    q2_base_center_y = expanded_center_y_without_shift(21)
    q3_base_center_y = expanded_center_y_without_shift(31)
    q4_base_center_y = expanded_center_y_without_shift(41)

    assert q1_center_y > q1_base_center_y
    assert q2_center_y > q2_base_center_y
    assert q3_center_y < q3_base_center_y
    assert q4_center_y < q4_base_center_y


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


def test_generate_anatomical_tooth_targets_outputs_anatomy_fields():
    transform = GenerateAnatomicalToothTargets(
        num_contour_points=16,
        line_width=2,
        use_udp=False,
        point_encoder=dict(
            type='SimCCLabel',
            input_size=(100, 100),
            sigma=(4.0, 4.0),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False))

    results = dict(
        root_polygon=np.array([[25.0, 10.0], [20.0, 75.0], [40.0, 95.0],
                               [60.0, 95.0], [80.0, 75.0], [75.0, 10.0]],
                              dtype=np.float32),
        side_contours=dict(
            M=np.array([[25.0, 10.0], [22.0, 30.0], [24.0, 55.0], [40.0, 95.0]],
                       dtype=np.float32),
            D=np.array([[75.0, 10.0], [78.0, 30.0], [76.0, 55.0], [60.0, 95.0]],
                       dtype=np.float32)),
        apex_midpoint=np.array([50.0, 95.0], dtype=np.float32),
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
    assert output['mesial_anatomy'].shape == (1, 100, 100)
    assert output['distal_anatomy'].shape == (1, 100, 100)
    assert output['mesial_anatomy_distance'].shape == (1, 100, 100)
    assert output['distal_anatomy_distance'].shape == (1, 100, 100)
    assert output['apex_midpoint_target'].shape == (2, )
    assert output['mesial_keypoint_x_labels'].shape == (1, 2, 200)
    assert output['mesial_keypoint_y_labels'].shape == (1, 2, 200)
    assert output['apex_keypoint_x_labels'].shape == (1, 1, 200)
    assert output['apex_keypoint_y_labels'].shape == (1, 1, 200)
    assert output['distal_keypoint_x_labels'].shape == (1, 2, 200)
    assert output['distal_keypoint_y_labels'].shape == (1, 2, 200)


def test_generate_anatomical_pointmask_targets_outputs_point_driven_fields():
    transform = GenerateAnatomicalPointMaskTargets(
        line_width=2,
        use_udp=False,
        point_encoder=dict(
            type='SimCCLabel',
            input_size=(100, 100),
            sigma=(4.0, 4.0),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False))

    results = dict(
        root_polygon=np.array([[25.0, 10.0], [20.0, 75.0], [40.0, 95.0],
                               [60.0, 95.0], [80.0, 75.0], [75.0, 10.0]],
                              dtype=np.float32),
        apex_midpoint=np.array([50.0, 95.0], dtype=np.float32),
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
    assert output['mesial_polyline_map'].shape == (1, 100, 100)
    assert output['distal_polyline_map'].shape == (1, 100, 100)
    assert output['mesial_polyline_distance'].shape == (1, 100, 100)
    assert output['distal_polyline_distance'].shape == (1, 100, 100)
    assert output['apex_midpoint_target'].shape == (2, )
    assert output['mesial_keypoint_x_labels'].shape == (1, 2, 200)
    assert output['apex_keypoint_x_labels'].shape == (1, 1, 200)
    assert output['distal_keypoint_x_labels'].shape == (1, 2, 200)
    assert 'mesial_boundary' not in output
    assert 'mesial_contour' not in output


def test_generate_anatomical_pointmask_targets_with_sideprior_outputs_dense_fields():
    transform = GenerateAnatomicalPointMaskTargets(
        line_width=2,
        use_udp=False,
        use_side_contour_prior=True,
        num_contour_points=16,
        point_encoder=dict(
            type='SimCCLabel',
            input_size=(100, 100),
            sigma=(4.0, 4.0),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False))

    results = dict(
        root_polygon=np.array([[25.0, 10.0], [20.0, 75.0], [40.0, 95.0],
                               [60.0, 95.0], [80.0, 75.0], [75.0, 10.0]],
                              dtype=np.float32),
        side_contours=dict(
            M=np.array([[25.0, 10.0], [22.0, 30.0], [24.0, 55.0], [40.0, 95.0]],
                       dtype=np.float32),
            D=np.array([[75.0, 10.0], [78.0, 30.0], [76.0, 55.0], [60.0, 95.0]],
                       dtype=np.float32)),
        apex_midpoint=np.array([50.0, 95.0], dtype=np.float32),
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
    assert output['mesial_anatomy'].shape == (1, 100, 100)
    assert output['distal_anatomy'].shape == (1, 100, 100)
    assert output['mesial_anatomy_distance'].shape == (1, 100, 100)
    assert output['distal_anatomy_distance'].shape == (1, 100, 100)
    assert output['mesial_distance'].shape == (1, 100, 100)
    assert output['distal_distance'].shape == (1, 100, 100)
    assert output['mesial_contour'].shape == (16, 2)
    assert output['distal_contour'].shape == (16, 2)
