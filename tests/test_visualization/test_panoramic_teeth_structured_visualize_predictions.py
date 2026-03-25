from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_ROOT = ROOT / 'tools' / 'analysis_tools'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import panoramic_teeth_structured_visualize_predictions as vis


def test_make_image_level_visualization_panel_returns_expected_size():
    image = np.full((64, 48, 3), 120, dtype=np.uint8)
    mask = np.zeros((64, 48), dtype=np.uint8)
    mask[10:30, 10:20] = 1

    gt_records = [
        dict(
            annotation_id=1,
            tooth_id=11,
            bbox_xywh=[10, 10, 20, 30],
            keypoints_xy=np.array(
                [[12, 12], [15, 20], [18, 28], [24, 20], [28, 12]],
                dtype=np.float32),
            keypoint_scores=np.ones((5, ), dtype=np.float32),
            mesial_contour=np.array([[12, 12], [15, 20], [18, 28]],
                                    dtype=np.float32),
            distal_contour=np.array([[28, 12], [24, 20], [18, 28]],
                                    dtype=np.float32),
            mask_binary=mask)
    ]
    pred_records = [
        dict(
            annotation_id=1,
            tooth_id=11,
            bbox_xywh=[10, 10, 20, 30],
            keypoints_xy=np.array(
                [[13, 12], [16, 20], [18, 27], [23, 20], [27, 12]],
                dtype=np.float32),
            keypoint_scores=np.full((5, ), 0.9, dtype=np.float32),
            mesial_contour=np.array([[13, 12], [16, 20], [18, 27]],
                                    dtype=np.float32),
            distal_contour=np.array([[27, 12], [23, 20], [18, 27]],
                                    dtype=np.float32),
            mask_binary=mask)
    ]
    names = ['M_C', 'M_B', 'A', 'D_B', 'D_C']
    colors = [(255, 178, 0), (255, 178, 0), (10, 214, 255), (82, 82, 255),
              (82, 82, 255)]

    panel = vis.make_image_level_visualization_panel(
        image=image,
        gt_records=gt_records,
        pred_records=pred_records,
        keypoint_names=names,
        semantic_colors=colors,
        pred_kpt_thr=0.2)

    assert panel.shape == (64 * 2, 48 * 2, 3)
    assert panel.dtype == np.uint8
    assert np.any(panel != 120)


def test_load_annotation_bundle_and_match_image_info(tmp_path):
    ann_path = tmp_path / 'ann.json'
    image_path = tmp_path / 'images' / '001.png'
    image_path.parent.mkdir()
    image_path.write_bytes(b'')
    ann_path.write_text(
        json.dumps(
            dict(
                images=[dict(id=1, file_name='images/001.png', width=100,
                             height=50)],
                annotations=[dict(id=7, image_id=1, tooth_id=11)]),
            ensure_ascii=False),
        encoding='utf-8')

    bundle = vis.load_annotation_bundle(ann_path)
    matched = vis.match_image_info(image_path, tmp_path,
                                   bundle['images'].values())

    assert bundle['anns_by_image'][1][0]['id'] == 7
    assert matched['id'] == 1
