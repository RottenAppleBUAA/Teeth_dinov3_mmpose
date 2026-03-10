from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_ROOT = ROOT / 'tools' / 'analysis_tools'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import panoramic_teeth_visualize_predictions as vis


def test_make_visualization_panel_returns_expected_size():
    image = np.full((64, 48, 3), 120, dtype=np.uint8)
    gt_mask = np.zeros((64, 48), dtype=np.float32)
    gt_mask[8:28, 10:24] = 1.0
    pred_mask = np.zeros((64, 48), dtype=np.float32)
    pred_mask[16:36, 18:30] = 0.9

    gt_keypoints = np.array([[12, 12], [18, 18], [22, 26], [28, 18], [34, 12]],
                            dtype=np.float32)
    pred_keypoints = gt_keypoints + 2.0
    pred_scores = np.full((5, ), 0.8, dtype=np.float32)
    gt_visible = np.ones((5, ), dtype=np.float32)
    labels = ['M_C', 'M_B', 'A', 'D_B', 'D_C']
    colors = np.array([
        [0, 178, 255],
        [0, 178, 255],
        [255, 214, 10],
        [255, 82, 82],
        [255, 82, 82],
    ],
                      dtype=np.uint8)
    skeleton_links = [(0, 1), (1, 2), (2, 3), (3, 4)]

    panel = vis.make_visualization_panel(
        image=image,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        gt_keypoints=gt_keypoints,
        pred_keypoints=pred_keypoints,
        keypoint_labels=labels,
        keypoint_colors=colors,
        skeleton_links=skeleton_links,
        pred_scores=pred_scores,
        gt_visible=gt_visible,
        mask_thr=0.5,
        pred_kpt_thr=0.2)

    assert panel.shape == (64, 48 * 4, 3)
    assert panel.dtype == np.uint8
    assert np.any(panel != 120)
