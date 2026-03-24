from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_ROOT = ROOT / 'tools' / 'analysis_tools'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import panoramic_teeth_export_panoramic_predictions as export_vis


def test_render_panoramic_overlay_draws_mesial_and_distal_lines():
    export_vis.cv2 = cv2
    export_vis.np = np

    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)

    tooth = dict(
        tooth_id=11,
        mask_binary=mask,
        mask_bbox=None,
        keypoints_xy=np.array(
            [[10, 10], [20, 20], [30, 30], [40, 20], [50, 10]],
            dtype=np.float32),
    )
    keypoint_colors = np.array([
        [0, 178, 255],
        [0, 178, 255],
        [255, 214, 10],
        [255, 82, 82],
        [255, 82, 82],
    ],
                              dtype=np.uint8)

    overlay = export_vis.render_panoramic_overlay(
        image=image,
        teeth_predictions=[tooth],
        keypoint_colors=keypoint_colors)

    assert overlay.shape == image.shape
    assert tuple(overlay[15, 15]) != (255, 255, 255)
    assert tuple(overlay[15, 45]) != (255, 255, 255)
