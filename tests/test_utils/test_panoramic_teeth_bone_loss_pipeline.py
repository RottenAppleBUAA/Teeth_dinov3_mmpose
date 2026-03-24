from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis_tools.panoramic_teeth_bone_loss_pipeline import (  # noqa: E402
    resolve_stage1_summary_path,
)


def test_resolve_stage1_summary_path_defaults_to_output_stage1_dir():
    stage1_dir = Path('/tmp/panoramic_teeth/stage1_seg')
    summary_path = resolve_stage1_summary_path(stage1_dir, None)
    assert summary_path == (stage1_dir / 'predictions.json').resolve()


def test_resolve_stage1_summary_path_uses_explicit_path():
    explicit = Path('/tmp/custom_predictions.json')
    summary_path = resolve_stage1_summary_path(
        Path('/tmp/panoramic_teeth/stage1_seg'), explicit)
    assert summary_path == explicit.resolve()
