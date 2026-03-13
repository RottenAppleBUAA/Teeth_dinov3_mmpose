from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = (
    REPO_ROOT / 'tools' / 'analysis_tools' /
    'panoramic_teeth_export_panoramic_predictions.py')
DEFAULT_CONFIG = (
    REPO_ROOT / 'projects' / 'panoramic_teeth' / 'configs' /
    'rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_'
    'stage1_a100.py')
DEFAULT_CHECKPOINT = (
    REPO_ROOT / 'work_dirs' /
    'rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_'
    'stage1_a100' / 'best_NME_epoch_110.pth')
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / 'work_dirs' /
    'rtmpose-dinov3-convnext-s_1xb64-60e_panoramic-teeth-v2-256x384_'
    'stage1_a100' / 'panoramic_exports')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run panoramic-level export with the fixed stage1 A100 '
        'panoramic_teeth checkpoint.')
    parser.add_argument(
        '--image',
        default=None,
        help='Optional image path. If omitted, the script opens file '
        'selection interactively.')
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Output directory for panoramic JSON/mask/overlay results.')
    parser.add_argument(
        '--phase',
        default='auto',
        choices=['auto', 'train', 'val', 'test'],
        help='Dataset split used to look up the image annotations. Use auto '
        'to search train/val/test automatically.')
    parser.add_argument(
        '--ann-file',
        default=None,
        help='Optional COCO annotation file override.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Inference device, for example cuda:0 or cpu.')
    parser.add_argument(
        '--mask-thr',
        type=float,
        default=0.5,
        help='Threshold used to binarize the panoramic mask.')
    parser.add_argument(
        '--min-contour-area',
        type=float,
        default=20.0,
        help='Drop tiny contours smaller than this area when exporting '
        'segmentation polygons.')
    parser.add_argument(
        '--skip-empty',
        action='store_true',
        help='Skip teeth whose inverse-warped mask is empty.')
    parser.add_argument(
        '--cfg-options',
        nargs='*',
        default=[],
        help='Extra config overrides in key=value form.')
    return parser.parse_args()


def ensure_paths():
    if not EXPORT_SCRIPT.exists():
        raise FileNotFoundError(f'Export script not found: {EXPORT_SCRIPT}')
    if not DEFAULT_CONFIG.exists():
        raise FileNotFoundError(f'Config not found: {DEFAULT_CONFIG}')
    if not DEFAULT_CHECKPOINT.exists():
        raise FileNotFoundError(f'Checkpoint not found: {DEFAULT_CHECKPOINT}')


def main():
    args = parse_args()
    ensure_paths()

    command = [
        sys.executable,
        str(EXPORT_SCRIPT),
        str(DEFAULT_CONFIG),
        str(DEFAULT_CHECKPOINT),
        '--output-dir',
        str(Path(args.output_dir).expanduser().resolve()),
        '--phase',
        args.phase,
        '--device',
        args.device,
        '--mask-thr',
        str(args.mask_thr),
        '--min-contour-area',
        str(args.min_contour_area),
    ]

    if args.image:
        command.extend(['--image', str(Path(args.image).expanduser().resolve())])
    if args.ann_file:
        command.extend(['--ann-file', str(Path(args.ann_file).expanduser().resolve())])
    if args.skip_empty:
        command.append('--skip-empty')
    if args.cfg_options:
        command.append('--cfg-options')
        command.extend(args.cfg_options)

    raise SystemExit(subprocess.call(command, cwd=str(REPO_ROOT)))


if __name__ == '__main__':
    main()
