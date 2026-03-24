from __future__ import annotations

import argparse
import csv
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.panoramic_teeth.bone_loss_utils import (  # noqa: E402
    augment_panoramic_prediction_payload,
    iter_bone_loss_rows,
    parse_stage1_predictions,
)


IMAGE_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tif',
    '.tiff',
    '.JPG',
    '.JPEG',
    '.PNG',
}


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run stage1 tooth instance segmentation, stage2 '
        'panoramic_teeth keypoint/root-mask inference, and bone-loss export.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mm-per-pixel', type=float, required=True)
    parser.add_argument(
        '--seg-project-dir',
        type=Path,
        default=Path('/home/tianruiliu/codespace/data_process/SemiT-SAM'))
    parser.add_argument(
        '--seg-python',
        default='conda run -n gd_env python',
        help='Command prefix used to run the stage1 demo script.')
    parser.add_argument(
        '--seg-config-file',
        default='configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    parser.add_argument(
        '--seg-weights',
        default='pretrained_models/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth')
    parser.add_argument('--seg-device', default='cuda')
    parser.add_argument('--seg-score-threshold', type=float, default=0.0)
    parser.add_argument('--seg-nms-threshold', type=float, default=-1.0)
    parser.add_argument('--seg-max-detections', type=int, default=-1)
    parser.add_argument(
        '--skip-stage1',
        action='store_true',
        help='Reuse an existing stage1 predictions.json instead of rerunning '
        'SemiT-SAM.')
    parser.add_argument(
        '--stage1-summary',
        type=Path,
        default=None,
        help='Optional path to an existing stage1 predictions.json. When '
        '--skip-stage1 is set and this flag is omitted, the script looks for '
        '<output-dir>/stage1_seg/predictions.json.')
    parser.add_argument('--pose-device', default='cuda:0')
    parser.add_argument(
        '--pose-config',
        type=Path,
        default=Path(
            'projects/panoramic_teeth/configs/'
            'rtmpose-dinov3-convnext-s_1xb32-20e_'
            'panoramic-teeth-v2-256x384_stage2_a100.py'))
    parser.add_argument(
        '--pose-checkpoint',
        type=Path,
        default=Path(
            'work_dirs/'
            'rtmpose-dinov3-convnext-s_1xb32-20e_'
            'panoramic-teeth-v2-256x384_stage2_a100/'
            'best_NME_epoch_40.pth'))
    parser.add_argument('--mask-thr', type=float, default=0.5)
    parser.add_argument('--min-contour-area', type=float, default=20.0)
    parser.add_argument('--skip-empty', action='store_true')
    parser.add_argument('--keypoint-score-thr', type=float, default=0.0)
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir()
                  if path.is_file() and path.suffix in IMAGE_EXTENSIONS)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_stage1_segmentation(args: argparse.Namespace, stage1_dir: Path) -> Path:
    seg_demo = args.seg_project_dir / 'demo_infer.py'
    command = shlex.split(args.seg_python) + [
        str(seg_demo),
        '--config-file',
        str(args.seg_config_file),
        '--weights',
        str(args.seg_weights),
        '--input-dir',
        str(args.input_dir),
        '--output-dir',
        str(stage1_dir),
        '--limit',
        '-1',
        '--device',
        args.seg_device,
        '--score-threshold',
        str(args.seg_score_threshold),
        '--nms-threshold',
        str(args.seg_nms_threshold),
        '--max-detections',
        str(args.seg_max_detections),
    ]
    subprocess.run(
        command,
        cwd=str(args.seg_project_dir),
        check=True,
    )
    summary_path = stage1_dir / 'predictions.json'
    if not summary_path.exists():
        raise FileNotFoundError(
            f'Stage1 segmentation summary not found: {summary_path}')
    return summary_path


def resolve_stage1_summary_path(stage1_dir: Path,
                                stage1_summary: Path | None) -> Path:
    if stage1_summary is None:
        return (stage1_dir / 'predictions.json').resolve()
    return stage1_summary.resolve()


def prepare_stage1_summary(args: argparse.Namespace, stage1_dir: Path) -> Path:
    summary_path = resolve_stage1_summary_path(stage1_dir, args.stage1_summary)
    if args.skip_stage1:
        if not summary_path.exists():
            raise FileNotFoundError(
                f'--skip-stage1 was set but stage1 summary was not found: '
                f'{summary_path}')
        return summary_path
    return run_stage1_segmentation(args, stage1_dir)


def load_stage1_mapping(summary_path: Path) -> dict[str, list[dict]]:
    with summary_path.open('r', encoding='utf-8') as f:
        summary = json.load(f)
    return parse_stage1_predictions(summary)


def build_stage2_annotations(instances: Sequence[dict], image_id: int) -> list[dict]:
    annotations = []
    for ann_id, instance in enumerate(instances, start=1):
        annotations.append(
            dict(
                id=ann_id,
                image_id=image_id,
                category_id=1,
                bbox=list(instance['bbox_xywh']),
                tooth_id=int(instance['tooth_id']),
                det_score=float(instance['score']),
            ))
    return annotations


def init_pose_runtime(args: argparse.Namespace):
    from tools.analysis_tools import panoramic_teeth_export_panoramic_predictions as pose_export

    pose_export.import_runtime_deps()
    pose_export.register_all_modules(init_default_scope=False)
    importlib.import_module('projects.panoramic_teeth')

    raw_cfg = pose_export.Config.fromfile(str(args.pose_config))
    cfg, use_udp = pose_export.prepare_cfg(raw_cfg, 'test', {})
    scope = cfg.get('default_scope', 'mmpose')
    if scope is not None:
        pose_export.init_default_scope(scope)
    model = pose_export.init_model(
        cfg, str(args.pose_checkpoint), device=args.pose_device)
    return pose_export, model, use_udp


def build_image_info(pose_export, image_path: Path, image_id: int) -> dict:
    image = pose_export.cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f'Failed to read image from {image_path}')
    image_height, image_width = image.shape[:2]
    return dict(
        id=image_id,
        file_name=image_path.name,
        abs_path=str(image_path.resolve()),
        width=image_width,
        height=image_height,
    )


def build_empty_pose_payload(args: argparse.Namespace, image_info: dict) -> dict:
    return dict(
        image=image_info,
        model=dict(
            config=str(Path(args.pose_config).resolve()),
            checkpoint=str(Path(args.pose_checkpoint).resolve()),
            phase='inference_from_stage1_boxes',
            device=args.pose_device,
            mask_thr=float(args.mask_thr),
        ),
        keypoint_order=['M_C', 'M_B', 'A', 'D_B', 'D_C'],
        panoramic_prediction=dict(
            num_teeth=0,
            union_mask_area=0,
            teeth=[],
        ),
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def save_image(pose_export, path: Path, image) -> None:
    if not pose_export.cv2.imwrite(str(path), image):
        raise RuntimeError(f'Failed to save image to {path}')


def write_summary_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fieldnames = [
        'image_name',
        'tooth_id',
        'bone_loss_mesial_pct',
        'bone_loss_distal_pct',
        'bone_loss_mean_pct',
        'mesial_cb_mm',
        'mesial_ca_mm',
        'distal_cb_mm',
        'distal_ca_mm',
        'bone_loss_valid',
        'bone_loss_invalid_reasons',
    ]
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.seg_project_dir = args.seg_project_dir.resolve()
    args.pose_config = resolve_repo_path(args.pose_config)
    args.pose_checkpoint = resolve_repo_path(args.pose_checkpoint)
    stage1_dir = ensure_dir(output_dir / 'stage1_seg')
    stage2_dir = ensure_dir(output_dir / 'stage2_pose')
    stage2_json_dir = ensure_dir(stage2_dir / 'json')
    stage2_overlay_dir = ensure_dir(stage2_dir / 'overlay')
    stage2_mask_dir = ensure_dir(stage2_dir / 'mask')
    bone_loss_dir = ensure_dir(output_dir / 'bone_loss_json')
    summary_dir = ensure_dir(output_dir / 'summary')

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f'No supported images found in {input_dir}')

    summary_path = prepare_stage1_summary(args, stage1_dir)
    stage1_predictions = load_stage1_mapping(summary_path)
    pose_export, model, use_udp = init_pose_runtime(args)

    csv_rows = []
    pose_args = SimpleNamespace(
        config=str(args.pose_config),
        checkpoint=str(args.pose_checkpoint),
        phase='inference_from_stage1_boxes',
        device=args.pose_device,
        mask_thr=float(args.mask_thr),
        min_contour_area=float(args.min_contour_area),
        skip_empty=bool(args.skip_empty),
    )

    for image_id, image_path in enumerate(image_paths, start=1):
        image_info = build_image_info(pose_export, image_path, image_id)
        instances = stage1_predictions.get(image_path.name, [])
        annotations = build_stage2_annotations(instances, image_id=image_id)

        if annotations:
            pose_payload, overlay, panoramic_mask = (
                pose_export.export_panoramic_prediction(
                    model=model,
                    image_path=image_path,
                    image_info=image_info,
                    annotations=annotations,
                    dataset_meta=model.dataset_meta,
                    args=pose_args,
                    use_udp=use_udp))
        else:
            image = pose_export.cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f'Failed to read image from {image_path}')
            pose_payload = build_empty_pose_payload(args, image_info)
            overlay = image
            panoramic_mask = pose_export.np.zeros(
                (image_info['height'], image_info['width']), dtype='uint8')

        stem = image_path.stem.replace(' ', '_')
        pose_json_path = stage2_json_dir / f'{stem}_panoramic_prediction.json'
        overlay_path = stage2_overlay_dir / f'{stem}_panoramic_overlay.png'
        mask_path = stage2_mask_dir / f'{stem}_panoramic_mask.png'

        save_json(pose_json_path, pose_payload)
        save_image(pose_export, overlay_path, overlay)
        save_image(pose_export, mask_path, panoramic_mask)

        final_payload = augment_panoramic_prediction_payload(
            payload=pose_payload,
            mm_per_pixel=args.mm_per_pixel,
            keypoint_score_thr=args.keypoint_score_thr)
        final_payload['stage1'] = dict(
            predictions_json=str(summary_path),
            num_stage1_instances=len(instances),
        )

        bone_loss_json_path = bone_loss_dir / f'{stem}_bone_loss.json'
        save_json(bone_loss_json_path, final_payload)
        csv_rows.extend(iter_bone_loss_rows(final_payload))

        print(
            f'{image_path.name}: stage1={len(instances)} teeth, '
            f'stage2={final_payload["panoramic_prediction"]["num_teeth"]} teeth, '
            f'json={bone_loss_json_path}')

    csv_path = summary_dir / 'teeth_bone_loss.csv'
    write_summary_csv(csv_path, csv_rows)
    print(f'Stage1 summary: {summary_path}')
    print(f'Bone-loss CSV: {csv_path}')


if __name__ == '__main__':
    main()
