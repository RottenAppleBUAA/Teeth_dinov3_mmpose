from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import panoramic_teeth_annotation_quality_report as quality_report  # noqa: E402
import panoramic_teeth_excel_to_coco as legacy  # noqa: E402
import panoramic_teeth_excel_to_coco_v2 as converter_v2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export debug views for panoramic-teeth alignment review.')
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('datasets/586份数据20260116'),
        help='Root directory containing Excel files, original images and '
        'traced images.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory used to store generated debug views. Defaults to '
        '<dataset-root>/debug_views_all.')
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--source-canvas-width', type=int, default=None)
    parser.add_argument('--source-canvas-height', type=int, default=512)
    parser.add_argument('--point-offset-x', type=float, default=0.0)
    parser.add_argument('--point-offset-y', type=float, default=0.0)
    parser.add_argument('--blue-threshold', type=int, default=100)
    parser.add_argument('--red-max', type=int, default=80)
    parser.add_argument('--green-max', type=int, default=80)
    parser.add_argument(
        '--sample-list-json',
        type=Path,
        default=None,
        help='Optional JSON file containing sample ids to keep.')
    parser.add_argument('--skip-original', action='store_true')
    parser.add_argument('--skip-blue-pixels', action='store_true')
    parser.add_argument('--skip-composite', action='store_true')
    return parser.parse_args()


def build_sample_ids(dataset_root: Path) -> Tuple[Path, Path, Path, List[int]]:
    excel_dir = legacy.detect_subdir(dataset_root, ['excel', 'Excel'])
    image_dir = legacy.detect_subdir(dataset_root, ['原图'])
    traced_dir = legacy.detect_subdir(dataset_root, ['描点图'])

    excel_ids = {
        legacy.extract_sample_id(path)
        for path in sorted(excel_dir.glob('*.xlsx'))
        if not path.name.startswith('.~') and not path.name.startswith('~$')
    }
    image_ids = {
        legacy.extract_sample_id(path)
        for path in sorted(image_dir.glob('*')) if path.is_file()
    }
    traced_ids = {
        legacy.extract_sample_id(path)
        for path in sorted(traced_dir.glob('*')) if path.is_file()
    }
    return excel_dir, image_dir, traced_dir, sorted(excel_ids & image_ids & traced_ids)


def draw_blue_pixel_overlay(size: Tuple[int, int],
                            blue_points: Sequence[Tuple[int, int]]) -> Image.Image:
    canvas = Image.new('RGBA', size, (0, 0, 0, 0))
    pixels = canvas.load()
    for x, y in blue_points:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            pixels[x, y] = (0, 178, 255, 220)
    return canvas


def draw_blue_pixel_image(size: Tuple[int, int],
                          blue_points: Sequence[Tuple[int, int]]) -> Image.Image:
    canvas = Image.new('RGB', size, (0, 0, 0))
    pixels = canvas.load()
    for x, y in blue_points:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            pixels[x, y] = (0, 178, 255)
    return canvas


def draw_annotation_overlay(record: legacy.ImageRecord) -> Image.Image:
    canvas = Image.new('RGBA', (record.width, record.height), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(canvas)
    colors = {
        'M': (0, 178, 255, 255),
        'D': (255, 82, 82, 255),
        'AL': (255, 214, 10, 255),
        'TEXT': (255, 255, 255, 255),
    }

    scratch_summary: DefaultDict[str, int] = defaultdict(int)
    for tooth_id in legacy.TARGET_TEETH:
        resolved = legacy.resolve_entries_for_tooth(
            tooth_id, record.side_entries.get(tooth_id, []), scratch_summary)
        label_point = None
        for side_name in ('M', 'D'):
            entry = resolved.get(side_name)
            if entry is None:
                continue
            converted_points, visibility = legacy.convert_side_entry(entry)
            visible_points: List[Tuple[float, float]] = []
            for index, (point, point_visibility) in enumerate(
                    zip(converted_points, visibility)):
                if point_visibility <= 0:
                    continue
                x, y = float(point[0]), float(point[1])
                if label_point is None:
                    label_point = (x, y)
                color = colors['AL'] if index == 5 else colors[side_name]
                radius = 4 if index == 5 else 3
                drawer.ellipse([x - radius, y - radius, x + radius, y + radius],
                               fill=color, outline=color)
                if index < 5:
                    visible_points.append((x, y))
            if len(visible_points) >= 2:
                drawer.line(visible_points, fill=colors[side_name], width=2)
        if label_point is not None:
            drawer.text((label_point[0] + 4, label_point[1] - 10), str(tooth_id),
                        fill=colors['TEXT'])
    return canvas


def draw_annotation_image(record: legacy.ImageRecord) -> Image.Image:
    overlay = draw_annotation_overlay(record)
    canvas = Image.new('RGBA', (record.width, record.height), (0, 0, 0, 255))
    return Image.alpha_composite(canvas, overlay).convert('RGB')


def draw_composite_image(record: legacy.ImageRecord,
                         blue_points: Sequence[Tuple[int, int]]) -> Image.Image:
    with Image.open(record.image_path) as image:
        base = image.convert('RGBA')
    blue_overlay = draw_blue_pixel_overlay((record.width, record.height),
                                           blue_points)
    annotation_overlay = draw_annotation_overlay(record)
    return Image.alpha_composite(
        Image.alpha_composite(base, blue_overlay), annotation_overlay).convert('RGB')


def copy_original_image(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir or dataset_root / 'debug_views_all'
    excel_dir, image_dir, traced_dir, sample_ids = build_sample_ids(dataset_root)
    if args.sample_list_json is not None:
        allowed_sample_ids = set(
            converter_v2.load_sample_ids_from_json(args.sample_list_json))
        sample_ids = [sample_id for sample_id in sample_ids
                      if sample_id in allowed_sample_ids]
    if args.sample_limit is not None:
        sample_ids = sample_ids[:args.sample_limit]

    summary: DefaultDict[str, int] = defaultdict(int)
    records, _, _ = converter_v2.build_records(
        dataset_root,
        excel_dir,
        image_dir,
        traced_dir,
        sample_ids,
        summary,
        args.source_canvas_width,
        args.source_canvas_height,
        args.point_offset_x,
        args.point_offset_y)

    original_dir = output_dir / 'original'
    blue_dir = output_dir / 'blue_pixels'
    annotation_dir = output_dir / 'annotations'
    composite_dir = output_dir / 'composites'
    manifest_rows: List[dict] = []

    for sample_id in sample_ids:
        record = records[sample_id]
        original_path = original_dir / f'sample_{sample_id}_original{record.image_path.suffix}'
        blue_path = blue_dir / f'sample_{sample_id}_blue_pixels.png'
        annotation_path = annotation_dir / f'sample_{sample_id}_annotation.png'
        composite_path = composite_dir / f'sample_{sample_id}_composite.png'

        if not args.skip_original:
            copy_original_image(record.image_path, original_path)

        blue_points = quality_report.extract_blue_points(
            record.traced_image_path,
            args.blue_threshold,
            args.red_max,
            args.green_max,
        ) if record.traced_image_path else []
        if not args.skip_blue_pixels:
            blue_image = draw_blue_pixel_image((record.width, record.height),
                                               blue_points)
            blue_path.parent.mkdir(parents=True, exist_ok=True)
            blue_image.save(blue_path)

        annotation_image = draw_annotation_image(record)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_image.save(annotation_path)

        if not args.skip_composite:
            composite_image = draw_composite_image(record, blue_points)
            composite_path.parent.mkdir(parents=True, exist_ok=True)
            composite_image.save(composite_path)

        manifest_rows.append({
            'sample_id': sample_id,
            'original_image': '' if args.skip_original else str(original_path),
            'annotation_image': str(annotation_path),
            'blue_pixel_image': '' if args.skip_blue_pixels else str(blue_path),
            'composite_image': '' if args.skip_composite else str(composite_path),
            'source_original_image': str(record.image_path),
            'source_traced_image': str(record.traced_image_path)
            if record.traced_image_path else '',
            'image_width': record.width,
            'image_height': record.height,
            'num_blue_pixels': len(blue_points),
        })

    summary_payload = {
        'dataset_root': str(dataset_root),
        'output_dir': str(output_dir),
        'num_samples': len(sample_ids),
        'sample_list_json': str(args.sample_list_json)
        if args.sample_list_json else None,
        'source_canvas_width': args.source_canvas_width,
        'source_canvas_height': args.source_canvas_height,
        'point_offset_x': args.point_offset_x,
        'point_offset_y': args.point_offset_y,
        'blue_threshold': args.blue_threshold,
        'red_max': args.red_max,
        'green_max': args.green_max,
        'skip_original': args.skip_original,
        'skip_blue_pixels': args.skip_blue_pixels,
        'skip_composite': args.skip_composite,
    }
    write_json(output_dir / 'summary.json', summary_payload)
    write_csv(output_dir / 'manifest.csv', manifest_rows)
    print(f'Wrote debug views for {len(sample_ids)} samples to {output_dir}')


if __name__ == '__main__':
    main()
