from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import panoramic_teeth_excel_to_coco as legacy  # noqa: E402
from projects.panoramic_teeth.datasets.annotation_utils import build_tooth_instance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert panoramic-teeth Excel annotations to v2 '
        'single-tooth COCO JSON with 5 keypoints and root polygons.')
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('datasets/586份数据20260116'),
        help='Root directory containing Excel files and images.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory used to store generated annotation files. '
        'Defaults to <dataset-root>/annotations_v2.')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--visualize-count', type=int, default=50)
    parser.add_argument('--skip-visualization', action='store_true')
    parser.add_argument('--skip-split', action='store_true')
    parser.add_argument('--image-subdir', type=Path, default=None)
    parser.add_argument('--excel-subdir', type=Path, default=None)
    parser.add_argument('--traced-subdir', type=Path, default=None)
    parser.add_argument('--source-canvas-width', type=int, default=None)
    parser.add_argument('--source-canvas-height', type=int, default=512)
    parser.add_argument(
        '--point-offset-x',
        type=float,
        default=None,
        help='Explicit global x offset applied after scaling. When omitted, '
        'v2 export estimates it from traced images.')
    parser.add_argument(
        '--point-offset-y',
        type=float,
        default=None,
        help='Explicit global y offset applied after scaling. When omitted, '
        'v2 export estimates it from traced images.')
    parser.add_argument(
        '--quality-report-json',
        type=Path,
        default=None,
        help='Optional sample_metrics.json from the annotation quality report. '
        'Used together with --min-match-ratio to filter samples.')
    parser.add_argument(
        '--min-match-ratio',
        type=float,
        default=None,
        help='Keep only samples whose quality-report match_ratio is greater '
        'than or equal to this threshold.')
    parser.add_argument(
        '--sample-list-json',
        type=Path,
        default=None,
        help='Optional JSON file containing sample ids to keep. Supports '
        '{"sample_ids": [...]} or {"samples": [{"sample_id": ...}, ...]}.')
    return parser.parse_args()


def coco_categories() -> List[dict]:
    return [{
        'supercategory': 'tooth',
        'id': 1,
        'name': 'panoramic_teeth_root_v2',
        'keypoints': ['M_C', 'M_B', 'A', 'D_B', 'D_C'],
        'skeleton': [[1, 2], [2, 3], [3, 4], [4, 5]],
    }]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Object of type {type(value)!r} is not JSON serializable')


def build_dataset_payload(images: List[dict], annotations: List[dict],
                          split_name: str, summary: dict) -> dict:
    return {
        'info': {
            'year': datetime.now().year,
            'version': '2.0',
            'description': 'Panoramic-teeth single-tooth dataset with 5 '
            'keypoints and root polygon mask derived from Excel PointList',
            'contributor': 'Codex',
            'date_created': datetime.now().isoformat(timespec='seconds'),
            'split': split_name,
            'summary': summary,
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': coco_categories(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default)


def initial_summary(dataset_root: Path, excel_dir: Path, image_dir: Path,
                    traced_dir: Path, args: argparse.Namespace) -> dict:
    return {
        'dataset_root': str(dataset_root),
        'excel_dir': str(excel_dir),
        'image_dir': str(image_dir),
        'traced_dir': str(traced_dir),
        'source_canvas_width': args.source_canvas_width,
        'source_canvas_height': args.source_canvas_height,
        'quality_report_json': str(args.quality_report_json)
        if args.quality_report_json else None,
        'sample_list_json': str(args.sample_list_json)
        if args.sample_list_json else None,
        'min_match_ratio': args.min_match_ratio,
        'num_samples_filtered_out_by_sample_list': 0,
        'excluded_sample_ids_by_sample_list': [],
        'num_samples_filtered_out_by_match_ratio': 0,
        'excluded_sample_ids_by_match_ratio': [],
        'point_offset_x': 0.0,
        'point_offset_y': 0.0,
        'estimated_point_offset_x': 0.0,
        'estimated_point_offset_y': 0.0,
        'alignment_median_px_before': 0.0,
        'alignment_median_px_after': 0.0,
        'alignment_matched_points_before': 0,
        'alignment_matched_points_after': 0,
        'empty_row_after_parse': 0,
        'entry_with_invalid_tooth': 0,
        'unknown_label_format': 0,
        'null_label': 0,
        'segment_without_label': 0,
        'segment_without_pointlist': 0,
        'pointlist_parse_error': 0,
        'duplicate_manual_side': 0,
        'unknown_side_single_fill': 0,
        'discarded_extra_side': 0,
        'multi_candidate_tooth': 0,
        'ambiguous_tooth_label': 0,
        'single_sided_unpaired_tooth': 0,
        'invalid_tooth_number_pair': 0,
        'missing_required_side': 0,
        'too_few_points': 0,
        'incomplete_side_contour': 0,
        'invalid_polygon': 0,
        'ambiguous_repaired_polygon': 0,
        'polygon_area_too_small': 0,
        'polygon_vertices_too_few': 0,
        'snapped_keypoints': 0,
        'instances_written': 0,
    }


def load_quality_filtered_sample_ids(
    sample_ids: Sequence[int],
    quality_report_json: Path,
    min_match_ratio: float,
) -> tuple[List[int], List[int]]:
    with quality_report_json.open('r', encoding='utf-8') as file:
        payload = json.load(file)

    samples = payload.get('samples')
    if not isinstance(samples, list):
        raise ValueError(
            f'Invalid quality report format: missing "samples" list in '
            f'{quality_report_json}')

    match_ratio_by_id: Dict[int, float] = {}
    for item in samples:
        sample_id = item.get('sample_id')
        match_ratio = item.get('match_ratio')
        if sample_id is None or match_ratio is None:
            continue
        match_ratio_by_id[int(sample_id)] = float(match_ratio)

    kept_sample_ids: List[int] = []
    excluded_sample_ids: List[int] = []
    for sample_id in sample_ids:
        match_ratio = match_ratio_by_id.get(sample_id)
        if match_ratio is None or match_ratio < min_match_ratio:
            excluded_sample_ids.append(sample_id)
            continue
        kept_sample_ids.append(sample_id)
    return kept_sample_ids, excluded_sample_ids


def load_sample_ids_from_json(sample_list_json: Path) -> List[int]:
    with sample_list_json.open('r', encoding='utf-8') as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        if isinstance(payload.get('sample_ids'), list):
            return sorted({int(sample_id) for sample_id in payload['sample_ids']})
        if isinstance(payload.get('samples'), list):
            sample_ids = []
            for item in payload['samples']:
                if not isinstance(item, dict) or 'sample_id' not in item:
                    continue
                sample_ids.append(int(item['sample_id']))
            return sorted(set(sample_ids))
    raise ValueError(
        f'Invalid sample list format in {sample_list_json}. Expected '
        '{"sample_ids": [...]} or {"samples": [{"sample_id": ...}, ...]}.')


def has_explicit_tooth_and_side(entry: legacy.RawEntry) -> bool:
    return (entry.tooth_id_from_label and entry.side_from_label
            and entry.side_token in {'M', 'D', 'left', 'right'})


def canonical_side_for_entry(tooth_id: int,
                             entry: legacy.RawEntry) -> Optional[str]:
    if entry.side_token in {'M', 'D'}:
        return entry.side_token
    if entry.side_token in {'left', 'right'}:
        return legacy.image_side_to_md(tooth_id, entry.side_token)
    return None


def resolve_strict_pair_entries(
    tooth_id: int,
    entries: Sequence[legacy.RawEntry],
    summary: dict,
) -> Optional[Dict[str, legacy.RawEntry]]:
    if not entries:
        summary['single_sided_unpaired_tooth'] = (
            summary.get('single_sided_unpaired_tooth', 0) + 1)
        return None

    if any(not has_explicit_tooth_and_side(entry) for entry in entries):
        summary['ambiguous_tooth_label'] = (
            summary.get('ambiguous_tooth_label', 0) + 1)
        return None

    grouped_entries: Dict[str, List[legacy.RawEntry]] = {'M': [], 'D': []}
    for entry in entries:
        if entry.tooth_id != tooth_id:
            summary['invalid_tooth_number_pair'] = (
                summary.get('invalid_tooth_number_pair', 0) + 1)
            return None
        canonical_side = canonical_side_for_entry(tooth_id, entry)
        if canonical_side is None:
            summary['ambiguous_tooth_label'] = (
                summary.get('ambiguous_tooth_label', 0) + 1)
            return None
        grouped_entries[canonical_side].append(entry)

    if not grouped_entries['M'] or not grouped_entries['D']:
        summary['single_sided_unpaired_tooth'] = (
            summary.get('single_sided_unpaired_tooth', 0) + 1)
        return None

    resolved = {
        'M': legacy.choose_best(grouped_entries['M']),
        'D': legacy.choose_best(grouped_entries['D']),
    }
    summary['discarded_extra_side'] = (
        summary.get('discarded_extra_side', 0)
        + max(0, len(grouped_entries['M']) - 1)
        + max(0, len(grouped_entries['D']) - 1))
    return resolved


def export_visualizations(records: Dict[int, legacy.ImageRecord],
                          image_id_map: Dict[int, int],
                          annotations: List[dict],
                          output_dir: Path,
                          count: int,
                          seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ann_by_image = defaultdict(list)
    for annotation in annotations:
        ann_by_image[annotation['image_id']].append(annotation)

    sample_ids = [sample_id for sample_id, image_id in image_id_map.items()
                  if image_id in ann_by_image]
    if count <= 0 or not sample_ids:
        return

    randomizer = random.Random(seed)
    selected = sample_ids if count >= len(sample_ids) else sorted(
        randomizer.sample(sample_ids, count))

    for sample_id in selected:
        record = records[sample_id]
        image_id = image_id_map[sample_id]
        with Image.open(record.image_path) as image:
            canvas = image.convert('RGBA')
        overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        drawer = ImageDraw.Draw(overlay)

        for annotation in ann_by_image[image_id]:
            polygon = annotation['segmentation'][0]
            polygon_xy = [(polygon[i], polygon[i + 1])
                          for i in range(0, len(polygon), 2)]
            drawer.polygon(polygon_xy, fill=(255, 90, 90, 40),
                           outline=(255, 90, 90, 180))

            keypoints = annotation['keypoints']
            labels = ['M_C', 'M_B', 'A', 'D_B', 'D_C']
            colors = {
                'M_C': (0, 178, 255, 255),
                'M_B': (0, 178, 255, 255),
                'A': (255, 214, 10, 255),
                'D_B': (255, 82, 82, 255),
                'D_C': (255, 82, 82, 255),
            }
            for index, label in enumerate(labels):
                x = keypoints[index * 3]
                y = keypoints[index * 3 + 1]
                drawer.ellipse([x - 3, y - 3, x + 3, y + 3],
                               fill=colors[label], outline=colors[label])
                drawer.text((x + 4, y - 10), label, fill=colors[label])

            label_x = keypoints[0]
            label_y = keypoints[1]
            drawer.text((label_x + 8, label_y + 4), str(annotation['tooth_id']),
                        fill=(255, 255, 255, 255))

        output = Image.alpha_composite(canvas, overlay).convert('RGB')
        output.save(output_dir / f'overlay_{sample_id}.jpg', quality=95)


def build_records(dataset_root: Path, excel_dir: Path, image_dir: Path,
                  traced_dir: Path, sample_ids: List[int],
                  summary: dict, source_canvas_width: Optional[int],
                  source_canvas_height: int, point_offset_x: float,
                  point_offset_y: float):
    excel_map = {
        legacy.extract_sample_id(path): path
        for path in sorted(excel_dir.glob('*.xlsx'))
        if not path.name.startswith('.~') and not path.name.startswith('~$')
    }
    image_map = {
        legacy.extract_sample_id(path): path
        for path in sorted(image_dir.glob('*')) if path.is_file()
    }
    traced_map = {
        legacy.extract_sample_id(path): path
        for path in sorted(traced_dir.glob('*')) if path.is_file()
    }

    records: Dict[int, legacy.ImageRecord] = {}
    images: List[dict] = []
    image_id_map: Dict[int, int] = {}
    for image_id, sample_id in enumerate(sample_ids):
        record = legacy.build_record(
            sample_id=sample_id,
            image_path=image_map[sample_id],
            traced_image_path=traced_map.get(sample_id),
            excel_path=excel_map[sample_id],
            source_canvas_width=source_canvas_width,
            source_canvas_height=source_canvas_height,
            point_offset_x=point_offset_x,
            point_offset_y=point_offset_y,
            summary=summary)
        records[sample_id] = record
        image_id_map[sample_id] = image_id
        images.append({
            'id': image_id,
            'file_name': legacy.relative_file_name(dataset_root, record.image_path),
            'width': record.width,
            'height': record.height,
        })
    return records, images, image_id_map


def build_annotations(records: Dict[int, legacy.ImageRecord],
                      image_id_map: Dict[int, int], summary: dict) -> List[dict]:
    annotations: List[dict] = []
    annotation_id = 0
    for sample_id, record in records.items():
        image_id = image_id_map[sample_id]
        for tooth_id in legacy.TARGET_TEETH:
            resolved = resolve_strict_pair_entries(
                tooth_id, record.side_entries.get(tooth_id, []), summary)
            if resolved is None:
                continue

            annotation, debug = build_tooth_instance(
                tooth_id=tooth_id,
                mesial_points=resolved['M'].raw_points,
                distal_points=resolved['D'].raw_points,
                image_id=image_id,
                annotation_id=annotation_id,
                image_width=record.width,
                image_height=record.height)
            if annotation is None:
                summary[debug['skip_reason']] = summary.get(debug['skip_reason'], 0) + 1
                continue

            summary['snapped_keypoints'] += int(debug['snapped_keypoints'])
            summary['instances_written'] += 1
            annotations.append(annotation)
            annotation_id += 1
    return annotations


def subset_payload(images: List[dict], annotations: List[dict],
                   split_sample_ids: List[int], image_id_map: Dict[int, int],
                   summary: dict, split_name: str) -> dict:
    selected_image_ids = {image_id_map[sample_id] for sample_id in split_sample_ids}
    split_images = [image for image in images if image['id'] in selected_image_ids]
    split_annotations = [annotation for annotation in annotations
                         if annotation['image_id'] in selected_image_ids]
    split_summary = dict(summary)
    split_summary['num_images'] = len(split_images)
    split_summary['num_annotations'] = len(split_annotations)
    return build_dataset_payload(split_images, split_annotations, split_name,
                                 split_summary)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir or dataset_root / 'annotations_v2'
    excel_dir = args.excel_subdir or legacy.detect_subdir(dataset_root, ['excel', 'Excel'])
    image_dir = args.image_subdir or legacy.detect_subdir(dataset_root, ['原图'])
    traced_dir = args.traced_subdir or legacy.detect_subdir(dataset_root, ['描点图'])

    excel_ids = {
        legacy.extract_sample_id(path)
        for path in sorted(excel_dir.glob('*.xlsx'))
        if not path.name.startswith('.~') and not path.name.startswith('~$')
    }
    image_ids = {
        legacy.extract_sample_id(path)
        for path in sorted(image_dir.glob('*')) if path.is_file()
    }
    sample_ids = sorted(excel_ids & image_ids)
    if args.sample_limit is not None:
        sample_ids = sample_ids[:args.sample_limit]

    summary = initial_summary(dataset_root, excel_dir, image_dir, traced_dir, args)
    summary['num_excels'] = len(excel_ids)
    summary['num_images'] = len(image_ids)
    summary['num_samples_before_filtering'] = len(sample_ids)

    if args.sample_list_json is not None:
        allowed_sample_ids = set(load_sample_ids_from_json(args.sample_list_json))
        excluded_sample_ids = [
            sample_id for sample_id in sample_ids if sample_id not in allowed_sample_ids
        ]
        sample_ids = [sample_id for sample_id in sample_ids
                      if sample_id in allowed_sample_ids]
        summary['sample_list_json'] = str(args.sample_list_json)
        summary['num_samples_filtered_out_by_sample_list'] = len(excluded_sample_ids)
        summary['excluded_sample_ids_by_sample_list'] = excluded_sample_ids

    if args.min_match_ratio is not None:
        quality_report_json = (
            args.quality_report_json or
            dataset_root / 'annotation_quality_report' / 'sample_metrics.json')
        kept_sample_ids, excluded_sample_ids = load_quality_filtered_sample_ids(
            sample_ids, quality_report_json, args.min_match_ratio)
        sample_ids = kept_sample_ids
        summary['quality_report_json'] = str(quality_report_json)
        summary['num_samples_filtered_out_by_match_ratio'] = len(
            excluded_sample_ids)
        summary['excluded_sample_ids_by_match_ratio'] = excluded_sample_ids

    summary['num_samples'] = len(sample_ids)

    initial_point_offset_x = float(args.point_offset_x or 0.0)
    initial_point_offset_y = float(args.point_offset_y or 0.0)
    records, images, image_id_map = build_records(
        dataset_root,
        excel_dir,
        image_dir,
        traced_dir,
        sample_ids,
        summary,
        source_canvas_width=args.source_canvas_width,
        source_canvas_height=args.source_canvas_height,
        point_offset_x=initial_point_offset_x,
        point_offset_y=initial_point_offset_y)
    summary['point_offset_x'] = initial_point_offset_x
    summary['point_offset_y'] = initial_point_offset_y

    annotations = build_annotations(records, image_id_map, summary)
    summary['num_annotations'] = len(annotations)

    full_payload = build_dataset_payload(images, annotations, 'all', summary)
    write_json(output_dir / 'panoramic_teeth_instances_all.json', full_payload)

    if not args.skip_split:
        split_map = legacy.split_dataset(sample_ids, args.train_ratio,
                                         args.val_ratio, args.test_ratio,
                                         args.seed)
        summary['splits'] = {name: len(ids) for name, ids in split_map.items()}
        for split_name, split_ids in split_map.items():
            payload = subset_payload(images, annotations, split_ids, image_id_map,
                                     summary, split_name)
            write_json(output_dir / f'panoramic_teeth_instances_{split_name}.json',
                       payload)

    if not args.skip_visualization:
        export_visualizations(records, image_id_map, annotations,
                              output_dir / 'visualizations',
                              args.visualize_count, args.seed)

    write_json(output_dir / 'panoramic_teeth_instances_summary.json', summary)


if __name__ == '__main__':
    main()
