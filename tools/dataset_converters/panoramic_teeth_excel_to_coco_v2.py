from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import panoramic_teeth_converter_common as common  # noqa: E402
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


def is_ignored_auxiliary_file(path: Path) -> bool:
    name = path.name
    return (name.startswith('.')
            or name.startswith('~$')
            or name.startswith('.~')
            or ':Zone.Identifier' in name)


def strip_excel_export_suffix(stem: str) -> str:
    match = re.match(
        r'^(?P<base>.+?\.(?:jpe?g|png|bmp|tif|tiff))_(?:\d{8})_(?:\d{6})$',
        stem,
        re.IGNORECASE)
    if match:
        return Path(match.group('base')).stem
    return stem


def derive_numeric_sample_token(path: Path) -> Optional[str]:
    try:
        return f'id:{common.extract_sample_id(path)}'
    except ValueError:
        return None


def derive_name_sample_token(path: Path, source_kind: str) -> str:
    stem = path.stem
    if source_kind == 'excel':
        stem = strip_excel_export_suffix(stem)
    elif source_kind == 'traced' and stem.lower().endswith('_reserve'):
        stem = stem[:-len('_reserve')]
    return f'name:{stem.casefold()}'


def collect_source_files(
    source_dir: Path,
    source_kind: str,
    required_suffixes: Tuple[str, ...],
    token_mode: str,
) -> Dict[str, Path]:
    token_map: Dict[str, Path] = {}
    for path in sorted(source_dir.glob('*')):
        if not path.is_file() or is_ignored_auxiliary_file(path):
            continue
        if required_suffixes and path.suffix.lower() not in required_suffixes:
            continue
        if token_mode == 'numeric':
            token = derive_numeric_sample_token(path)
        elif token_mode == 'name':
            token = derive_name_sample_token(path, source_kind)
        else:
            raise ValueError(f'Unsupported token mode: {token_mode}')
        if token is not None:
            token_map[token] = path
    return token_map


def collect_source_maps(excel_dir: Path, image_dir: Path, traced_dir: Path,
                        token_mode: str) -> Tuple[Dict[str, Path], Dict[str, Path],
                                                  Dict[str, Path]]:
    excel_map = collect_source_files(excel_dir, 'excel', ('.xlsx',), token_mode)
    image_map = collect_source_files(
        image_dir, 'image',
        ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'), token_mode)
    traced_map = collect_source_files(
        traced_dir, 'traced',
        ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'), token_mode)
    return excel_map, image_map, traced_map


def count_valid_source_files(source_dir: Path,
                             required_suffixes: Tuple[str, ...]) -> int:
    count = 0
    for path in sorted(source_dir.glob('*')):
        if not path.is_file() or is_ignored_auxiliary_file(path):
            continue
        if required_suffixes and path.suffix.lower() not in required_suffixes:
            continue
        count += 1
    return count


def select_sample_sources(
    excel_dir: Path,
    image_dir: Path,
    traced_dir: Path,
    require_traced: bool = False,
) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path], List[str], dict]:
    numeric_excel_map, numeric_image_map, numeric_traced_map = collect_source_maps(
        excel_dir, image_dir, traced_dir, 'numeric')
    name_excel_map, name_image_map, name_traced_map = collect_source_maps(
        excel_dir, image_dir, traced_dir, 'name')

    maps_by_mode = {
        'numeric': {
            'excel': numeric_excel_map,
            'image': numeric_image_map,
            'traced': numeric_traced_map,
        },
        'name': {
            'excel': name_excel_map,
            'image': name_image_map,
            'traced': name_traced_map,
        },
    }
    numeric_common_keys = set(numeric_excel_map) & set(numeric_image_map)
    name_common_keys = set(name_excel_map) & set(name_image_map)
    if require_traced:
        numeric_common_keys &= set(numeric_traced_map)
        name_common_keys &= set(name_traced_map)

    merged_excel_map: Dict[str, Path] = {}
    merged_image_map: Dict[str, Path] = {}
    merged_traced_map: Dict[str, Path] = {}
    selected_keys: List[str] = []
    seen_source_triplets = set()
    mode_usage = {'numeric': 0, 'name': 0}

    for mode_name, common_keys in (
            ('numeric', sorted(numeric_common_keys)),
            ('name', sorted(name_common_keys))):
        mode_maps = maps_by_mode[mode_name]
        for sample_key in common_keys:
            traced_path = mode_maps['traced'].get(sample_key)
            source_triplet = (
                mode_maps['excel'][sample_key],
                mode_maps['image'][sample_key],
                traced_path,
            )
            if source_triplet in seen_source_triplets:
                continue
            seen_source_triplets.add(source_triplet)
            selected_keys.append(sample_key)
            merged_excel_map[sample_key] = mode_maps['excel'][sample_key]
            merged_image_map[sample_key] = mode_maps['image'][sample_key]
            if traced_path is not None:
                merged_traced_map[sample_key] = traced_path
            mode_usage[mode_name] += 1

    stats = {
        'num_valid_excels': count_valid_source_files(excel_dir, ('.xlsx',)),
        'num_valid_images': count_valid_source_files(
            image_dir, ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')),
        'num_valid_traced_images': count_valid_source_files(
            traced_dir, ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')),
        'num_common_samples_numeric_mode': len(numeric_common_keys),
        'num_common_samples_name_mode': len(name_common_keys),
        'num_selected_samples_numeric_mode': mode_usage['numeric'],
        'num_selected_samples_name_mode': mode_usage['name'],
        'sample_key_mode': (
            'mixed'
            if mode_usage['numeric'] > 0 and mode_usage['name'] > 0
            else ('numeric_id' if mode_usage['numeric'] > 0 else 'file_stem')),
    }
    return merged_excel_map, merged_image_map, merged_traced_map, selected_keys, stats


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


def has_explicit_tooth_and_side(entry: common.RawEntry) -> bool:
    return (entry.tooth_id_from_label and entry.side_from_label
            and entry.side_token in {'M', 'D', 'left', 'right'})


def canonical_side_for_entry(tooth_id: int,
                             entry: common.RawEntry) -> Optional[str]:
    if entry.side_token in {'M', 'D'}:
        return entry.side_token
    if entry.side_token in {'left', 'right'}:
        return common.image_side_to_md(tooth_id, entry.side_token)
    return None


def resolve_strict_pair_entries(
    tooth_id: int,
    entries: Sequence[common.RawEntry],
    summary: dict,
) -> Optional[Dict[str, common.RawEntry]]:
    if not entries:
        summary['single_sided_unpaired_tooth'] = (
            summary.get('single_sided_unpaired_tooth', 0) + 1)
        return None

    if any(not has_explicit_tooth_and_side(entry) for entry in entries):
        summary['ambiguous_tooth_label'] = (
            summary.get('ambiguous_tooth_label', 0) + 1)
        return None

    grouped_entries: Dict[str, List[common.RawEntry]] = {'M': [], 'D': []}
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
        'M': common.choose_best(grouped_entries['M']),
        'D': common.choose_best(grouped_entries['D']),
    }
    summary['discarded_extra_side'] = (
        summary.get('discarded_extra_side', 0)
        + max(0, len(grouped_entries['M']) - 1)
        + max(0, len(grouped_entries['D']) - 1))
    return resolved


def export_visualizations(records: Dict[int, common.ImageRecord],
                          image_id_map: Dict[int, int],
                          annotations: List[dict],
                          output_dir: Path,
                          count: int,
                          seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob('overlay_*.jpg'):
        if stale_file.is_file():
            stale_file.unlink()
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


def build_records(dataset_root: Path, excel_map: Dict[str, Path],
                  image_map: Dict[str, Path], traced_map: Dict[str, Path],
                  sample_ids: List[int],
                  sample_key_by_id: Dict[int, str],
                  summary: dict, source_canvas_width: Optional[int],
                  source_canvas_height: int, point_offset_x: float,
                  point_offset_y: float):
    records: Dict[int, common.ImageRecord] = {}
    images: List[dict] = []
    image_id_map: Dict[int, int] = {}
    for image_id, sample_id in enumerate(sample_ids):
        sample_key = sample_key_by_id[sample_id]
        record = common.build_record(
            sample_id=sample_id,
            image_path=image_map[sample_key],
            traced_image_path=traced_map.get(sample_key),
            excel_path=excel_map[sample_key],
            source_canvas_width=source_canvas_width,
            source_canvas_height=source_canvas_height,
            point_offset_x=point_offset_x,
            point_offset_y=point_offset_y,
            summary=summary)
        records[sample_id] = record
        image_id_map[sample_id] = image_id
        images.append({
            'id': image_id,
            'file_name': common.relative_file_name(dataset_root, record.image_path),
            'width': record.width,
            'height': record.height,
        })
    return records, images, image_id_map


def build_annotations(records: Dict[int, common.ImageRecord],
                      image_id_map: Dict[int, int], summary: dict) -> List[dict]:
    annotations: List[dict] = []
    annotation_id = 0
    for sample_id, record in records.items():
        image_id = image_id_map[sample_id]
        for tooth_id in common.TARGET_TEETH:
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
    excel_dir = args.excel_subdir or common.detect_subdir(dataset_root, ['excel', 'Excel'])
    image_dir = args.image_subdir or common.detect_subdir(dataset_root, ['原图'])
    traced_dir = args.traced_subdir or common.detect_subdir(dataset_root, ['描点图'])

    excel_map, image_map, traced_map, common_sample_keys, mode_stats = (
        select_sample_sources(excel_dir, image_dir, traced_dir, require_traced=False))
    if args.sample_limit is not None:
        common_sample_keys = common_sample_keys[:args.sample_limit]
    sample_key_by_id = {
        sample_id: sample_key
        for sample_id, sample_key in enumerate(common_sample_keys)
    }
    sample_ids = sorted(sample_key_by_id.keys())
    if args.sample_limit is not None:
        sample_ids = sample_ids[:args.sample_limit]

    summary = initial_summary(dataset_root, excel_dir, image_dir, traced_dir, args)
    summary['num_excels'] = mode_stats['num_valid_excels']
    summary['num_images'] = mode_stats['num_valid_images']
    summary['num_traced_images'] = mode_stats['num_valid_traced_images']
    summary['num_samples_before_filtering'] = len(sample_ids)
    summary.update(mode_stats)
    summary['num_samples_with_traced_image'] = len(
        set(common_sample_keys) & set(traced_map.keys()))
    summary['num_samples_missing_traced_image'] = len(common_sample_keys) - len(
        set(common_sample_keys) & set(traced_map.keys()))

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
        excel_map,
        image_map,
        traced_map,
        sample_ids,
        sample_key_by_id,
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
        split_map = common.split_dataset(sample_ids, args.train_ratio,
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
