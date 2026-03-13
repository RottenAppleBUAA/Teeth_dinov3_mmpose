from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import panoramic_teeth_converter_common as common  # noqa: E402
import panoramic_teeth_excel_to_coco_v2 as converter_v2  # noqa: E402

Point2D = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure per-sample alignment between panoramic-teeth '
        'Excel annotations and traced-image blue polylines.')
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
        help='Directory used to store the generated quality report. '
        'Defaults to <dataset-root>/annotation_quality_report.')
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--source-canvas-width', type=int, default=None)
    parser.add_argument('--source-canvas-height', type=int, default=512)
    parser.add_argument('--point-offset-x', type=float, default=0.0)
    parser.add_argument('--point-offset-y', type=float, default=0.0)
    parser.add_argument('--max-distance', type=float, default=40.0)
    parser.add_argument('--cell-size', type=int, default=16)
    parser.add_argument('--blue-threshold', type=int, default=100)
    parser.add_argument('--red-max', type=int, default=80)
    parser.add_argument('--green-max', type=int, default=80)
    parser.add_argument('--top-k', type=int, default=50)
    return parser.parse_args()


def extract_blue_points(
    image_path: Path,
    blue_threshold: int,
    red_max: int,
    green_max: int,
) -> List[Tuple[int, int]]:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert('RGB'))
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = ((blue > blue_threshold) & (red < red_max) & (green < green_max))
    ys, xs = np.nonzero(mask)
    return list(zip(xs.tolist(), ys.tolist()))


def build_bucket_index(
    points: Sequence[Tuple[int, int]],
    cell_size: int,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    buckets: DefaultDict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for x, y in points:
        buckets[(x // cell_size, y // cell_size)].append((x, y))
    return dict(buckets)


def nearest_point_offset(
    point: Point2D,
    bucket_index: Dict[Tuple[int, int], List[Tuple[int, int]]],
    cell_size: int,
    max_distance: float,
) -> Optional[Tuple[float, float, float]]:
    if not bucket_index:
        return None

    point_x, point_y = point
    bucket_x = int(math.floor(point_x / cell_size))
    bucket_y = int(math.floor(point_y / cell_size))
    bucket_radius = int(math.ceil(max_distance / cell_size))
    max_distance_sq = max_distance * max_distance
    best: Optional[Tuple[float, float, float]] = None

    for offset_x in range(-bucket_radius, bucket_radius + 1):
        for offset_y in range(-bucket_radius, bucket_radius + 1):
            for candidate_x, candidate_y in bucket_index.get(
                    (bucket_x + offset_x, bucket_y + offset_y), ()):
                delta_x = float(candidate_x) - point_x
                delta_y = float(candidate_y) - point_y
                distance_sq = delta_x * delta_x + delta_y * delta_y
                if distance_sq > max_distance_sq:
                    continue
                if best is None or distance_sq < best[2]:
                    best = (delta_x, delta_y, distance_sq)

    if best is None:
        return None
    return best[0], best[1], math.sqrt(best[2])


def iter_eval_points(record: common.ImageRecord) -> List[Point2D]:
    scratch_summary: DefaultDict[str, int] = defaultdict(int)
    points: List[Point2D] = []
    for tooth_id in common.TARGET_TEETH:
        resolved = common.resolve_entries_for_tooth(
            tooth_id, record.side_entries.get(tooth_id, []), scratch_summary)
        for entry in resolved.values():
            converted_points, visibility = common.convert_side_entry(entry)
            for point, point_visibility in zip(converted_points, visibility):
                if point_visibility <= 0:
                    continue
                points.append((float(point[0]), float(point[1])))
    return points


def summarize_matches(
    sample_id: int,
    record: common.ImageRecord,
    matches: List[Tuple[float, float, float]],
    total_points: int,
    total_blue_points: int,
    max_distance: float,
) -> dict:
    unmatched_points = total_points - len(matches)
    payload = {
        'sample_id': sample_id,
        'image_path': str(record.image_path),
        'traced_image_path': str(record.traced_image_path)
        if record.traced_image_path else '',
        'num_eval_points': total_points,
        'num_blue_pixels': total_blue_points,
        'num_matched_points': len(matches),
        'num_unmatched_points': unmatched_points,
        'match_ratio': float(len(matches) / total_points) if total_points else 0.0,
        'unmatched_ratio': float(unmatched_points / total_points)
        if total_points else 0.0,
        'max_distance_px': float(max_distance),
        'median_distance_px': None,
        'p90_distance_px': None,
        'mean_distance_px': None,
        'max_observed_distance_px': None,
        'median_dx_px': None,
        'median_dy_px': None,
        'mean_dx_px': None,
        'mean_dy_px': None,
    }
    if not matches:
        return payload

    dxs = [item[0] for item in matches]
    dys = [item[1] for item in matches]
    distances = sorted(item[2] for item in matches)
    p90_index = min(len(distances) - 1, int(len(distances) * 0.9))
    payload.update({
        'median_distance_px': float(median(distances)),
        'p90_distance_px': float(distances[p90_index]),
        'mean_distance_px': float(mean(distances)),
        'max_observed_distance_px': float(max(distances)),
        'median_dx_px': float(median(dxs)),
        'median_dy_px': float(median(dys)),
        'mean_dx_px': float(mean(dxs)),
        'mean_dy_px': float(mean(dys)),
    })
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open('w', encoding='utf-8', newline='') as file:
            file.write('')
        return
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_records(sample_ids: Sequence[int], dataset_root: Path,
                  excel_map: Dict[str, Path], image_map: Dict[str, Path],
                  traced_map: Dict[str, Path],
                  sample_key_by_id: Dict[int, str],
                  args: argparse.Namespace) -> Dict[int, common.ImageRecord]:
    summary: DefaultDict[str, int] = defaultdict(int)
    records, _, _ = converter_v2.build_records(
        dataset_root,
        excel_map,
        image_map,
        traced_map,
        list(sample_ids),
        sample_key_by_id,
        summary,
        args.source_canvas_width,
        args.source_canvas_height,
        args.point_offset_x,
        args.point_offset_y)
    return records


def sort_key(metric: dict) -> Tuple[float, float, int]:
    median_distance = metric['median_distance_px']
    if median_distance is None:
        median_distance = -1.0
    return (metric['unmatched_ratio'], median_distance, metric['sample_id'])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir or dataset_root / 'annotation_quality_report'
    excel_dir = common.detect_subdir(dataset_root, ['excel', 'Excel'])
    image_dir = common.detect_subdir(dataset_root, ['原图'])
    traced_dir = common.detect_subdir(dataset_root, ['描点图'])

    excel_map, image_map, traced_map, common_sample_keys, _ = (
        converter_v2.select_sample_sources(
            excel_dir, image_dir, traced_dir, require_traced=True))
    sample_key_by_id = {
        sample_id: sample_key
        for sample_id, sample_key in enumerate(common_sample_keys)
    }
    sample_ids = sorted(sample_key_by_id.keys())
    if args.sample_limit is not None:
        sample_ids = sample_ids[:args.sample_limit]
        sample_key_by_id = {
            sample_id: sample_key_by_id[sample_id]
            for sample_id in sample_ids
        }

    records = build_records(sample_ids, dataset_root, excel_map, image_map,
                            traced_map, sample_key_by_id, args)
    metrics: List[dict] = []

    for sample_id in sample_ids:
        record = records[sample_id]
        eval_points = iter_eval_points(record)
        blue_points = extract_blue_points(
            record.traced_image_path,
            args.blue_threshold,
            args.red_max,
            args.green_max) if record.traced_image_path else []
        bucket_index = build_bucket_index(blue_points, args.cell_size)
        matches: List[Tuple[float, float, float]] = []
        for point in eval_points:
            match = nearest_point_offset(point, bucket_index, args.cell_size,
                                         args.max_distance)
            if match is None:
                continue
            matches.append(match)
        metrics.append(
            summarize_matches(sample_id, record, matches, len(eval_points),
                              len(blue_points), args.max_distance))

    ranked_metrics = sorted(metrics, key=sort_key, reverse=True)
    top_metrics = ranked_metrics[:args.top_k]

    matched_metrics = [metric for metric in metrics if metric['num_matched_points'] > 0]
    summary = {
        'dataset_root': str(dataset_root),
        'num_samples': len(sample_ids),
        'num_ranked_samples': len(metrics),
        'num_samples_with_matches': len(matched_metrics),
        'max_distance_px': float(args.max_distance),
        'blue_threshold': args.blue_threshold,
        'red_max': args.red_max,
        'green_max': args.green_max,
        'top_k': args.top_k,
        'median_of_sample_medians_px': float(
            median([
                metric['median_distance_px'] for metric in matched_metrics
                if metric['median_distance_px'] is not None
            ])) if matched_metrics else None,
        'mean_of_sample_medians_px': float(
            mean([
                metric['median_distance_px'] for metric in matched_metrics
                if metric['median_distance_px'] is not None
            ])) if matched_metrics else None,
        'worst_sample_id': ranked_metrics[0]['sample_id'] if ranked_metrics else None,
        'worst_sample_unmatched_ratio':
        ranked_metrics[0]['unmatched_ratio'] if ranked_metrics else None,
        'worst_sample_median_distance_px':
        ranked_metrics[0]['median_distance_px'] if ranked_metrics else None,
    }

    write_json(output_dir / 'summary.json', summary)
    write_json(output_dir / 'sample_metrics.json', {'samples': metrics})
    write_json(output_dir / 'top_bad_samples.json', {'samples': top_metrics})
    write_csv(output_dir / 'sample_metrics.csv', metrics)
    write_csv(output_dir / 'top_bad_samples.csv', top_metrics)

    print(f'Wrote summary to {output_dir / "summary.json"}')
    print(f'Wrote sample metrics to {output_dir / "sample_metrics.csv"}')
    if ranked_metrics:
        print('Worst samples:')
        for metric in top_metrics[:10]:
            print(
                f"  sample={metric['sample_id']} unmatched={metric['unmatched_ratio']:.3f} "
                f"median={metric['median_distance_px']}")


if __name__ == '__main__':
    main()
