from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the final panoramic-teeth v2 export pipeline: '
        'quality report, strict sample filtering, and pair-checked '
        'single-tooth annotation export.')
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('datasets/586份数据20260116'),
        help='Root directory containing Excel files and images.')
    parser.add_argument(
        '--report-dir',
        type=Path,
        default=None,
        help='Quality report output directory. Defaults to '
        '<dataset-root>/annotation_quality_report_b100_rg80.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Final annotation output directory. Defaults to '
        '<dataset-root>/annotations_v2_good_strict3_pairchecked.')
    parser.add_argument('--sample-limit', type=int, default=None)
    parser.add_argument('--source-canvas-width', type=int, default=None)
    parser.add_argument('--source-canvas-height', type=int, default=512)
    parser.add_argument('--blue-threshold', type=int, default=100)
    parser.add_argument('--red-max', type=int, default=80)
    parser.add_argument('--green-max', type=int, default=80)
    parser.add_argument('--max-distance', type=float, default=40.0)
    parser.add_argument('--match-ratio-threshold', type=float, default=0.98)
    parser.add_argument('--median-distance-threshold', type=float, default=3.0)
    parser.add_argument('--max-median-dx', type=float, default=3.0)
    parser.add_argument('--max-median-dy', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-visualization', action='store_true')
    return parser.parse_args()


def run_command(command: List[str]) -> None:
    subprocess.run(command, check=True)


def build_strict3_lists(
    report_dir: Path,
    match_ratio_threshold: float,
    median_distance_threshold: float,
    max_median_dx: float,
    max_median_dy: float,
) -> Tuple[Path, Path]:
    sample_metrics_path = report_dir / 'sample_metrics.json'
    samples = json.loads(sample_metrics_path.read_text(
        encoding='utf-8'))['samples']

    good_samples = []
    bad_samples = []
    for sample in samples:
        reasons = []
        if sample['match_ratio'] < match_ratio_threshold:
            reasons.append('match_ratio')
        if (sample['median_distance_px'] is None
                or sample['median_distance_px'] > median_distance_threshold):
            reasons.append('median_distance')
        if (sample['median_dx_px'] is None
                or abs(sample['median_dx_px']) > max_median_dx):
            reasons.append('median_dx')
        if (sample['median_dy_px'] is None
                or abs(sample['median_dy_px']) > max_median_dy):
            reasons.append('median_dy')
        enriched = dict(sample)
        enriched['fail_reasons'] = '|'.join(reasons)
        if reasons:
            bad_samples.append(enriched)
        else:
            good_samples.append(enriched)

    good_json_path = report_dir / 'good_samples_strict3.json'
    bad_json_path = report_dir / 'bad_samples_strict3.json'
    good_csv_path = report_dir / 'good_samples_strict3.csv'
    bad_csv_path = report_dir / 'bad_samples_strict3.csv'

    good_json_path.write_text(
        json.dumps({'samples': good_samples}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    bad_json_path.write_text(
        json.dumps({'samples': bad_samples}, ensure_ascii=False, indent=2),
        encoding='utf-8')

    for path, rows in ((good_csv_path, good_samples), (bad_csv_path, bad_samples)):
        with path.open('w', encoding='utf-8', newline='') as file:
            if not rows:
                file.write('')
                continue
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return good_json_path, bad_json_path


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dataset_root = args.dataset_root
    report_dir = args.report_dir or dataset_root / 'annotation_quality_report_b100_rg80'
    output_dir = args.output_dir or dataset_root / 'annotations_v2_good_strict3_pairchecked'

    quality_report_cmd = [
        sys.executable,
        str(script_dir / 'panoramic_teeth_annotation_quality_report.py'),
        '--dataset-root',
        str(dataset_root),
        '--output-dir',
        str(report_dir),
        '--blue-threshold',
        str(args.blue_threshold),
        '--red-max',
        str(args.red_max),
        '--green-max',
        str(args.green_max),
        '--max-distance',
        str(args.max_distance),
        '--source-canvas-height',
        str(args.source_canvas_height),
    ]
    if args.source_canvas_width is not None:
        quality_report_cmd.extend([
            '--source-canvas-width',
            str(args.source_canvas_width),
        ])
    if args.sample_limit is not None:
        quality_report_cmd.extend(['--sample-limit', str(args.sample_limit)])
    run_command(quality_report_cmd)

    good_json_path, bad_json_path = build_strict3_lists(
        report_dir,
        match_ratio_threshold=args.match_ratio_threshold,
        median_distance_threshold=args.median_distance_threshold,
        max_median_dx=args.max_median_dx,
        max_median_dy=args.max_median_dy,
    )

    export_cmd = [
        sys.executable,
        str(script_dir / 'panoramic_teeth_excel_to_coco_v2.py'),
        '--dataset-root',
        str(dataset_root),
        '--output-dir',
        str(output_dir),
        '--sample-list-json',
        str(good_json_path),
        '--seed',
        str(args.seed),
        '--source-canvas-height',
        str(args.source_canvas_height),
    ]
    if args.source_canvas_width is not None:
        export_cmd.extend(['--source-canvas-width', str(args.source_canvas_width)])
    if args.sample_limit is not None:
        export_cmd.extend(['--sample-limit', str(args.sample_limit)])
    if args.skip_visualization:
        export_cmd.append('--skip-visualization')
    else:
        export_cmd.extend(['--visualize-count', '10000'])
    run_command(export_cmd)

    print(f'Wrote strict report to {report_dir}')
    print(f'Good sample list: {good_json_path}')
    print(f'Bad sample list: {bad_json_path}')
    print(f'Final annotations: {output_dir}')


if __name__ == '__main__':
    main()
