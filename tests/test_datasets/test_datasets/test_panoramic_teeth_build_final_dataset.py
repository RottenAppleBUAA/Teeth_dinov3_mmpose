import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_ROOT = ROOT / 'tools' / 'dataset_converters'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import panoramic_teeth_build_final_dataset as pipeline


def test_build_strict3_lists_splits_good_and_bad(tmp_path):
    report_dir = tmp_path / 'report'
    report_dir.mkdir()
    (report_dir / 'sample_metrics.json').write_text(
        json.dumps({
            'samples': [
                {
                    'sample_id': 1,
                    'match_ratio': 1.0,
                    'median_distance_px': 2.0,
                    'median_dx_px': 1.0,
                    'median_dy_px': 1.0,
                },
                {
                    'sample_id': 2,
                    'match_ratio': 0.97,
                    'median_distance_px': 2.0,
                    'median_dx_px': 1.0,
                    'median_dy_px': 1.0,
                },
                {
                    'sample_id': 3,
                    'match_ratio': 1.0,
                    'median_distance_px': 4.5,
                    'median_dx_px': 4.0,
                    'median_dy_px': 0.0,
                },
            ]
        }),
        encoding='utf-8')

    good_path, bad_path = pipeline.build_strict3_lists(
        report_dir,
        match_ratio_threshold=0.98,
        median_distance_threshold=3.0,
        max_median_dx=3.0,
        max_median_dy=3.0,
    )

    good = json.loads(good_path.read_text(encoding='utf-8'))['samples']
    bad = json.loads(bad_path.read_text(encoding='utf-8'))['samples']

    assert [sample['sample_id'] for sample in good] == [1]
    assert [sample['sample_id'] for sample in bad] == [2, 3]
