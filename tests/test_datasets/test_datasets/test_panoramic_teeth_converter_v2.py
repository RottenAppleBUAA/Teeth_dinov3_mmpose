import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_ROOT = ROOT / 'tools' / 'dataset_converters'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import panoramic_teeth_excel_to_coco as legacy
import panoramic_teeth_excel_to_coco_v2 as converter_v2


def make_raw_entry(
    tooth_id,
    side_token,
    *,
    kind='manual',
    tooth_id_from_label=True,
    side_from_label=True,
):
    return legacy.RawEntry(
        tooth_id=tooth_id,
        side_token=side_token,
        kind=kind,
        tooth_id_from_label=tooth_id_from_label,
        side_from_label=side_from_label,
        raw_points=[(1.0, 1.0), (2.0, 2.0)],
        label='test',
        row_index=2,
    )


def test_resolve_source_canvas_width_uses_rounding():
    assert legacy.resolve_source_canvas_width(3029, 1480, None, 512) == 1048


def test_load_quality_filtered_sample_ids(tmp_path):
    report_path = tmp_path / 'sample_metrics.json'
    report_path.write_text(
        json.dumps({
            'samples': [
                {
                    'sample_id': 1,
                    'match_ratio': 1.0
                },
                {
                    'sample_id': 2,
                    'match_ratio': 0.98
                },
                {
                    'sample_id': 3,
                    'match_ratio': 0.979
                },
            ]
        }),
        encoding='utf-8')

    kept, excluded = converter_v2.load_quality_filtered_sample_ids(
        [1, 2, 3, 4], report_path, 0.98)

    assert kept == [1, 2]
    assert excluded == [3, 4]


def test_load_sample_ids_from_json_supports_samples_payload(tmp_path):
    sample_list = tmp_path / 'samples.json'
    sample_list.write_text(
        json.dumps({
            'samples': [
                {'sample_id': 5},
                {'sample_id': 2},
                {'sample_id': 5},
            ]
        }),
        encoding='utf-8')

    assert converter_v2.load_sample_ids_from_json(sample_list) == [2, 5]


def test_resolve_strict_pair_entries_accepts_auto_left_right():
    summary = {}
    resolved = converter_v2.resolve_strict_pair_entries(
        11,
        [
            make_raw_entry(11, 'left', kind='auto'),
            make_raw_entry(11, 'right', kind='auto'),
        ],
        summary,
    )

    assert resolved is not None
    assert set(resolved) == {'M', 'D'}


def test_resolve_strict_pair_entries_rejects_ambiguous_label():
    summary = {}
    resolved = converter_v2.resolve_strict_pair_entries(
        11,
        [
            make_raw_entry(11, 'M'),
            make_raw_entry(11, 'D', tooth_id_from_label=False),
        ],
        summary,
    )

    assert resolved is None
    assert summary['ambiguous_tooth_label'] == 1


def test_resolve_strict_pair_entries_rejects_single_sided_tooth():
    summary = {}
    resolved = converter_v2.resolve_strict_pair_entries(
        11,
        [make_raw_entry(11, 'M')],
        summary,
    )

    assert resolved is None
    assert summary['single_sided_unpaired_tooth'] == 1


def test_resolve_strict_pair_entries_rejects_duplicate_auto_side():
    summary = {}
    resolved = converter_v2.resolve_strict_pair_entries(
        25,
        [
            make_raw_entry(25, 'left', kind='auto'),
            make_raw_entry(25, 'left', kind='auto'),
        ],
        summary,
    )

    assert resolved is None
    assert summary['single_sided_unpaired_tooth'] == 1
