import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_ROOT = ROOT / 'tools' / 'dataset_converters'
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import panoramic_teeth_export_debug_views as debug_views


def test_draw_blue_pixel_image_marks_expected_pixels():
    image = debug_views.draw_blue_pixel_image((4, 3), [(1, 1), (3, 2)])

    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((1, 1)) == (0, 178, 255)
    assert image.getpixel((3, 2)) == (0, 178, 255)


def test_draw_blue_pixel_overlay_preserves_transparency():
    image = debug_views.draw_blue_pixel_overlay((3, 2), [(1, 0)])

    assert image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert image.getpixel((1, 0)) == (0, 178, 255, 220)
