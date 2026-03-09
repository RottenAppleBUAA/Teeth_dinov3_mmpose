import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mmcv')
pytest.importorskip('mmengine')

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.panoramic_teeth.datasets.transforms import GenerateRootMask


def test_generate_root_mask_rasterizes_polygon():
    transform = GenerateRootMask(use_udp=False)
    results = {
        'root_polygon': np.array([[20.0, 20.0], [20.0, 60.0], [40.0, 80.0],
                                  [60.0, 80.0], [80.0, 60.0], [80.0, 20.0]],
                                 dtype=np.float32),
        'input_center': np.array([50.0, 50.0], dtype=np.float32),
        'input_scale': np.array([100.0, 100.0], dtype=np.float32),
        'input_size': (100, 100),
        'img_shape': (100, 100),
        'flip': False,
    }

    output = transform(results)

    assert output is not None
    assert output['root_mask'].shape == (1, 100, 100)
    assert output['transformed_root_polygon'].shape == (6, 2)
    assert output['root_mask'].sum() > 0
