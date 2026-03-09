from .annotation_utils import build_tooth_instance, dedupe_consecutive_points

__all__ = [
    'build_tooth_instance',
    'dedupe_consecutive_points',
]

try:
    from .panoramic_teeth_dataset import PanoramicTeethRootDataset
    from .transforms import GenerateRootMask, PackTeethInputs
    __all__.extend([
        'PanoramicTeethRootDataset',
        'GenerateRootMask',
        'PackTeethInputs',
    ])
except ModuleNotFoundError as exc:
    if exc.name not in {'mmcv', 'mmengine'}:
        raise
