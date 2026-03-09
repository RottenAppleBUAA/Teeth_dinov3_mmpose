from .datasets import build_tooth_instance, dedupe_consecutive_points

try:
    from .datasets import *  # noqa: F401,F403
    from .evaluation import *  # noqa: F401,F403
    from .models import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    if exc.name not in {'mmcv', 'mmengine'}:
        raise
