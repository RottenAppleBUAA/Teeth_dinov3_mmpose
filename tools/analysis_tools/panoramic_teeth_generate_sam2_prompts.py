from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence


def _patch_torch_pytree_compat():
    try:
        import torch.utils._pytree as torch_pytree
    except Exception:
        return

    if hasattr(torch_pytree, 'register_pytree_node'):
        return
    if not hasattr(torch_pytree, '_register_pytree_node'):
        return

    def _compat_register_pytree_node(typ,
                                     flatten_fn,
                                     unflatten_fn,
                                     *,
                                     serialized_type_name=None,
                                     to_dumpable_context=None,
                                     from_dumpable_context=None):
        del serialized_type_name
        return torch_pytree._register_pytree_node(
            typ,
            flatten_fn,
            unflatten_fn,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context)

    torch_pytree.register_pytree_node = _compat_register_pytree_node


_patch_torch_pytree_compat()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cv2 = None
mmengine = None
np = None
Config = None
init_default_scope = None
inference_topdown = None
init_model = None
get_udp_warp_matrix = None
get_warp_matrix = None
register_all_modules = None

DEFAULT_PACK_META_KEYS = (
    'id', 'img_id', 'img_path', 'category_id', 'ori_shape', 'img_shape',
    'input_size', 'input_center', 'input_scale', 'flip', 'flip_direction',
    'flip_indices', 'raw_ann_info', 'dataset_name', 'tooth_id')


def import_runtime_deps():
    global cv2, mmengine, np
    global Config, init_default_scope
    global inference_topdown, init_model
    global get_udp_warp_matrix, get_warp_matrix
    global register_all_modules

    import cv2 as _cv2
    import mmengine as _mmengine
    import numpy as _np

    from mmengine.config import Config as _Config
    from mmengine.registry import init_default_scope as _init_default_scope
    from mmpose.apis import inference_topdown as _inference_topdown
    from mmpose.apis import init_model as _init_model
    from mmpose.structures.bbox import get_udp_warp_matrix as _get_udp_warp_matrix
    from mmpose.structures.bbox import get_warp_matrix as _get_warp_matrix
    from mmpose.utils import register_all_modules as _register_all_modules

    cv2 = _cv2
    mmengine = _mmengine
    np = _np
    Config = _Config
    init_default_scope = _init_default_scope
    inference_topdown = _inference_topdown
    init_model = _init_model
    get_udp_warp_matrix = _get_udp_warp_matrix
    get_warp_matrix = _get_warp_matrix
    register_all_modules = _register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate SAM2 prompts from panoramic_teeth predictions.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--phase',
        default='auto',
        choices=['auto', 'train', 'val', 'test'],
        help='Dataset split used to locate image annotations.')
    parser.add_argument(
        '--image',
        default=None,
        help='Optional image path. If omitted, a file dialog or console '
        'prompt will be used for interactive selection.')
    parser.add_argument(
        '--ann-file',
        default=None,
        help='Optional COCO annotation file. Defaults to the ann_file '
        'configured in the selected split.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--mask-thr',
        type=float,
        default=0.5,
        help='Threshold used to binarize the predicted tooth mask.')
    parser.add_argument(
        '--box-expand-ratio',
        type=float,
        default=0.15,
        help='Expansion ratio applied to the coarse-mask tight bbox.')
    parser.add_argument(
        '--max-keypoint-mask-distance',
        type=float,
        default=6.0,
        help='Maximum pixel distance allowed for keeping a keypoint that '
        'falls slightly outside the mask.')
    parser.add_argument(
        '--min-positive-points',
        type=int,
        default=3,
        help='If fewer than this many positive points remain, downgrade the '
        'prompt to box_only.')
    parser.add_argument(
        '--output-dir',
        default=str(REPO_ROOT / 'prompt'),
        help='Directory used to save prompt JSON files. Defaults to '
        '<repo>/prompt.')
    parser.add_argument(
        '--cfg-options',
        nargs='*',
        default=[],
        help='Override config options, for example '
        '"test_dataloader.dataset.ann_file=...".')
    return parser.parse_args()


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_path(path: Path) -> str:
    return str(path.resolve()).replace('\\', '/')


def _resolve_from_repo(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_cfg_options(option_list: Sequence[str]) -> dict:
    parsed = {}
    for option in option_list:
        if '=' not in option:
            raise ValueError(
                f'Invalid cfg option "{option}". Expected key=value.')
        key, raw_value = option.split('=', 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            value = ast.literal_eval(raw_value)
        except Exception:
            value = raw_value
        parsed[key] = value
    return parsed


def _select_image_interactively(initial_dir: Path) -> Path:
    if initial_dir.is_file():
        initial_dir = initial_dir.parent

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title='Select one panoramic dental X-ray image',
            initialdir=str(initial_dir))
        root.destroy()
        if selected:
            return Path(selected).resolve()
    except Exception:
        pass

    user_input = input(
        f'Enter image path for prompt generation [{initial_dir}]: ').strip()
    if not user_input:
        raise RuntimeError('No image was selected.')
    return Path(user_input).expanduser().resolve()


def _find_topdown_affine_cfg(pipeline_cfg: Sequence[dict]) -> Optional[dict]:
    for transform in pipeline_cfg:
        if transform.get('type') == 'TopdownAffine':
            return transform
    return None


def prepare_cfg(cfg: Config, phase: str, cfg_options: dict) -> tuple[Config, bool]:
    cfg = deepcopy(cfg)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    dataloader_cfg = cfg[f'{phase}_dataloader']
    pipeline_cfg = dataloader_cfg.dataset.pipeline
    if not pipeline_cfg:
        raise ValueError(f'{phase}_dataloader.dataset.pipeline is empty.')

    inference_pipeline = []
    for transform in pipeline_cfg:
        transform_type = transform.get('type')
        if transform_type in {'GenerateRootMask', 'GenerateTarget'}:
            continue
        inference_pipeline.append(deepcopy(transform))

    if not inference_pipeline:
        raise ValueError('Inference pipeline became empty after filtering.')

    pack_cfg = inference_pipeline[-1]
    if pack_cfg.get('type') != 'PackTeethInputs':
        raise ValueError('The last transform in the inference pipeline must '
                         f'be PackTeethInputs, but got {pack_cfg.get("type")}.')

    meta_keys = list(pack_cfg.get('meta_keys', DEFAULT_PACK_META_KEYS))
    if 'bbox_rotation' not in meta_keys:
        meta_keys.append('bbox_rotation')
    pack_cfg['meta_keys'] = tuple(meta_keys)
    pack_cfg['pack_transformed'] = False
    dataloader_cfg.dataset.pipeline = inference_pipeline

    affine_cfg = _find_topdown_affine_cfg(inference_pipeline)
    use_udp = bool(affine_cfg.get('use_udp', False)) if affine_cfg else False
    return cfg, use_udp


def select_inference_phase(cfg: Config, requested_phase: str) -> str:
    for phase in ('test', 'val'):
        if cfg.get(f'{phase}_dataloader', None) is not None:
            return phase
    if cfg.get(f'{requested_phase}_dataloader', None) is not None:
        return requested_phase
    raise RuntimeError('Failed to find a usable dataloader for inference.')


def load_annotation_bundle(ann_file: Path) -> dict:
    with ann_file.open('r', encoding='utf-8') as f:
        bundle = json.load(f)

    images = {image['id']: image for image in bundle.get('images', [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in bundle.get('annotations', []):
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    for ann_list in anns_by_image.values():
        ann_list.sort(key=lambda ann: (int(ann.get('tooth_id', 9999)), ann['id']))

    return dict(raw=bundle, images=images, anns_by_image=anns_by_image)


def choose_target_image(args, data_root: Path) -> Path:
    if args.image:
        return Path(args.image).expanduser().resolve()
    return _select_image_interactively(data_root)


def extract_image_number(path: Path) -> Optional[int]:
    match = re.search(r'(\d+)', path.stem)
    if not match:
        return None
    return int(match.group(1))


def match_image_info(selected_path: Path, data_root: Path,
                     images: Sequence[dict]) -> dict:
    selected_norm = _normalize_path(selected_path)
    selected_number = extract_image_number(selected_path)
    matched = []
    basename_matches = []
    number_matches = []

    for image_info in images:
        candidate_path = (data_root / image_info['file_name']).resolve()
        if _normalize_path(candidate_path) == selected_norm:
            matched.append(image_info)
        elif candidate_path.name == selected_path.name:
            basename_matches.append(image_info)
        else:
            candidate_number = extract_image_number(candidate_path)
            if (selected_number is not None
                    and candidate_number == selected_number):
                number_matches.append(image_info)

    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise RuntimeError(f'Multiple images matched {selected_path}.')
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(number_matches) == 1:
        return number_matches[0]

    raise FileNotFoundError(
        f'Image {selected_path} was not found in the annotation file.')


def _mask_to_2d(mask) -> Optional[np.ndarray]:
    mask = _to_numpy(mask)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(np.float32)


def _build_warp_matrix(data_sample, use_udp: bool) -> np.ndarray:
    center = np.asarray(data_sample.metainfo['input_center'], dtype=np.float32)
    scale = np.asarray(data_sample.metainfo['input_scale'], dtype=np.float32)
    input_width, input_height = data_sample.metainfo['input_size']
    rotation = data_sample.metainfo.get('bbox_rotation', np.array([0.0],
                                                                  dtype=np.float32))
    rotation = _to_numpy(rotation)
    if rotation is None:
        rot = 0.0
    else:
        rot = float(rotation.reshape(-1)[0])
    if use_udp:
        return get_udp_warp_matrix(
            center, scale, rot, output_size=(input_width, input_height))
    return get_warp_matrix(
        center, scale, rot, output_size=(input_width, input_height))


def unwarp_mask_to_image(mask_crop: np.ndarray,
                         data_sample,
                         image_size: tuple[int, int],
                         use_udp: bool) -> np.ndarray:
    width, height = image_size
    warp_mat = _build_warp_matrix(data_sample, use_udp=use_udp)
    inv_warp = cv2.invertAffineTransform(warp_mat)
    return cv2.warpAffine(
        mask_crop.astype(np.float32),
        inv_warp,
        dsize=(int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0)


def get_data_root(cfg: Config) -> Path:
    for split in ('test', 'val', 'train'):
        dataset_cfg = cfg.get(f'{split}_dataloader', None)
        if dataset_cfg is None:
            continue
        return _resolve_from_repo(dataset_cfg.dataset.data_root)
    raise RuntimeError('Failed to determine dataset data_root from config.')


def ensure_annotation_path(cfg: Config, phase: str, args) -> tuple[Path, Path]:
    dataset_cfg = cfg[f'{phase}_dataloader'].dataset
    data_root = _resolve_from_repo(dataset_cfg.data_root)
    ann_file = args.ann_file or dataset_cfg.ann_file

    ann_path = Path(ann_file)
    if not ann_path.is_absolute():
        ann_path = (data_root / ann_file).resolve()
    return data_root, ann_path


def resolve_annotation_context(cfg: Config, args, image_path: Path):
    if args.ann_file:
        phase = 'test' if args.phase == 'auto' else args.phase
        data_root, ann_path = ensure_annotation_path(cfg, phase, args)
        if not ann_path.exists():
            raise FileNotFoundError(f'Annotation file not found: {ann_path}')
        ann_bundle = load_annotation_bundle(ann_path)
        image_info = match_image_info(
            selected_path=image_path,
            data_root=data_root,
            images=ann_bundle['images'].values())
        return phase, data_root, ann_path, ann_bundle, image_info

    if args.phase != 'auto':
        data_root, ann_path = ensure_annotation_path(cfg, args.phase, args)
        if not ann_path.exists():
            raise FileNotFoundError(f'Annotation file not found: {ann_path}')
        ann_bundle = load_annotation_bundle(ann_path)
        image_info = match_image_info(
            selected_path=image_path,
            data_root=data_root,
            images=ann_bundle['images'].values())
        return args.phase, data_root, ann_path, ann_bundle, image_info

    failures = []
    for phase in ('train', 'val', 'test'):
        data_root, ann_path = ensure_annotation_path(cfg, phase, args)
        if not ann_path.exists():
            failures.append(f'{phase}: missing annotation file {ann_path}')
            continue
        ann_bundle = load_annotation_bundle(ann_path)
        try:
            image_info = match_image_info(
                selected_path=image_path,
                data_root=data_root,
                images=ann_bundle['images'].values())
        except FileNotFoundError:
            failures.append(f'{phase}: image not present')
            continue
        return phase, data_root, ann_path, ann_bundle, image_info

    detail = '; '.join(failures)
    raise FileNotFoundError(
        f'Image {image_path} was not found in train/val/test annotations. '
        f'Details: {detail}')


def _binary_mask_bbox_xyxy(mask_binary: np.ndarray) -> Optional[list[float]]:
    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max()),
        float(ys.max()),
    ]


def _expand_bbox_xyxy(bbox_xyxy: Sequence[float],
                      image_width: int,
                      image_height: int,
                      expand_ratio: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    width = max(x2 - x1 + 1.0, 1.0)
    height = max(y2 - y1 + 1.0, 1.0)
    pad_x = width * float(expand_ratio)
    pad_y = height * float(expand_ratio)
    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width - 1), x2 + pad_x),
        min(float(image_height - 1), y2 + pad_y),
    ]


def _mask_centroid(mask_binary: np.ndarray) -> Optional[list[float]]:
    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def _point_to_mask_distance_map(mask_binary: np.ndarray) -> np.ndarray:
    inverted = (mask_binary <= 0).astype(np.uint8)
    return cv2.distanceTransform(inverted, cv2.DIST_L2, 3)


def _clip_point(point_xy: Sequence[float],
                image_width: int,
                image_height: int) -> list[float]:
    return [
        float(min(max(point_xy[0], 0.0), image_width - 1)),
        float(min(max(point_xy[1], 0.0), image_height - 1)),
    ]


def _derive_arch_hint(tooth_id: Optional[int]) -> Optional[str]:
    if tooth_id is None:
        return None
    tooth_id = int(tooth_id)
    if 1 <= tooth_id <= 16:
        return 'upper'
    if 17 <= tooth_id <= 32:
        return 'lower'
    return None


def _prepare_keypoints(pred_instances) -> tuple[np.ndarray, Optional[np.ndarray]]:
    keypoints_xy = _to_numpy(pred_instances.keypoints)
    if keypoints_xy.ndim == 3:
        keypoints_xy = keypoints_xy[0]
    keypoints_xy = keypoints_xy.astype(np.float32)

    keypoint_scores = _to_numpy(getattr(pred_instances, 'keypoint_scores', None))
    if keypoint_scores is not None and keypoint_scores.ndim == 2:
        keypoint_scores = keypoint_scores[0]
    return keypoints_xy, keypoint_scores


def build_prompt_for_prediction(ann: dict,
                                prediction,
                                image_path: Path,
                                image_size: tuple[int, int],
                                args,
                                keypoint_names: Sequence[str],
                                use_udp: bool) -> tuple[Optional[dict], Optional[dict]]:
    image_width, image_height = image_size
    pred_instances = prediction.pred_instances
    pred_fields = getattr(prediction, 'pred_fields', None)
    mask_crop = _mask_to_2d(getattr(pred_fields, 'root_mask', None))
    if mask_crop is None:
        return None, dict(
            annotation_id=int(ann['id']),
            tooth_id=int(ann.get('tooth_id', -1)),
            reason='missing_pred_mask')

    mask_prob = unwarp_mask_to_image(
        mask_crop=mask_crop,
        data_sample=prediction,
        image_size=(image_width, image_height),
        use_udp=use_udp)
    mask_binary = (mask_prob >= float(args.mask_thr)).astype(np.uint8)
    mask_area = int(mask_binary.sum())
    tight_bbox_xyxy = _binary_mask_bbox_xyxy(mask_binary)
    if tight_bbox_xyxy is None:
        return None, dict(
            annotation_id=int(ann['id']),
            tooth_id=int(ann.get('tooth_id', -1)),
            reason='empty_mask_after_threshold',
            mask_threshold=float(args.mask_thr))

    prompt_box_xyxy = _expand_bbox_xyxy(
        tight_bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
        expand_ratio=float(args.box_expand_ratio))
    centroid_xy = _mask_centroid(mask_binary)
    if centroid_xy is None:
        return None, dict(
            annotation_id=int(ann['id']),
            tooth_id=int(ann.get('tooth_id', -1)),
            reason='missing_mask_centroid')

    distance_map = _point_to_mask_distance_map(mask_binary)
    keypoints_xy, keypoint_scores = _prepare_keypoints(pred_instances)

    positive_points_xy: list[list[float]] = []
    positive_labels: list[int] = []
    kept_keypoints = []
    dropped_keypoints = []
    max_distance = float(args.max_keypoint_mask_distance)

    for index, point in enumerate(keypoints_xy):
        clipped_point = _clip_point(point, image_width, image_height)
        px = int(round(clipped_point[0]))
        py = int(round(clipped_point[1]))
        distance = float(distance_map[py, px])
        score = None
        if keypoint_scores is not None and index < len(keypoint_scores):
            score = float(keypoint_scores[index])

        point_record = dict(
            index=int(index),
            name=str(keypoint_names[index]),
            xy=[float(clipped_point[0]), float(clipped_point[1])],
            score=score,
            distance_to_mask=float(distance))

        if distance <= max_distance:
            positive_points_xy.append(point_record['xy'])
            positive_labels.append(1)
            kept_keypoints.append(point_record)
        else:
            dropped_keypoints.append(point_record)

    centroid_record = dict(
        name='centroid',
        xy=[float(centroid_xy[0]), float(centroid_xy[1])],
        score=None,
        distance_to_mask=0.0)
    positive_points_xy.append(centroid_record['xy'])
    positive_labels.append(1)

    if len(positive_points_xy) >= int(args.min_positive_points):
        prompt_mode = 'box_positive'
    else:
        positive_points_xy = []
        positive_labels = []
        prompt_mode = 'box_only'

    negative_points_xy: list[list[float]] = []
    negative_labels: list[int] = []
    point_coords = positive_points_xy + negative_points_xy
    point_labels = positive_labels + negative_labels

    source_metadata = dict(
        annotation_id=int(ann['id']),
        tooth_id=int(ann.get('tooth_id', -1)),
        arch_hint_optional=_derive_arch_hint(ann.get('tooth_id', None)),
        source_bbox_xywh=[float(v) for v in ann['bbox']],
        coarse_mask_area=int(mask_area),
        coarse_mask_bbox_xyxy=[float(v) for v in tight_bbox_xyxy],
        coarse_mask_threshold=float(args.mask_thr),
        keypoints_kept=kept_keypoints,
        keypoints_dropped=dropped_keypoints,
        centroid=centroid_record,
        box_expand_ratio=float(args.box_expand_ratio),
        prompt_point_count=int(len(point_coords)),
    )

    prompt = dict(
        image_path=str(image_path),
        candidate_id=int(ann['id']),
        tooth_id_optional=int(ann.get('tooth_id', -1)),
        prompt_box_xyxy=[float(v) for v in prompt_box_xyxy],
        positive_points_xy=positive_points_xy,
        positive_labels=positive_labels,
        negative_points_xy=negative_points_xy,
        negative_labels=negative_labels,
        prompt_mode=prompt_mode,
        negative_gate_status='disabled_default',
        source_metadata=source_metadata,
        box=[float(v) for v in prompt_box_xyxy],
        point_coords=point_coords,
        point_labels=point_labels,
    )
    return prompt, None


def generate_prompts(model,
                     image_path: Path,
                     image_info: dict,
                     annotations: Sequence[dict],
                     args,
                     use_udp: bool) -> dict:
    bboxes = [ann['bbox'] for ann in annotations]
    predictions = inference_topdown(
        model, str(image_path), bboxes=bboxes, bbox_format='xywh')
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f'Failed to read image from {image_path}')

    image_height, image_width = image.shape[:2]
    dataset_meta = model.dataset_meta
    keypoint_names = [
        dataset_meta['keypoint_id2name'][index]
        for index in range(dataset_meta['num_keypoints'])
    ]

    prompts = []
    skipped_candidates = []
    for ann, prediction in zip(annotations, predictions):
        prompt, skipped = build_prompt_for_prediction(
            ann=ann,
            prediction=prediction,
            image_path=image_path,
            image_size=(image_width, image_height),
            args=args,
            keypoint_names=keypoint_names,
            use_udp=use_udp)
        if prompt is not None:
            prompts.append(prompt)
        if skipped is not None:
            skipped_candidates.append(skipped)

    prompts.sort(
        key=lambda item: (int(item.get('tooth_id_optional', 9999)),
                          int(item['candidate_id'])))

    return dict(
        image=dict(
            id=int(image_info['id']),
            file_name=image_info['file_name'],
            abs_path=str(image_path),
            width=int(image_info['width']),
            height=int(image_info['height']),
        ),
        model=dict(
            config=str(Path(args.config).resolve()),
            checkpoint=str(Path(args.checkpoint).resolve()),
            phase=args.phase,
            device=args.device,
            mask_thr=float(args.mask_thr),
        ),
        prompt_generation=dict(
            strategy='box_positive_default',
            box_expand_ratio=float(args.box_expand_ratio),
            max_keypoint_mask_distance=float(args.max_keypoint_mask_distance),
            min_positive_points=int(args.min_positive_points),
            negative_points='disabled_default',
        ),
        prompt_count=int(len(prompts)),
        skipped_count=int(len(skipped_candidates)),
        prompts=prompts,
        skipped_candidates=skipped_candidates,
    )


def main():
    args = parse_args()
    import_runtime_deps()
    cfg_options = parse_cfg_options(args.cfg_options)

    register_all_modules(init_default_scope=False)
    importlib.import_module('projects.panoramic_teeth')

    raw_cfg = Config.fromfile(args.config)
    requested_phase = 'test' if args.phase == 'auto' else args.phase
    prepare_phase = select_inference_phase(raw_cfg, requested_phase)
    cfg, use_udp = prepare_cfg(raw_cfg, prepare_phase, cfg_options)

    scope = cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    data_root = get_data_root(cfg)
    image_path = choose_target_image(args, data_root)
    resolved_phase, _, ann_path, ann_bundle, image_info = resolve_annotation_context(
        cfg, args, image_path)
    annotations = ann_bundle['anns_by_image'].get(image_info['id'], [])
    if not annotations:
        raise RuntimeError(f'No tooth annotations found for {image_path}.')

    model = init_model(cfg, args.checkpoint, device=args.device)
    payload = generate_prompts(
        model=model,
        image_path=image_path,
        image_info=image_info,
        annotations=annotations,
        args=args,
        use_udp=use_udp)
    payload['model']['resolved_phase'] = resolved_phase
    payload['model']['annotation_file'] = str(ann_path)
    payload['model']['inference_pipeline_phase'] = prepare_phase

    output_dir = Path(args.output_dir).resolve()
    mmengine.mkdir_or_exist(str(output_dir))
    stem = image_path.stem.replace(' ', '_')
    json_path = output_dir / f'{stem}_sam2_prompts.json'
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Selected image: {image_path}')
    print(f'Resolved phase: {resolved_phase}')
    print(f'Annotation file: {ann_path}')
    print(f'Annotations used: {len(annotations)}')
    print(f'Prompts saved: {payload["prompt_count"]}')
    print(f'Skipped candidates: {payload["skipped_count"]}')
    print(f'JSON saved to: {json_path}')


if __name__ == '__main__':
    main()
