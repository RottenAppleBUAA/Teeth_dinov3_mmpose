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
torch = None
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
    global cv2, mmengine, np, torch
    global Config, init_default_scope
    global inference_topdown, init_model
    global get_udp_warp_matrix, get_warp_matrix
    global register_all_modules

    import cv2 as _cv2
    import mmengine as _mmengine
    import numpy as _np
    import torch as _torch

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
    torch = _torch
    Config = _Config
    init_default_scope = _init_default_scope
    inference_topdown = _inference_topdown
    init_model = _init_model
    get_udp_warp_matrix = _get_udp_warp_matrix
    get_warp_matrix = _get_warp_matrix
    register_all_modules = _register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interactively select one panoramic image and export a '
        'single panoramic-level prediction JSON/visualization.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--phase',
        default='auto',
        choices=['auto', 'train', 'val', 'test'],
        help='Dataset split used to locate image annotations. Use auto to '
        'search train/val/test automatically.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory used to save panoramic-level outputs.')
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
        '--min-contour-area',
        type=float,
        default=20.0,
        help='Contours smaller than this value will be dropped when exporting '
        'segmentation polygons.')
    parser.add_argument(
        '--skip-empty',
        action='store_true',
        help='Skip teeth whose panoramic mask becomes empty after inverse '
        'warping.')
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

    prompt = ('Enter image path for panoramic inference '
              f'[{initial_dir}]: ').strip()
    user_input = input(prompt).strip()
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


def load_annotation_bundle(ann_file: Path) -> dict:
    with ann_file.open('r', encoding='utf-8') as f:
        bundle = json.load(f)

    images = {image['id']: image for image in bundle.get('images', [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in bundle.get('annotations', []):
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    for ann_list in anns_by_image.values():
        ann_list.sort(key=lambda ann: (int(ann.get('tooth_id', 9999)), ann['id']))

    return dict(
        raw=bundle,
        images=images,
        anns_by_image=anns_by_image,
    )


def choose_target_image(args, data_root: Path, images: Sequence[dict]) -> Path:
    if args.image:
        return Path(args.image).expanduser().resolve()
    return _select_image_interactively(data_root)


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
        f'Image {selected_path} was not found in the annotation file. '
        'Please choose an image that exists in the selected split.')


def extract_image_number(path: Path) -> Optional[int]:
    match = re.search(r'(\d+)', path.stem)
    if not match:
        return None
    return int(match.group(1))


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


def extract_segmentation(binary_mask: np.ndarray,
                         min_contour_area: float) -> tuple[list[list[float]],
                                                           float,
                                                           Optional[list[float]]]:
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    segmentations: list[list[float]] = []
    total_area = 0.0
    xs: list[float] = []
    ys: list[float] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_contour_area):
            continue
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 3:
            continue
        segmentations.append(points.reshape(-1).astype(float).tolist())
        total_area += area
        xs.extend(points[:, 0].tolist())
        ys.extend(points[:, 1].tolist())

    if not xs or not ys:
        return [], 0.0, None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
    return segmentations, total_area, bbox


def build_keypoint_payload(keypoints: np.ndarray,
                           scores: Optional[np.ndarray],
                           keypoint_names: Sequence[str]) -> list[dict]:
    payload = []
    for index, name in enumerate(keypoint_names):
        point = keypoints[index].astype(np.float32)
        score = None
        if scores is not None and index < len(scores):
            score = float(scores[index])
        payload.append(
            dict(name=name, x=float(point[0]), y=float(point[1]), score=score))
    return payload


def make_tooth_color(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (72, 201, 176),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (46, 204, 113),
        (230, 126, 34),
        (236, 112, 99),
    ]
    return palette[index % len(palette)]


def render_panoramic_overlay(image: np.ndarray,
                             teeth_predictions: Sequence[dict],
                             keypoint_colors: Sequence[Sequence[int]]) -> np.ndarray:
    canvas = image.copy()
    keypoint_palette = [
        tuple(int(v) for v in color[::-1]) for color in keypoint_colors
    ]

    for index, tooth in enumerate(teeth_predictions):
        color = make_tooth_color(index)
        mask = tooth['mask_binary']
        overlay = np.zeros_like(canvas, dtype=np.uint8)
        overlay[...] = np.asarray(color, dtype=np.uint8)
        canvas[mask > 0] = cv2.addWeighted(canvas, 0.68, overlay, 0.32,
                                           0)[mask > 0]

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

        for kp_index, point in enumerate(tooth['keypoints_xy']):
            point_int = tuple(np.round(point).astype(int))
            kp_color = keypoint_palette[kp_index % len(keypoint_palette)]
            cv2.circle(canvas, point_int, 4, kp_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, point_int, 5, (0, 0, 0), 1, cv2.LINE_AA)

        label_anchor = None
        if tooth['mask_bbox'] is not None:
            label_anchor = (
                int(round(tooth['mask_bbox'][0])),
                int(round(max(0.0, tooth['mask_bbox'][1] - 6))),
            )
        else:
            label_anchor = tuple(np.round(tooth['keypoints_xy'][0]).astype(int))

        cv2.putText(
            canvas,
            str(tooth['tooth_id']),
            label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA)

    return canvas


def export_panoramic_prediction(model,
                                image_path: Path,
                                image_info: dict,
                                annotations: Sequence[dict],
                                dataset_meta: dict,
                                args,
                                use_udp: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    bboxes = [ann['bbox'] for ann in annotations]
    predictions = inference_topdown(
        model, str(image_path), bboxes=bboxes, bbox_format='xywh')
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f'Failed to read image from {image_path}')

    image_height, image_width = image.shape[:2]
    panoramic_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    keypoint_names = [
        dataset_meta['keypoint_id2name'][index]
        for index in range(dataset_meta['num_keypoints'])
    ]

    teeth_predictions: list[dict] = []
    for ann, prediction in zip(annotations, predictions):
        pred_instances = prediction.pred_instances
        pred_fields = getattr(prediction, 'pred_fields', None)
        mask_crop = _mask_to_2d(getattr(pred_fields, 'root_mask', None))
        if mask_crop is None:
            continue

        mask_prob = unwarp_mask_to_image(
            mask_crop=mask_crop,
            data_sample=prediction,
            image_size=(image_width, image_height),
            use_udp=use_udp)
        mask_binary = (mask_prob >= float(args.mask_thr)).astype(np.uint8)
        segmentations, mask_area, mask_bbox = extract_segmentation(
            mask_binary, min_contour_area=args.min_contour_area)

        if args.skip_empty and not segmentations:
            continue

        keypoints_xy = _to_numpy(pred_instances.keypoints)
        if keypoints_xy.ndim == 3:
            keypoints_xy = keypoints_xy[0]
        keypoints_xy = keypoints_xy.astype(np.float32)

        keypoint_scores = _to_numpy(getattr(pred_instances, 'keypoint_scores',
                                            None))
        if keypoint_scores is not None and keypoint_scores.ndim == 2:
            keypoint_scores = keypoint_scores[0]

        panoramic_mask = np.maximum(panoramic_mask, mask_binary * 255)
        teeth_predictions.append(
            dict(
                annotation_id=int(ann['id']),
                tooth_id=int(ann.get('tooth_id', -1)),
                source_bbox_xywh=[float(v) for v in ann['bbox']],
                mask_bbox=mask_bbox,
                mask_area=float(mask_area),
                segmentation=segmentations,
                keypoints_xy=keypoints_xy,
                keypoint_scores=keypoint_scores,
                mask_binary=mask_binary,
            ))

    teeth_predictions.sort(key=lambda item: item['tooth_id'])

    overlay = render_panoramic_overlay(
        image=image,
        teeth_predictions=teeth_predictions,
        keypoint_colors=dataset_meta['keypoint_colors'])

    payload_teeth = []
    for tooth in teeth_predictions:
        payload_teeth.append(
            dict(
                annotation_id=tooth['annotation_id'],
                tooth_id=tooth['tooth_id'],
                source_bbox_xywh=tooth['source_bbox_xywh'],
                pred_mask_bbox_xywh=tooth['mask_bbox'],
                pred_mask_area=tooth['mask_area'],
                segmentation=tooth['segmentation'],
                keypoints=build_keypoint_payload(
                    tooth['keypoints_xy'],
                    tooth['keypoint_scores'],
                    keypoint_names),
            ))

    payload = dict(
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
        keypoint_order=list(keypoint_names),
        panoramic_prediction=dict(
            num_teeth=len(payload_teeth),
            union_mask_area=int((panoramic_mask > 0).sum()),
            teeth=payload_teeth,
        ),
    )
    return payload, overlay, panoramic_mask


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


def resolve_annotation_context(cfg: Config, args,
                               image_path: Path) -> tuple[str, Path, Path, dict,
                                                          dict]:
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


def main():
    args = parse_args()
    import_runtime_deps()
    cfg_options = parse_cfg_options(args.cfg_options)

    register_all_modules(init_default_scope=False)
    importlib.import_module('projects.panoramic_teeth')

    raw_cfg = Config.fromfile(args.config)
    prepare_phase = 'test' if args.phase == 'auto' else args.phase
    cfg, use_udp = prepare_cfg(raw_cfg, prepare_phase, cfg_options)

    scope = cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    data_root = get_data_root(cfg)
    image_path = choose_target_image(args, data_root, ())
    resolved_phase, _, ann_path, ann_bundle, image_info = resolve_annotation_context(
        cfg, args, image_path)
    annotations = ann_bundle['anns_by_image'].get(image_info['id'], [])
    if not annotations:
        raise RuntimeError(f'No tooth annotations found for {image_path}.')

    model = init_model(cfg, args.checkpoint, device=args.device)
    payload, overlay, panoramic_mask = export_panoramic_prediction(
        model=model,
        image_path=image_path,
        image_info=image_info,
        annotations=annotations,
        dataset_meta=model.dataset_meta,
        args=args,
        use_udp=use_udp)

    output_dir = Path(args.output_dir).resolve()
    mmengine.mkdir_or_exist(str(output_dir))
    stem = image_path.stem.replace(' ', '_')
    json_path = output_dir / f'{stem}_panoramic_prediction.json'
    overlay_path = output_dir / f'{stem}_panoramic_overlay.png'
    mask_path = output_dir / f'{stem}_panoramic_mask.png'

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f'Failed to save overlay image to {overlay_path}')
    if not cv2.imwrite(str(mask_path), panoramic_mask):
        raise RuntimeError(f'Failed to save mask image to {mask_path}')

    print(f'Selected image: {image_path}')
    print(f'Resolved phase: {resolved_phase}')
    print(f'Annotation file: {ann_path}')
    print(f'Annotations used: {len(annotations)}')
    print(f'JSON saved to: {json_path}')
    print(f'Overlay saved to: {overlay_path}')
    print(f'Mask saved to: {mask_path}')


if __name__ == '__main__':
    main()
