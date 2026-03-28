from __future__ import annotations

import argparse
import ast
import importlib
import json
import os.path as osp
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import mmengine
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown, init_model
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
from mmpose.utils import register_all_modules


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

SEMANTIC_COLORS = dict(
    mesial=(255, 178, 0),
    distal=(82, 82, 255),
    apex=(10, 214, 255),
    gt_mask=(100, 220, 120),
    pred_mask=(70, 150, 255),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize predictions from the panoramic_teeth_structured '
        'project. Supports both single-image and batch image-level export.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory used to save overlays and JSON payloads.')
    parser.add_argument(
        '--image',
        default=None,
        help='Optional original panoramic image path. If set, run in single-'
        'image mode; otherwise run in batch mode.')
    parser.add_argument(
        '--phase',
        default='test',
        choices=['auto', 'train', 'val', 'test'],
        help='Annotation split to use. In single-image mode, auto searches '
        'train/val/test. In batch mode, use train/val/test.')
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
        help='Threshold used to binarize predicted root masks.')
    parser.add_argument(
        '--pred-kpt-thr',
        type=float,
        default=0.0,
        help='Threshold used to draw predicted keypoints.')
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start image index for batch export.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to export in batch mode.')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip image outputs that already exist.')
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


def _ensure_bgr_uint8(image) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8).copy()


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
        try:
            value = ast.literal_eval(raw_value)
        except Exception:
            value = raw_value
        parsed[key.strip()] = value
    return parsed


def _find_topdown_affine_cfg(pipeline_cfg: Iterable[dict]) -> Optional[dict]:
    for transform in pipeline_cfg:
        if transform.get('type') == 'TopdownAffine':
            return transform
    return None


def _get_reference_dataset_cfg(cfg: Config):
    for split in ('test', 'val', 'train'):
        dataloader_cfg = cfg.get(f'{split}_dataloader', None)
        if dataloader_cfg is not None:
            return deepcopy(dataloader_cfg.dataset)
    raise RuntimeError('Failed to locate any dataset config in the model config.')


def prepare_cfg(cfg: Config, cfg_options: dict) -> tuple[Config, bool]:
    cfg = deepcopy(cfg)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    dataset_cfg = _get_reference_dataset_cfg(cfg)
    pipeline_cfg = deepcopy(dataset_cfg.pipeline)
    if not pipeline_cfg:
        raise ValueError('Inference pipeline is empty.')

    inference_pipeline = []
    for transform in pipeline_cfg:
        transform_type = transform.get('type')
        if transform_type in {
                'GenerateStructuredToothTargets', 'GenerateAnatomicalToothTargets',
                'GenerateAnatomicalPointMaskTargets', 'GenerateTarget'
        }:
            continue
        if transform_type in {'RandomFlip', 'RandomBBoxTransform'}:
            continue
        inference_pipeline.append(transform)

    if not inference_pipeline:
        raise ValueError('Inference pipeline became empty after filtering.')

    pack_cfg = inference_pipeline[-1]
    expected_pack_types = {
        'PackStructuredToothInputs', 'PackAnatomicalToothInputs',
        'PackAnatomicalPointMaskInputs'
    }
    if pack_cfg.get('type') not in expected_pack_types:
        raise ValueError('The last transform in the inference pipeline must '
                         f'be one of {sorted(expected_pack_types)}, but got '
                         f'{pack_cfg.get("type")}.')

    if 'meta_keys' in pack_cfg:
        meta_keys = list(pack_cfg['meta_keys'])
        if 'bbox_rotation' not in meta_keys:
            meta_keys.append('bbox_rotation')
        pack_cfg['meta_keys'] = tuple(meta_keys)
    pack_cfg['pack_transformed'] = False

    cfg.test_dataloader.dataset.pipeline = inference_pipeline
    cfg.model.test_cfg = deepcopy(cfg.get('model', {}).get('test_cfg', {}))

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

    return dict(raw=bundle, images=images, anns_by_image=anns_by_image)


def extract_image_number(path: Path) -> Optional[int]:
    match = re.search(r'(\d+)', path.stem)
    if not match:
        return None
    return int(match.group(1))


def match_image_info(selected_path: Path, data_root: Path,
                     images: Sequence[dict]) -> dict:
    selected_norm = _normalize_path(selected_path)
    selected_number = extract_image_number(selected_path)
    exact_matches = []
    basename_matches = []
    number_matches = []

    for image_info in images:
        candidate_path = (data_root / image_info['file_name']).resolve()
        if _normalize_path(candidate_path) == selected_norm:
            exact_matches.append(image_info)
        elif candidate_path.name == selected_path.name:
            basename_matches.append(image_info)
        else:
            candidate_number = extract_image_number(candidate_path)
            if (selected_number is not None
                    and candidate_number == selected_number):
                number_matches.append(image_info)

    for matches in (exact_matches, basename_matches, number_matches):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f'Multiple images matched {selected_path}.')

    raise FileNotFoundError(
        f'Image {selected_path} was not found in the annotation file.')


def get_data_root(cfg: Config) -> Path:
    dataset_cfg = _get_reference_dataset_cfg(cfg)
    return _resolve_from_repo(dataset_cfg.data_root)


def ensure_annotation_path(cfg: Config, phase: str,
                           ann_file_override: Optional[str]) -> tuple[Path, Path]:
    dataloader_cfg = cfg.get(f'{phase}_dataloader', None)
    if dataloader_cfg is None:
        raise KeyError(f'Config does not define {phase}_dataloader.')
    dataset_cfg = dataloader_cfg.dataset
    data_root = _resolve_from_repo(dataset_cfg.data_root)
    ann_file = ann_file_override or dataset_cfg.ann_file
    ann_path = Path(ann_file)
    if not ann_path.is_absolute():
        ann_path = (data_root / ann_file).resolve()
    return data_root, ann_path


def resolve_annotation_context(cfg: Config, phase: str,
                               ann_file_override: Optional[str],
                               image_path: Optional[Path] = None):
    if image_path is None:
        if phase == 'auto':
            phase = 'test'
        data_root, ann_path = ensure_annotation_path(cfg, phase, ann_file_override)
        bundle = load_annotation_bundle(ann_path)
        return phase, data_root, ann_path, bundle, None

    if ann_file_override is not None:
        resolved_phase = 'test' if phase == 'auto' else phase
        data_root, ann_path = ensure_annotation_path(cfg, resolved_phase,
                                                     ann_file_override)
        bundle = load_annotation_bundle(ann_path)
        image_info = match_image_info(image_path, data_root,
                                      bundle['images'].values())
        return resolved_phase, data_root, ann_path, bundle, image_info

    search_phases = ('train', 'val', 'test') if phase == 'auto' else (phase, )
    failures = []
    for candidate_phase in search_phases:
        data_root, ann_path = ensure_annotation_path(cfg, candidate_phase, None)
        if not ann_path.exists():
            failures.append(f'{candidate_phase}: missing annotation file')
            continue
        bundle = load_annotation_bundle(ann_path)
        try:
            image_info = match_image_info(image_path, data_root,
                                          bundle['images'].values())
        except FileNotFoundError:
            failures.append(f'{candidate_phase}: image not present')
            continue
        return candidate_phase, data_root, ann_path, bundle, image_info

    raise FileNotFoundError(
        f'Image {image_path} was not found in train/val/test annotations. '
        f'Details: {"; ".join(failures)}')


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
    rot = 0.0 if rotation is None else float(rotation.reshape(-1)[0])
    if use_udp:
        return get_udp_warp_matrix(
            center, scale, rot, output_size=(input_width, input_height))
    return get_warp_matrix(
        center, scale, rot, output_size=(input_width, input_height))


def transform_crop_points_to_image(points_crop, data_sample,
                                   use_udp: bool) -> np.ndarray:
    points_crop = _to_numpy(points_crop)
    if points_crop is None:
        return np.zeros((0, 2), dtype=np.float32)
    points_crop = np.asarray(points_crop, dtype=np.float32)
    if points_crop.ndim == 3:
        points_crop = points_crop[0]
    if points_crop.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    inv_warp = cv2.invertAffineTransform(_build_warp_matrix(data_sample, use_udp))
    return cv2.transform(points_crop[None, ...], inv_warp)[0]


def unwarp_mask_to_image(mask_crop: np.ndarray,
                         data_sample,
                         image_size: tuple[int, int],
                         use_udp: bool) -> np.ndarray:
    width, height = image_size
    inv_warp = cv2.invertAffineTransform(_build_warp_matrix(data_sample, use_udp))
    return cv2.warpAffine(
        mask_crop.astype(np.float32),
        inv_warp,
        dsize=(int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0)


def segmentation_to_mask(segmentation: Sequence[Sequence[float]],
                         image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


def decode_ann_keypoints(ann: dict) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(ann.get('keypoints', []), dtype=np.float32).reshape(-1, 3)
    if values.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, ),
                                                             dtype=np.float32)
    return values[:, :2], values[:, 2]


def derive_polylines_from_keypoints(keypoints: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if len(keypoints) < 5:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32))
    mesial = keypoints[[0, 1, 2]].astype(np.float32)
    distal = keypoints[[4, 3, 2]].astype(np.float32)
    return mesial, distal


def build_gt_records(annotations: Sequence[dict],
                     image_size: tuple[int, int]) -> list[dict]:
    records = []
    for ann in annotations:
        keypoints, visibility = decode_ann_keypoints(ann)
        side_contours = ann.get('side_contours', {})
        mesial_polyline, distal_polyline = derive_polylines_from_keypoints(
            keypoints)
        mesial_contour = np.asarray(
            side_contours.get('M', []), dtype=np.float32).reshape(-1, 2)
        distal_contour = np.asarray(
            side_contours.get('D', []), dtype=np.float32).reshape(-1, 2)
        if len(mesial_contour) < 2:
            mesial_contour = mesial_polyline
        if len(distal_contour) < 2:
            distal_contour = distal_polyline
        records.append(
            dict(
                annotation_id=int(ann['id']),
                tooth_id=int(ann.get('tooth_id', -1)),
                bbox_xywh=[float(v) for v in ann.get('bbox', [])],
                keypoints_xy=keypoints.astype(np.float32),
                keypoint_scores=visibility.astype(np.float32),
                mesial_contour=mesial_contour,
                distal_contour=distal_contour,
                mesial_polyline=mesial_polyline,
                distal_polyline=distal_polyline,
                mask_binary=segmentation_to_mask(ann.get('segmentation', []),
                                                 image_size),
            ))
    return records


def build_pred_records(predictions,
                       annotations: Sequence[dict],
                       image_size: tuple[int, int],
                       use_udp: bool,
                       mask_thr: float) -> list[dict]:
    records = []
    for ann, prediction in zip(annotations, predictions):
        pred_instances = prediction.pred_instances
        pred_fields = getattr(prediction, 'pred_fields', None)
        root_mask_crop = _mask_to_2d(getattr(pred_fields, 'root_mask', None))
        if root_mask_crop is None:
            root_mask_prob = np.zeros((image_size[1], image_size[0]),
                                      dtype=np.float32)
        else:
            root_mask_prob = unwarp_mask_to_image(
                root_mask_crop, prediction, image_size, use_udp)
        root_mask_binary = (root_mask_prob >= float(mask_thr)).astype(np.uint8)

        keypoints_xy = _to_numpy(pred_instances.keypoints)
        if keypoints_xy.ndim == 3:
            keypoints_xy = keypoints_xy[0]

        keypoint_scores = _to_numpy(getattr(pred_instances, 'keypoint_scores',
                                            None))
        if keypoint_scores is not None and keypoint_scores.ndim == 2:
            keypoint_scores = keypoint_scores[0]

        mesial_contour = transform_crop_points_to_image(
            getattr(pred_instances, 'mesial_contour', None), prediction, use_udp)
        distal_contour = transform_crop_points_to_image(
            getattr(pred_instances, 'distal_contour', None), prediction, use_udp)
        mesial_polyline, distal_polyline = derive_polylines_from_keypoints(
            keypoints_xy)
        if len(mesial_contour) < 2:
            mesial_contour = mesial_polyline
        if len(distal_contour) < 2:
            distal_contour = distal_polyline

        records.append(
            dict(
                annotation_id=int(ann['id']),
                tooth_id=int(ann.get('tooth_id', -1)),
                bbox_xywh=[float(v) for v in ann.get('bbox', [])],
                keypoints_xy=np.asarray(keypoints_xy, dtype=np.float32),
                keypoint_scores=None if keypoint_scores is None else
                np.asarray(keypoint_scores, dtype=np.float32),
                mesial_contour=mesial_contour.astype(np.float32),
                distal_contour=distal_contour.astype(np.float32),
                mesial_polyline=mesial_polyline.astype(np.float32),
                distal_polyline=distal_polyline.astype(np.float32),
                mask_binary=root_mask_binary,
            ))
    return records


def overlay_mask(image: np.ndarray,
                 mask_binary: np.ndarray,
                 color: Sequence[int],
                 alpha: float = 0.30) -> np.ndarray:
    rendered = image.copy()
    if mask_binary is None or not np.any(mask_binary > 0):
        return rendered
    overlay = np.zeros_like(rendered, dtype=np.uint8)
    overlay[...] = np.asarray(color, dtype=np.uint8)
    positive = mask_binary > 0
    rendered[positive] = cv2.addWeighted(rendered, 1 - alpha, overlay, alpha,
                                         0)[positive]
    contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rendered, contours, -1, tuple(int(c) for c in color), 1)
    return rendered


def draw_polyline(image: np.ndarray,
                  points: np.ndarray,
                  color: Sequence[int],
                  thickness: int = 2) -> np.ndarray:
    rendered = image.copy()
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return rendered
    polyline = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    color = tuple(int(v) for v in color)
    cv2.polylines(
        rendered, [polyline], isClosed=False, color=(0, 0, 0),
        thickness=thickness + 2, lineType=cv2.LINE_AA)
    cv2.polylines(
        rendered, [polyline], isClosed=False, color=color, thickness=thickness,
        lineType=cv2.LINE_AA)
    return rendered


def draw_keypoints(image: np.ndarray,
                   keypoints: np.ndarray,
                   keypoint_names: Sequence[str],
                   semantic_colors: Sequence[Sequence[int]],
                   keypoint_scores: Optional[np.ndarray] = None,
                   score_thr: float = 0.0) -> np.ndarray:
    rendered = image.copy()
    for index, point in enumerate(np.asarray(keypoints, dtype=np.float32)):
        if keypoint_scores is not None and index < len(keypoint_scores):
            if keypoint_scores[index] < score_thr:
                continue
        color = tuple(int(v) for v in semantic_colors[index])
        x, y = np.round(point).astype(int)
        cv2.circle(rendered, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(rendered, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
        if index < len(keypoint_names):
            cv2.putText(rendered, keypoint_names[index], (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return rendered


def add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    rendered = image.copy()
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(rendered, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


def render_teeth_overlay(image: np.ndarray,
                         records: Sequence[dict],
                         keypoint_names: Sequence[str],
                         semantic_colors: Sequence[Sequence[int]],
                         mask_color: Sequence[int],
                         pred_kpt_thr: float = 0.0) -> np.ndarray:
    rendered = image.copy()
    for record in records:
        rendered = overlay_mask(rendered, record.get('mask_binary'), mask_color)
        rendered = draw_polyline(rendered, record.get('mesial_contour', []),
                                 SEMANTIC_COLORS['mesial'])
        rendered = draw_polyline(rendered, record.get('distal_contour', []),
                                 SEMANTIC_COLORS['distal'])
        rendered = draw_keypoints(
            rendered,
            record.get('keypoints_xy', []),
            keypoint_names=keypoint_names,
            semantic_colors=semantic_colors,
            keypoint_scores=record.get('keypoint_scores'),
            score_thr=pred_kpt_thr)

        label_anchor = None
        bbox = record.get('bbox_xywh', None)
        if bbox and len(bbox) == 4:
            label_anchor = (int(round(bbox[0])),
                            int(round(max(0.0, bbox[1] - 6.0))))
        elif len(record.get('keypoints_xy', [])) > 0:
            label_anchor = tuple(
                np.round(record['keypoints_xy'][0]).astype(int).tolist())
        if label_anchor is not None:
            cv2.putText(
                rendered,
                str(record.get('tooth_id', 'na')),
                label_anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                tuple(int(v) for v in mask_color),
                2,
                cv2.LINE_AA)
    return rendered


def make_image_level_visualization_panel(image: np.ndarray,
                                         gt_records: Sequence[dict],
                                         pred_records: Sequence[dict],
                                         keypoint_names: Sequence[str],
                                         semantic_colors: Sequence[Sequence[int]],
                                         pred_kpt_thr: float) -> np.ndarray:
    image = _ensure_bgr_uint8(image)
    base = image.copy()
    gt_panel = render_teeth_overlay(
        image, gt_records, keypoint_names, semantic_colors,
        mask_color=SEMANTIC_COLORS['gt_mask'], pred_kpt_thr=0.0)
    pred_panel = render_teeth_overlay(
        image, pred_records, keypoint_names, semantic_colors,
        mask_color=SEMANTIC_COLORS['pred_mask'], pred_kpt_thr=pred_kpt_thr)
    compare_panel = render_teeth_overlay(
        gt_panel, pred_records, keypoint_names, semantic_colors,
        mask_color=SEMANTIC_COLORS['pred_mask'], pred_kpt_thr=pred_kpt_thr)

    top = np.concatenate(
        [add_panel_title(base, 'Image'),
         add_panel_title(gt_panel, 'GT')],
        axis=1)
    bottom = np.concatenate(
        [add_panel_title(pred_panel, 'Prediction'),
         add_panel_title(compare_panel, 'GT + Prediction')],
        axis=1)
    return np.concatenate([top, bottom], axis=0)


def build_keypoint_payload(keypoints: np.ndarray,
                           scores: Optional[np.ndarray],
                           keypoint_names: Sequence[str]) -> list[dict]:
    payload = []
    for index, name in enumerate(keypoint_names):
        if index >= len(keypoints):
            break
        point = np.asarray(keypoints[index], dtype=np.float32)
        score = None
        if scores is not None and index < len(scores):
            score = float(scores[index])
        payload.append(
            dict(name=name, x=float(point[0]), y=float(point[1]), score=score))
    return payload


def serialize_records(records: Sequence[dict],
                      keypoint_names: Sequence[str]) -> list[dict]:
    payload = []
    for record in records:
        payload.append(
            dict(
                annotation_id=int(record['annotation_id']),
                tooth_id=int(record['tooth_id']),
                bbox_xywh=[float(v) for v in record.get('bbox_xywh', [])],
                keypoints=build_keypoint_payload(
                    record.get('keypoints_xy', []),
                    record.get('keypoint_scores'), keypoint_names),
                mesial_polyline=np.asarray(
                    record.get('mesial_polyline', record.get('mesial_contour', [])),
                    dtype=np.float32).reshape(-1, 2).astype(float).tolist(),
                distal_polyline=np.asarray(
                    record.get('distal_polyline', record.get('distal_contour', [])),
                    dtype=np.float32).reshape(-1, 2).astype(float).tolist(),
                mesial_contour=np.asarray(
                    record.get('mesial_contour', []),
                    dtype=np.float32).reshape(-1, 2).astype(float).tolist(),
                distal_contour=np.asarray(
                    record.get('distal_contour', []),
                    dtype=np.float32).reshape(-1, 2).astype(float).tolist(),
                mask_area=int(np.asarray(record.get('mask_binary', 0)).sum()),
            ))
    return payload


def build_output_stem(index: int, image_path: Path, image_info: dict) -> str:
    image_id = image_info.get('id', index)
    stem = image_path.stem.replace(' ', '_')
    return f'{index:06d}_img{image_id}_{stem}'


def export_one_image(model,
                     image_path: Path,
                     image_info: dict,
                     annotations: Sequence[dict],
                     args,
                     use_udp: bool) -> tuple[dict, np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f'Failed to read image from {image_path}')

    predictions = inference_topdown(
        model, str(image_path), bboxes=[ann['bbox'] for ann in annotations],
        bbox_format='xywh')
    image_size = (image.shape[1], image.shape[0])
    gt_records = build_gt_records(annotations, image_size)
    pred_records = build_pred_records(
        predictions=predictions,
        annotations=annotations,
        image_size=image_size,
        use_udp=use_udp,
        mask_thr=args.mask_thr)

    keypoint_names = [
        model.dataset_meta['keypoint_id2name'][index]
        for index in range(model.dataset_meta['num_keypoints'])
    ]
    semantic_colors = [
        tuple(int(v) for v in color[::-1])
        for color in model.dataset_meta['keypoint_colors']
    ]
    panel = make_image_level_visualization_panel(
        image=image,
        gt_records=gt_records,
        pred_records=pred_records,
        keypoint_names=keypoint_names,
        semantic_colors=semantic_colors,
        pred_kpt_thr=args.pred_kpt_thr)

    payload = dict(
        image=dict(
            id=int(image_info['id']),
            file_name=image_info['file_name'],
            abs_path=str(image_path.resolve()),
            width=int(image_info['width']),
            height=int(image_info['height']),
        ),
        model=dict(
            config=str(Path(args.config).resolve()),
            checkpoint=str(Path(args.checkpoint).resolve()),
            phase=args.phase,
            device=args.device,
            mask_thr=float(args.mask_thr),
            pred_kpt_thr=float(args.pred_kpt_thr),
        ),
        keypoint_order=keypoint_names,
        num_annotations=len(annotations),
        gt=serialize_records(gt_records, keypoint_names),
        prediction=serialize_records(pred_records, keypoint_names),
    )
    return payload, panel


def save_visualization(output_dir: Path,
                       stem: str,
                       payload: dict,
                       panel: np.ndarray) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / f'{stem}_structured_panel.png'
    json_path = output_dir / f'{stem}_structured_prediction.json'
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if not cv2.imwrite(str(panel_path), panel):
        raise RuntimeError(f'Failed to save visualization to {panel_path}')
    return panel_path, json_path


def run_single_image(model, cfg: Config, args, use_udp: bool) -> None:
    image_path = Path(args.image).expanduser().resolve()
    resolved_phase, _, ann_path, bundle, image_info = resolve_annotation_context(
        cfg, args.phase, args.ann_file, image_path)
    annotations = bundle['anns_by_image'].get(image_info['id'], [])
    if not annotations:
        raise RuntimeError(f'No tooth annotations found for {image_path}.')
    payload, panel = export_one_image(
        model=model,
        image_path=image_path,
        image_info=image_info,
        annotations=annotations,
        args=args,
        use_udp=use_udp)
    stem = build_output_stem(0, image_path, image_info)
    panel_path, json_path = save_visualization(Path(args.output_dir), stem,
                                               payload, panel)
    print(f'Selected image: {image_path}')
    print(f'Resolved phase: {resolved_phase}')
    print(f'Annotation file: {ann_path}')
    print(f'Annotations used: {len(annotations)}')
    print(f'Panel saved to: {panel_path}')
    print(f'JSON saved to: {json_path}')


def iter_batch_images(bundle: dict) -> list[tuple[dict, list[dict]]]:
    items = []
    for image_id in sorted(bundle['images']):
        image_info = bundle['images'][image_id]
        annotations = bundle['anns_by_image'].get(image_id, [])
        if annotations:
            items.append((image_info, annotations))
    return items


def run_batch_export(model, cfg: Config, args, use_udp: bool) -> None:
    if args.phase == 'auto':
        raise ValueError('Batch mode does not support --phase auto. '
                         'Use train, val or test.')

    resolved_phase, data_root, ann_path, bundle, _ = resolve_annotation_context(
        cfg, args.phase, args.ann_file, image_path=None)
    items = iter_batch_images(bundle)
    start = int(args.start_index)
    end = len(items) if args.max_images is None else min(
        len(items), start + int(args.max_images))
    progress = mmengine.ProgressBar(max(0, end - start))
    output_dir = Path(args.output_dir)

    for index in range(start, end):
        image_info, annotations = items[index]
        image_path = (data_root / image_info['file_name']).resolve()
        stem = build_output_stem(index, image_path, image_info)
        panel_path = output_dir / f'{stem}_structured_panel.png'
        if args.skip_existing and panel_path.exists():
            progress.update()
            continue

        payload, panel = export_one_image(
            model=model,
            image_path=image_path,
            image_info=image_info,
            annotations=annotations,
            args=args,
            use_udp=use_udp)
        save_visualization(output_dir, stem, payload, panel)
        progress.update()

    print(f'Batch phase: {resolved_phase}')
    print(f'Annotation file: {ann_path}')
    print(f'Output dir: {output_dir.resolve()}')
    print(f'Images exported: {max(0, end - start)}')


def main():
    args = parse_args()
    cfg_options = parse_cfg_options(args.cfg_options)

    register_all_modules(init_default_scope=False)
    importlib.import_module('projects.panoramic_teeth_structured')

    raw_cfg = Config.fromfile(args.config)
    cfg, use_udp = prepare_cfg(raw_cfg, cfg_options)

    scope = cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    model = init_model(cfg, args.checkpoint, device=args.device)

    if args.image:
        run_single_image(model, cfg, args, use_udp)
    else:
        run_batch_export(model, cfg, args, use_udp)


if __name__ == '__main__':
    main()
