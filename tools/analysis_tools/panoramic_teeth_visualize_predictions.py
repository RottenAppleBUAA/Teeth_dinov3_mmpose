from __future__ import annotations

import argparse
import importlib
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import mmengine
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.registry import build_from_cfg, init_default_scope


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

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmpose.apis import init_model
from mmpose.registry import DATASETS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
from mmpose.utils import register_all_modules

DEFAULT_PACK_META_KEYS = (
    'id', 'img_id', 'img_path', 'category_id', 'ori_shape', 'img_shape',
    'input_size', 'input_center', 'input_scale', 'flip', 'flip_direction',
    'flip_indices', 'raw_ann_info', 'dataset_name', 'tooth_id')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize panoramic-teeth predictions with keypoints and '
        'root masks.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--phase',
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split used to generate visualizations.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory used to save rendered visualizations.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='Maximum number of samples to visualize.')
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start visualizing from this dataset index.')
    parser.add_argument(
        '--mask-thr',
        type=float,
        default=0.5,
        help='Threshold used to binarize the predicted mask.')
    parser.add_argument(
        '--pred-kpt-thr',
        type=float,
        default=0.0,
        help='Threshold used to draw predicted keypoints.')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip samples whose output file already exists.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
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
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image.copy()


def _normalize_color(color: Sequence[int]) -> tuple[int, int, int]:
    color = np.asarray(color, dtype=np.uint8).reshape(-1)
    if color.size < 3:
        color = np.pad(color, (0, 3 - color.size), constant_values=255)
    return int(color[2]), int(color[1]), int(color[0])


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


def _transform_points(points, warp_mat: np.ndarray) -> np.ndarray:
    points = _to_numpy(points)
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3:
        points = points[0]
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return cv2.transform(points[None, ...], warp_mat)[0]


def _extract_gt_keypoints_crop(data_sample, warp_mat: np.ndarray) -> np.ndarray:
    gt_instances = data_sample.gt_instances
    if hasattr(gt_instances, 'transformed_keypoints'):
        return _to_numpy(gt_instances.transformed_keypoints)[0].astype(np.float32)
    return _transform_points(gt_instances.keypoints, warp_mat)


def _extract_pred_keypoints_crop(data_sample, warp_mat: np.ndarray) -> np.ndarray:
    if not hasattr(data_sample, 'pred_instances'):
        return np.zeros((0, 2), dtype=np.float32)
    pred_keypoints = _to_numpy(data_sample.pred_instances.keypoints)
    return _transform_points(pred_keypoints, warp_mat)


def _extract_pred_scores(data_sample) -> Optional[np.ndarray]:
    if not hasattr(data_sample, 'pred_instances'):
        return None
    scores = getattr(data_sample.pred_instances, 'keypoint_scores', None)
    scores = _to_numpy(scores)
    if scores is None:
        return None
    if scores.ndim == 2:
        scores = scores[0]
    return scores.astype(np.float32)


def _extract_gt_visible(data_sample) -> Optional[np.ndarray]:
    visible = getattr(data_sample.gt_instances, 'keypoints_visible', None)
    visible = _to_numpy(visible)
    if visible is None:
        return None
    if visible.ndim == 2:
        visible = visible[0]
    if visible.ndim == 3:
        visible = visible[0, :, 0]
    return visible.astype(np.float32)


def overlay_mask(image: np.ndarray,
                 mask: Optional[np.ndarray],
                 color: Sequence[int],
                 threshold: float,
                 alpha: float = 0.35) -> np.ndarray:
    if mask is None:
        return image.copy()
    binary = mask >= float(threshold)
    rendered = image.copy()
    if not np.any(binary):
        return rendered

    overlay = np.zeros_like(rendered, dtype=np.uint8)
    overlay[...] = np.asarray(color, dtype=np.uint8)
    rendered[binary] = cv2.addWeighted(rendered, 1 - alpha, overlay, alpha,
                                       0)[binary]

    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rendered, contours, -1, tuple(int(c) for c in color), 1)
    return rendered


def draw_keypoints(image: np.ndarray,
                   keypoints: np.ndarray,
                   labels: Sequence[str],
                   colors: Sequence[Sequence[int]],
                   skeleton_links: Sequence[Sequence[int]],
                   scores: Optional[np.ndarray] = None,
                   score_thr: float = 0.0,
                   visible: Optional[np.ndarray] = None,
                   draw_labels: bool = True) -> np.ndarray:
    rendered = image.copy()
    if keypoints.size == 0:
        return rendered

    def _is_drawn(index: int) -> bool:
        if scores is not None and index < len(scores) and scores[index] < score_thr:
            return False
        if visible is not None and index < len(visible) and visible[index] <= 0:
            return False
        return True

    for start, end in skeleton_links:
        if start >= len(keypoints) or end >= len(keypoints):
            continue
        if not _is_drawn(start) or not _is_drawn(end):
            continue
        start_xy = tuple(np.round(keypoints[start]).astype(int))
        end_xy = tuple(np.round(keypoints[end]).astype(int))
        cv2.line(rendered, start_xy, end_xy, (255, 255, 255), 1, cv2.LINE_AA)

    for index, point in enumerate(keypoints):
        if not _is_drawn(index):
            continue
        color = tuple(int(c) for c in colors[index])
        x, y = np.round(point).astype(int)
        cv2.circle(rendered, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(rendered, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
        if draw_labels and index < len(labels):
            cv2.putText(rendered, labels[index], (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return rendered


def add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    rendered = image.copy()
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(rendered, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


def make_visualization_panel(image: np.ndarray,
                             gt_mask: Optional[np.ndarray],
                             pred_mask: Optional[np.ndarray],
                             gt_keypoints: np.ndarray,
                             pred_keypoints: np.ndarray,
                             keypoint_labels: Sequence[str],
                             keypoint_colors: Sequence[Sequence[int]],
                             skeleton_links: Sequence[Sequence[int]],
                             pred_scores: Optional[np.ndarray],
                             gt_visible: Optional[np.ndarray],
                             mask_thr: float,
                             pred_kpt_thr: float) -> np.ndarray:
    image = _ensure_bgr_uint8(image)

    keypoint_colors = [_normalize_color(color) for color in keypoint_colors]
    gt_panel = overlay_mask(image, gt_mask, color=(80, 220, 120), threshold=0.5)
    gt_panel = draw_keypoints(
        gt_panel,
        gt_keypoints,
        labels=keypoint_labels,
        colors=keypoint_colors,
        skeleton_links=skeleton_links,
        visible=gt_visible,
        draw_labels=True)

    pred_panel = overlay_mask(
        image, pred_mask, color=(70, 150, 255), threshold=mask_thr)
    pred_panel = draw_keypoints(
        pred_panel,
        pred_keypoints,
        labels=keypoint_labels,
        colors=keypoint_colors,
        skeleton_links=skeleton_links,
        scores=pred_scores,
        score_thr=pred_kpt_thr,
        draw_labels=True)

    compare_panel = gt_panel.copy()
    compare_panel = overlay_mask(
        compare_panel, pred_mask, color=(70, 150, 255), threshold=mask_thr,
        alpha=0.25)
    compare_panel = draw_keypoints(
        compare_panel,
        pred_keypoints,
        labels=keypoint_labels,
        colors=[(255, 255, 255)] * len(keypoint_colors),
        skeleton_links=skeleton_links,
        scores=pred_scores,
        score_thr=pred_kpt_thr,
        draw_labels=False)

    base_panel = image.copy()

    panels = [
        add_panel_title(base_panel, 'Crop'),
        add_panel_title(gt_panel, 'GT'),
        add_panel_title(pred_panel, 'Prediction'),
        add_panel_title(compare_panel, 'Overlay')
    ]
    return np.concatenate(panels, axis=1)


def _find_topdown_affine_cfg(pipeline_cfg: Iterable) -> Optional[dict]:
    for transform in pipeline_cfg:
        if transform.get('type') == 'TopdownAffine':
            return transform
    return None


def prepare_cfg(cfg: Config, phase: str, cfg_options: dict) -> tuple[Config, bool]:
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    dataloader_cfg = cfg[f'{phase}_dataloader']
    pipeline_cfg = dataloader_cfg.dataset.pipeline
    if not pipeline_cfg:
        raise ValueError(f'{phase}_dataloader.dataset.pipeline is empty.')

    pack_cfg = pipeline_cfg[-1]
    pack_cfg.pack_transformed = True

    meta_keys = list(pack_cfg.get('meta_keys', DEFAULT_PACK_META_KEYS))
    if 'bbox_rotation' not in meta_keys:
        meta_keys.append('bbox_rotation')
    pack_cfg.meta_keys = tuple(meta_keys)

    affine_cfg = _find_topdown_affine_cfg(pipeline_cfg)
    use_udp = bool(affine_cfg.get('use_udp', False)) if affine_cfg else False
    return cfg, use_udp


def build_dataset(cfg: Config, phase: str):
    return build_from_cfg(cfg[f'{phase}_dataloader'].dataset, DATASETS)


def build_output_name(index: int, data_sample) -> str:
    img_path = Path(data_sample.img_path)
    ann_id = data_sample.metainfo.get('id', index)
    tooth_id = data_sample.metainfo.get('tooth_id', 'na')
    stem = img_path.stem.replace(' ', '_')
    return f'{index:06d}_ann{ann_id}_tooth{tooth_id}_{stem}.png'


def render_single_sample(model,
                         item: dict,
                         dataset_meta: dict,
                         output_file: str,
                         use_udp: bool,
                         mask_thr: float,
                         pred_kpt_thr: float) -> None:
    batch = pseudo_collate([item])
    with torch.no_grad():
        prediction = model.test_step(batch)[0]

    warp_mat = _build_warp_matrix(prediction, use_udp=use_udp)
    image = _ensure_bgr_uint8(item['inputs'])
    gt_mask = _mask_to_2d(prediction.gt_fields.root_mask)
    pred_fields = getattr(prediction, 'pred_fields', None)
    pred_mask = _mask_to_2d(getattr(pred_fields, 'root_mask', None))
    gt_keypoints = _extract_gt_keypoints_crop(prediction, warp_mat)
    pred_keypoints = _extract_pred_keypoints_crop(prediction, warp_mat)
    pred_scores = _extract_pred_scores(prediction)
    gt_visible = _extract_gt_visible(prediction)

    keypoint_labels = [
        dataset_meta['keypoint_id2name'][index]
        for index in range(dataset_meta['num_keypoints'])
    ]
    keypoint_colors = dataset_meta['keypoint_colors']
    skeleton_links = dataset_meta['skeleton_links']

    panel = make_visualization_panel(
        image=image,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        gt_keypoints=gt_keypoints,
        pred_keypoints=pred_keypoints,
        keypoint_labels=keypoint_labels,
        keypoint_colors=keypoint_colors,
        skeleton_links=skeleton_links,
        pred_scores=pred_scores,
        gt_visible=gt_visible,
        mask_thr=mask_thr,
        pred_kpt_thr=pred_kpt_thr)

    title = (
        f'img={Path(prediction.img_path).name}  '
        f'ann={prediction.metainfo.get("id", "na")}  '
        f'tooth={prediction.metainfo.get("tooth_id", "na")}')
    canvas = np.full((panel.shape[0] + 30, panel.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    canvas[30:, :, :] = panel

    mmengine.mkdir_or_exist(osp.dirname(output_file))
    if not cv2.imwrite(output_file, canvas):
        raise RuntimeError(f'Failed to save visualization to {output_file}')


def main():
    args = parse_args()

    register_all_modules(init_default_scope=False)
    importlib.import_module('projects.panoramic_teeth')

    cfg = Config.fromfile(args.config)
    cfg, use_udp = prepare_cfg(cfg, args.phase, args.cfg_options)

    scope = cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    dataset = build_dataset(cfg, args.phase)
    model = init_model(cfg, args.checkpoint, device=args.device)

    output_dir = Path(args.output_dir)
    mmengine.mkdir_or_exist(str(output_dir))

    max_items = len(dataset) if args.max_items is None else min(
        len(dataset), args.start_index + args.max_items)
    progress_bar = mmengine.ProgressBar(max(0, max_items - args.start_index))

    for index in range(args.start_index, max_items):
        item = dataset[index]
        output_file = output_dir / build_output_name(index, item['data_samples'])
        if args.skip_existing and output_file.exists():
            progress_bar.update()
            continue

        render_single_sample(
            model=model,
            item=item,
            dataset_meta=model.dataset_meta,
            output_file=str(output_file),
            use_udp=use_udp,
            mask_thr=args.mask_thr,
            pred_kpt_thr=args.pred_kpt_thr)
        progress_bar.update()


if __name__ == '__main__':
    main()
