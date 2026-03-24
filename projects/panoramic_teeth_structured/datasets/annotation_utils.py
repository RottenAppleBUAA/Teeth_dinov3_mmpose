from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

PointArray = np.ndarray


def flip_points(points: PointArray, image_shape,
                direction: str = 'horizontal') -> PointArray:
    """Flip 2D points in image space."""
    flipped = np.array(points, dtype=np.float32).reshape(-1, 2).copy()
    height, width = image_shape[:2]
    if direction == 'horizontal':
        flipped[:, 0] = width - 1 - flipped[:, 0]
    elif direction == 'vertical':
        flipped[:, 1] = height - 1 - flipped[:, 1]
    elif direction == 'diagonal':
        flipped[:, 0] = width - 1 - flipped[:, 0]
        flipped[:, 1] = height - 1 - flipped[:, 1]
    else:
        raise ValueError(f'Unsupported flip direction: {direction}')
    return flipped


def transform_points(points: PointArray, warp_mat: np.ndarray) -> PointArray:
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return points
    return cv2.transform(points[None, ...], warp_mat)[0]


def _cumulative_lengths(points: PointArray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros((len(points), ), dtype=np.float32)
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    return np.concatenate(
        [np.zeros((1, ), dtype=np.float32), np.cumsum(seg_lengths)])


def sample_polyline(points: PointArray, num_samples: int) -> PointArray:
    """Sample a polyline by arc length, including both endpoints."""
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((num_samples, 2), dtype=np.float32)
    if len(points) == 1 or num_samples <= 1:
        return np.repeat(points[:1], num_samples, axis=0)

    cumlen = _cumulative_lengths(points)
    total_length = float(cumlen[-1])
    if total_length <= 1e-6:
        return np.repeat(points[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total_length, num_samples, dtype=np.float32)
    sampled = np.zeros((num_samples, 2), dtype=np.float32)
    for index, target in enumerate(targets):
        right = int(np.searchsorted(cumlen, target, side='right'))
        right = min(max(right, 1), len(points) - 1)
        left = right - 1
        left_len = float(cumlen[left])
        right_len = float(cumlen[right])
        if right_len - left_len <= 1e-6:
            sampled[index] = points[right]
            continue
        ratio = (target - left_len) / (right_len - left_len)
        sampled[index] = points[left] * (1.0 - ratio) + points[right] * ratio
    return sampled


def resample_semantic_side(points: PointArray, num_points: int) -> PointArray:
    """Resample a side contour while preserving semantic anchors.

    The first point is the coronal point, the second point is the body point,
    and the last point is the side-specific apex endpoint.
    """
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError('A semantic side contour requires at least 2 points.')
    if num_points < 3:
        raise ValueError('num_points must be at least 3.')

    if len(points) == 2:
        tail = np.repeat(points[1:2], num_points - 1, axis=0)
    else:
        tail = sample_polyline(points[1:], num_points - 1)
        tail[0] = points[1]
        tail[-1] = points[-1]
    return np.concatenate([points[:1], tail], axis=0)


def rasterize_polygon(points: PointArray,
                      width: int,
                      height: int) -> np.ndarray:
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.float32)
    if len(points) < 3:
        return mask
    contour = np.round(points).astype(np.int32)
    cv2.fillPoly(mask, [contour], 1.0)
    return mask


def rasterize_polyline(points: PointArray,
                       width: int,
                       height: int,
                       thickness: int = 2) -> np.ndarray:
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.float32)
    if len(points) < 2:
        return mask
    contour = np.round(points).astype(np.int32)
    cv2.polylines(mask, [contour], isClosed=False, color=1.0, thickness=thickness)
    return np.clip(mask, 0.0, 1.0)


def distance_transform_from_binary(binary: np.ndarray,
                                   normalize_by: float) -> np.ndarray:
    binary = np.asarray(binary, dtype=np.float32)
    if binary.ndim != 2:
        raise ValueError(f'Expected a 2D binary map, but got {binary.shape!r}.')
    source = np.where(binary > 0.5, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 3).astype(np.float32)
    scale = max(float(normalize_by), 1.0)
    return np.clip(distance / scale, 0.0, 1.0)


def derive_keypoints_from_contours(
        mesial_contour: PointArray,
        distal_contour: PointArray) -> Tuple[np.ndarray, np.ndarray]:
    mesial_contour = np.array(mesial_contour, dtype=np.float32).reshape(-1, 2)
    distal_contour = np.array(distal_contour, dtype=np.float32).reshape(-1, 2)
    apex = 0.5 * (mesial_contour[-1] + distal_contour[-1])
    keypoints = np.stack([
        mesial_contour[0],
        mesial_contour[1],
        apex,
        distal_contour[1],
        distal_contour[0],
    ],
                         axis=0).astype(np.float32)
    return keypoints, apex.astype(np.float32)

