from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

Point2D = Tuple[float, float]


def dedupe_consecutive_points(points: Sequence[Point2D],
                              eps: float = 1e-6) -> List[Point2D]:
    deduped: List[Point2D] = []
    for point in points:
        xy = (float(point[0]), float(point[1]))
        if not deduped:
            deduped.append(xy)
            continue
        prev = deduped[-1]
        if abs(prev[0] - xy[0]) <= eps and abs(prev[1] - xy[1]) <= eps:
            continue
        deduped.append(xy)
    return deduped


def point_distance(point_a: Point2D, point_b: Point2D) -> float:
    return ((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)**0.5


def midpoint(point_a: Point2D, point_b: Point2D) -> Point2D:
    return ((point_a[0] + point_b[0]) * 0.5, (point_a[1] + point_b[1]) * 0.5)


def flatten_points(points: Sequence[Point2D]) -> List[float]:
    flat: List[float] = []
    for x, y in points:
        flat.extend([float(x), float(y)])
    return flat


def build_closed_polygon(mesial_points: Sequence[Point2D],
                         distal_points: Sequence[Point2D]) -> List[Point2D]:
    return list(mesial_points) + list(reversed(distal_points))


def repair_polygon(points: Sequence[Point2D]) -> Tuple[Optional[Polygon], Optional[str]]:
    if len(points) < 4:
        return None, 'invalid_polygon'

    polygon = Polygon(points)
    if polygon.is_empty:
        return None, 'invalid_polygon'
    if not polygon.is_valid:
        polygon = make_valid(polygon)
    if polygon.is_empty:
        return None, 'invalid_polygon'
    if polygon.geom_type == 'MultiPolygon':
        return None, 'ambiguous_repaired_polygon'
    if polygon.geom_type != 'Polygon':
        return None, 'invalid_polygon'
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None, 'invalid_polygon'
    if polygon.geom_type == 'MultiPolygon':
        return None, 'ambiguous_repaired_polygon'
    if polygon.geom_type != 'Polygon':
        return None, 'invalid_polygon'
    return polygon, None


def snap_point_to_boundary(point: Point2D, polygon: Polygon) -> Point2D:
    boundary = polygon.boundary
    distance = boundary.project(Point(point))
    snapped = boundary.interpolate(distance)
    return (float(snapped.x), float(snapped.y))


def ensure_point_not_outside(point: Point2D,
                             polygon: Polygon,
                             tol: float = 1e-3) -> Tuple[Point2D, bool]:
    point_geom = Point(point)
    if polygon.buffer(tol).covers(point_geom):
        return point, False
    return snap_point_to_boundary(point, polygon), True


def build_keypoint_triplets(keypoints: Sequence[Point2D]) -> List[float]:
    flat: List[float] = []
    for x, y in keypoints:
        flat.extend([float(x), float(y), 2.0])
    return flat


def polygon_bounds(points: Sequence[Point2D]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def build_expanded_bbox(
    polygon_points: Sequence[Point2D],
    image_width: int,
    image_height: int,
    pad_left_ratio: float = 0.10,
    pad_right_ratio: float = 0.10,
    pad_top_ratio: float = 0.12,
    pad_bottom_ratio: float = 0.08,
) -> List[float]:
    min_x, min_y, max_x, max_y = polygon_bounds(polygon_points)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    x1 = max(0.0, min_x - width * pad_left_ratio)
    y1 = max(0.0, min_y - height * pad_top_ratio)
    x2 = min(float(image_width - 1), max_x + width * pad_right_ratio)
    y2 = min(float(image_height - 1), max_y + height * pad_bottom_ratio)
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def build_tooth_instance(
    tooth_id: int,
    mesial_points: Sequence[Point2D],
    distal_points: Sequence[Point2D],
    image_id: int,
    annotation_id: int,
    image_width: int,
    image_height: int,
    apex_merge_threshold: float = 4.0,
    min_area: float = 16.0,
    min_side_points: int = 4,
) -> Tuple[Optional[dict], Dict[str, object]]:
    debug: Dict[str, object] = {
        'tooth_id': tooth_id,
        'snapped_keypoints': 0,
        'skip_reason': None,
    }

    m_points = dedupe_consecutive_points(mesial_points)
    d_points = dedupe_consecutive_points(distal_points)

    if len(m_points) < 2 or len(d_points) < 2:
        debug['skip_reason'] = 'too_few_points'
        return None, debug
    if len(m_points) < min_side_points or len(d_points) < min_side_points:
        debug['skip_reason'] = 'incomplete_side_contour'
        return None, debug

    raw_polygon_points = build_closed_polygon(m_points, d_points)
    polygon, repair_reason = repair_polygon(raw_polygon_points)
    if polygon is None:
        debug['skip_reason'] = repair_reason or 'invalid_polygon'
        return None, debug
    if polygon.area < min_area:
        debug['skip_reason'] = 'polygon_area_too_small'
        return None, debug

    segmentation_points = [(float(x), float(y))
                           for x, y in polygon.exterior.coords[:-1]]
    if len(segmentation_points) < 4:
        debug['skip_reason'] = 'polygon_vertices_too_few'
        return None, debug

    m_c = m_points[0]
    m_b = m_points[1]
    d_b = d_points[1]
    d_c = d_points[0]

    m_apex = m_points[-1]
    d_apex = d_points[-1]
    apex_distance = point_distance(m_apex, d_apex)
    a_point = midpoint(m_apex, d_apex)
    debug['apex_distance'] = apex_distance
    debug['apex_merged'] = apex_distance < apex_merge_threshold

    keypoints = [m_c, m_b, a_point, d_b, d_c]
    snapped_keypoints: List[Point2D] = []
    for point in keypoints:
        snapped_point, snapped = ensure_point_not_outside(point, polygon)
        if snapped:
            debug['snapped_keypoints'] = int(debug['snapped_keypoints']) + 1
        snapped_keypoints.append(snapped_point)

    bbox = build_expanded_bbox(segmentation_points, image_width, image_height)
    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': 1,
        'bbox': bbox,
        'area': float(polygon.area),
        'iscrowd': 0,
        'num_keypoints': 5,
        'keypoints': build_keypoint_triplets(snapped_keypoints),
        'segmentation': [flatten_points(segmentation_points)],
        'tooth_id': int(tooth_id),
        'side_contours': {
            'M': [[float(x), float(y)] for x, y in m_points],
            'D': [[float(x), float(y)] for x, y in d_points],
        },
        'apex_midpoint': [float(a_point[0]), float(a_point[1])],
    }
    return annotation, debug
