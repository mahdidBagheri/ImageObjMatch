#!/usr/bin/env python3
"""Match annotated objects between the top and bottom halves of an image.

The LabelMe annotations in ``dastchin`` and ``goo`` outline the same physical
objects with different labels.  Labels are deliberately ignored and duplicate
outlines are merged before matching.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import os
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class Observation:
    source: str
    shape_index: int
    center_x: float
    center_y: float
    area: float
    bbox: tuple[float, float, float, float]


@dataclass
class Object:
    id: int
    center_x: float
    center_y: float
    area: float
    bbox: tuple[float, float, float, float]
    observations: list[Observation]


@dataclass(frozen=True)
class LinearTransform:
    """Affine camera mapping: bottom = matrix * top + translation."""

    xx: float = 1.0
    xy: float = 0.0
    x_offset: float = 0.0
    yx: float = 0.0
    yy: float = 1.0
    y_offset: float = 0.0
    inlier_count: int = 0
    sample_count: int = 0
    median_error: float = 0.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.xx * x + self.xy * y + self.x_offset,
            self.yx * x + self.yy * y + self.y_offset,
        )


def polygon_measurements(points: Sequence[Sequence[float]]) -> tuple[
    float, float, float, tuple[float, float, float, float]
]:
    """Return a stable centroid, absolute area and bounding box."""
    if not points:
        raise ValueError("annotation shape has no points")
    pts = [(float(point[0]), float(point[1])) for point in points]
    xs, ys = zip(*pts)
    bbox = (min(xs), min(ys), max(xs), max(ys))

    cross_sum = cx_sum = cy_sum = 0.0
    if len(pts) >= 3:
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            cross = x1 * y2 - x2 * y1
            cross_sum += cross
            cx_sum += (x1 + x2) * cross
            cy_sum += (y1 + y2) * cross
    if abs(cross_sum) > 1e-9:
        return (
            cx_sum / (3.0 * cross_sum),
            cy_sum / (3.0 * cross_sum),
            abs(cross_sum) / 2.0,
            bbox,
        )

    # Handles points, lines and degenerate polygons.
    return sum(xs) / len(xs), sum(ys) / len(ys), 0.0, bbox


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def read_labelme(path: Path) -> tuple[dict, list[Observation]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    observations: list[Observation] = []
    for index, shape in enumerate(data.get("shapes", [])):
        points = shape.get("points", [])
        if not points:
            continue
        cx, cy, area, bbox = polygon_measurements(points)
        observations.append(Observation(path.parent.name, index, cx, cy, area, bbox))
    return data, observations


def _object_from_members(object_id: int, members: list[Observation]) -> Object:
    count = len(members)
    return Object(
        id=object_id,
        center_x=sum(item.center_x for item in members) / count,
        center_y=sum(item.center_y for item in members) / count,
        area=sum(item.area for item in members) / count,
        bbox=tuple(sum(item.bbox[k] for item in members) / count for k in range(4)),
        observations=members,
    )


def merge_annotations(annotation_paths: Sequence[Path], iou_threshold: float = 0.5) -> tuple[dict, list[Object]]:
    """Merge duplicate shapes across annotation files; never within one file."""
    metadata: dict | None = None
    clusters: list[list[Observation]] = []
    for source_number, path in enumerate(annotation_paths):
        current_metadata, observations = read_labelme(path)
        metadata = metadata or current_metadata
        if source_number == 0:
            clusters = [[item] for item in observations]
            continue

        # Global greedy matching by IoU works well here because alternate
        # annotations are near-identical, and prevents a cluster being reused.
        candidates: list[tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for cluster_index, cluster in enumerate(clusters):
                if any(member.source == observation.source for member in cluster):
                    continue
                score = max(bbox_iou(observation.bbox, member.bbox) for member in cluster)
                if score >= iou_threshold:
                    candidates.append((score, observation_index, cluster_index))
        used_observations: set[int] = set()
        used_clusters: set[int] = set()
        for _, observation_index, cluster_index in sorted(candidates, reverse=True):
            if observation_index in used_observations or cluster_index in used_clusters:
                continue
            clusters[cluster_index].append(observations[observation_index])
            used_observations.add(observation_index)
            used_clusters.add(cluster_index)
        clusters.extend([item] for i, item in enumerate(observations) if i not in used_observations)

    if metadata is None:
        raise ValueError("no annotation files were supplied")
    objects = [_object_from_members(index, members) for index, members in enumerate(clusters)]
    return metadata, objects


def bmp_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(26)
    if len(header) < 26 or header[:2] != b"BM":
        raise ValueError(f"not a supported BMP file: {path}")
    width, height = struct.unpack_from("<ii", header, 18)
    return abs(width), abs(height)


def _solve_3x3(matrix: Sequence[Sequence[float]], values: Sequence[float]) -> tuple[float, float, float] | None:
    augmented = [list(row) + [float(value)] for row, value in zip(matrix, values)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-8:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[row][3] for row in range(3))


def _least_squares_affine(pairs: Sequence[tuple[Object, Object]]) -> LinearTransform | None:
    if len(pairs) < 3:
        return None
    normal = [[0.0] * 3 for _ in range(3)]
    target_x = [0.0] * 3
    target_y = [0.0] * 3
    for top, bottom in pairs:
        row = (top.center_x, top.center_y, 1.0)
        for i in range(3):
            target_x[i] += row[i] * bottom.center_x
            target_y[i] += row[i] * bottom.center_y
            for j in range(3):
                normal[i][j] += row[i] * row[j]
    x_parameters = _solve_3x3(normal, target_x)
    y_parameters = _solve_3x3(normal, target_y)
    if x_parameters is None or y_parameters is None:
        return None
    return LinearTransform(*x_parameters, *y_parameters)


def transform_error(transform: LinearTransform, pair: tuple[Object, Object]) -> float:
    top, bottom = pair
    predicted_x, predicted_y = transform.apply(top.center_x, top.center_y)
    return math.hypot(predicted_x - bottom.center_x, predicted_y - bottom.center_y)


def _axis_aligned_transform(pairs: Sequence[tuple[Object, Object]]) -> LinearTransform:
    def fit_axis(source: list[float], target: list[float]) -> tuple[float, float]:
        source_mean = sum(source) / len(source)
        target_mean = sum(target) / len(target)
        variance = sum((value - source_mean) ** 2 for value in source)
        if variance < 1e-8:
            return 1.0, target_mean - source_mean
        scale = sum(
            (source_value - source_mean) * (target_value - target_mean)
            for source_value, target_value in zip(source, target)
        ) / variance
        return scale, target_mean - scale * source_mean

    x_scale, x_offset = fit_axis(
        [top.center_x for top, _ in pairs], [bottom.center_x for _, bottom in pairs]
    )
    y_scale, y_offset = fit_axis(
        [top.center_y for top, _ in pairs], [bottom.center_y for _, bottom in pairs]
    )
    provisional = LinearTransform(
        xx=x_scale,
        x_offset=x_offset,
        yy=y_scale,
        y_offset=y_offset,
    )
    errors = sorted(transform_error(provisional, pair) for pair in pairs)
    return LinearTransform(
        xx=x_scale,
        x_offset=x_offset,
        yy=y_scale,
        y_offset=y_offset,
        inlier_count=len(pairs),
        sample_count=len(pairs),
        median_error=errors[len(errors) // 2],
    )


def fit_linear_transform(
    pairs: Sequence[tuple[Object, Object]], width: float, seed: int = 0
) -> LinearTransform:
    """Fit an affine mapping with deterministic RANSAC outlier rejection."""
    pairs = list(pairs)
    if not pairs:
        return LinearTransform()
    if len(pairs) < 6:
        return _axis_aligned_transform(pairs)

    threshold = max(15.0, width * 0.018)
    combination_count = math.comb(len(pairs), 3)
    if combination_count <= 2500:
        samples: Iterable[tuple[int, int, int]] = itertools.combinations(range(len(pairs)), 3)
    else:
        generator = random.Random(seed)
        unique_samples: set[tuple[int, int, int]] = set()
        while len(unique_samples) < 2500:
            unique_samples.add(tuple(sorted(generator.sample(range(len(pairs)), 3))))
        samples = sorted(unique_samples)

    best_transform: LinearTransform | None = None
    best_inliers: list[tuple[Object, Object]] = []
    best_score = (0, float("-inf"))
    for indices in samples:
        candidate = _least_squares_affine([pairs[index] for index in indices])
        if candidate is None:
            continue
        errors = [transform_error(candidate, pair) for pair in pairs]
        inliers = [pair for pair, error in zip(pairs, errors) if error <= threshold]
        # First maximize consensus, then favor the tighter model.
        score = (len(inliers), -sum(min(error, threshold) for error in errors))
        if score > best_score:
            best_score = score
            best_transform = candidate
            best_inliers = inliers

    if best_transform is None or len(best_inliers) < 3:
        # Degenerate geometry: translation is safer than an unstable affine fit.
        x_shifts = sorted(bottom.center_x - top.center_x for top, bottom in pairs)
        y_shifts = sorted(bottom.center_y - top.center_y for top, bottom in pairs)
        middle = len(pairs) // 2
        return LinearTransform(
            x_offset=x_shifts[middle],
            y_offset=y_shifts[middle],
            inlier_count=len(pairs),
            sample_count=len(pairs),
        )

    # Refit on consensus, then update consensus twice around the refined model.
    for _ in range(2):
        refined = _least_squares_affine(best_inliers)
        if refined is None:
            break
        best_transform = refined
        updated = [pair for pair in pairs if transform_error(best_transform, pair) <= threshold]
        if len(updated) < 3 or len(updated) == len(best_inliers):
            best_inliers = updated or best_inliers
            break
        best_inliers = updated
    errors = sorted(transform_error(best_transform, pair) for pair in best_inliers)
    median_error = errors[len(errors) // 2] if errors else 0.0
    return LinearTransform(
        best_transform.xx,
        best_transform.xy,
        best_transform.x_offset,
        best_transform.yx,
        best_transform.yy,
        best_transform.y_offset,
        len(best_inliers),
        len(pairs),
        median_error,
    )


def pair_cost(
    top: Object,
    bottom: Object,
    width: float,
    transform: LinearTransform | None = None,
) -> float:
    # With no calibration this is direct line length. With calibration it is
    # residual distance from the location predicted in the bottom image.
    predicted_x, predicted_y = (transform or LinearTransform()).apply(top.center_x, top.center_y)
    return math.hypot(predicted_x - bottom.center_x, predicted_y - bottom.center_y) / max(width, 1.0)


def monotone_match(
    top: Sequence[Object],
    bottom: Sequence[Object],
    width: float,
    transform: LinearTransform | None = None,
) -> list[tuple[Object, Object]]:
    """Return a maximum-cardinality, minimum-cost bipartite match.

    This is the rectangular Hungarian algorithm. The historical function name
    is retained for API compatibility. With a supplied camera transform, cost
    is prediction error rather than raw line length.
    """
    if not top or not bottom:
        return []

    swapped = len(top) > len(bottom)
    rows, columns = (bottom, top) if swapped else (top, bottom)
    row_count, column_count = len(rows), len(columns)
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    column_match = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        column_match[0] = row
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        current_column = 0
        while True:
            used[current_column] = True
            current_row = column_match[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    pair_cost(
                        columns[column - 1] if swapped else rows[current_row - 1],
                        rows[current_row - 1] if swapped else columns[column - 1],
                        width,
                        transform,
                    )
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum[column]:
                    minimum[column] = reduced_cost
                    previous_column[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break
        while True:
            next_column = previous_column[current_column]
            column_match[current_column] = column_match[next_column]
            current_column = next_column
            if current_column == 0:
                break

    pairs: list[tuple[Object, Object]] = []
    for column in range(1, column_count + 1):
        if not column_match[column]:
            continue
        first, second = rows[column_match[column] - 1], columns[column - 1]
        pairs.append((second, first) if swapped else (first, second))
    return sorted(pairs, key=lambda pair: (pair[0].center_x, pair[0].center_y))


def count_intersections(pairs: Sequence[tuple[Object, Object]]) -> int:
    def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(first: tuple[tuple[float, float], tuple[float, float]], second: tuple[tuple[float, float], tuple[float, float]]) -> bool:
        a, b = first
        c, d = second
        return orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0

    intersections = 0
    for index, (top_a, bottom_a) in enumerate(pairs):
        for top_b, bottom_b in pairs[index + 1 :]:
            first = ((top_a.center_x, top_a.center_y), (bottom_a.center_x, bottom_a.center_y))
            second = ((top_b.center_x, top_b.center_y), (bottom_b.center_x, bottom_b.center_y))
            if intersects(first, second):
                intersections += 1
    return intersections


def transform_guided_match(
    top: Sequence[Object],
    bottom: Sequence[Object],
    width: float,
    transform: LinearTransform | None,
) -> list[tuple[Object, Object]]:
    """Improve transform agreement while preserving minimum intersections.

    Raw Euclidean assignment supplies a zero-crossing baseline. Local pair
    swaps and matched/unmatched replacements are accepted only when they keep
    that intersection count and lower total calibrated prediction error.
    """
    pairs = monotone_match(top, bottom, width)
    if transform is None or not pairs:
        return pairs
    minimum_intersections = count_intersections(pairs)

    def objective(candidate: Sequence[tuple[Object, Object]]) -> float:
        return sum(pair_cost(a, b, width, transform) for a, b in candidate)

    for _ in range(25):
        current_cost = objective(pairs)
        best_cost = current_cost
        best_pairs: list[tuple[Object, Object]] | None = None

        # Reassign two existing pairs.
        for first in range(len(pairs)):
            for second in range(first + 1, len(pairs)):
                candidate = list(pairs)
                top_a, bottom_a = candidate[first]
                top_b, bottom_b = candidate[second]
                candidate[first] = (top_a, bottom_b)
                candidate[second] = (top_b, bottom_a)
                if count_intersections(candidate) != minimum_intersections:
                    continue
                candidate_cost = objective(candidate)
                if candidate_cost < best_cost - 1e-9:
                    best_cost, best_pairs = candidate_cost, candidate

        # Permit calibration to choose a different unmatched object too.
        matched_top = {item.id for item, _ in pairs}
        matched_bottom = {item.id for _, item in pairs}
        unused_top = [item for item in top if item.id not in matched_top]
        unused_bottom = [item for item in bottom if item.id not in matched_bottom]
        for pair_index, (_, paired_bottom) in enumerate(pairs):
            for replacement_top in unused_top:
                candidate = list(pairs)
                candidate[pair_index] = (replacement_top, paired_bottom)
                if count_intersections(candidate) == minimum_intersections:
                    candidate_cost = objective(candidate)
                    if candidate_cost < best_cost - 1e-9:
                        best_cost, best_pairs = candidate_cost, candidate
        for pair_index, (paired_top, _) in enumerate(pairs):
            for replacement_bottom in unused_bottom:
                candidate = list(pairs)
                candidate[pair_index] = (paired_top, replacement_bottom)
                if count_intersections(candidate) == minimum_intersections:
                    candidate_cost = objective(candidate)
                    if candidate_cost < best_cost - 1e-9:
                        best_cost, best_pairs = candidate_cost, candidate

        if best_pairs is None:
            break
        pairs = best_pairs
    return sorted(pairs, key=lambda pair: (pair[0].center_x, pair[0].center_y))


def object_json(item: Object) -> dict:
    result = asdict(item)
    result["bbox"] = list(item.bbox)
    return result


MATCH_COLORS = [
    (255, 59, 48),
    (52, 199, 89),
    (0, 122, 255),
    (255, 149, 0),
    (175, 82, 222),
    (0, 199, 190),
    (255, 45, 85),
]


def svg_visualization(image_href: str, width: int, height: int, top: Sequence[Object], bottom: Sequence[Object], pairs: Sequence[tuple[Object, Object]]) -> str:
    matched_ids = {item.id for pair in pairs for item in pair}
    colors = ["#%02x%02x%02x" % color for color in MATCH_COLORS]
    relative_image = html.escape(image_href, quote=True)
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <image href="{relative_image}" width="{width}" height="{height}"/>',
        f'  <line x1="0" y1="{height / 2:.2f}" x2="{width}" y2="{height / 2:.2f}" stroke="white" stroke-opacity=".65" stroke-dasharray="10 8"/>',
    ]
    for match_id, (top_item, bottom_item) in enumerate(pairs):
        color = colors[match_id % len(colors)]
        lines.append(f'  <line x1="{top_item.center_x:.2f}" y1="{top_item.center_y:.2f}" x2="{bottom_item.center_x:.2f}" y2="{bottom_item.center_y:.2f}" stroke="{color}" stroke-width="4" stroke-opacity=".85"/>')
        for item in (top_item, bottom_item):
            lines.append(f'  <circle cx="{item.center_x:.2f}" cy="{item.center_y:.2f}" r="13" fill="{color}" stroke="white" stroke-width="3"/>')
            lines.append(f'  <text x="{item.center_x:.2f}" y="{item.center_y + 5:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{match_id}</text>')
    for item in [*top, *bottom]:
        if item.id not in matched_ids:
            lines.append(f'  <circle cx="{item.center_x:.2f}" cy="{item.center_y:.2f}" r="14" fill="none" stroke="white" stroke-width="4"/>')
            lines.append(f'  <path d="M {item.center_x-8:.2f} {item.center_y-8:.2f} L {item.center_x+8:.2f} {item.center_y+8:.2f} M {item.center_x+8:.2f} {item.center_y-8:.2f} L {item.center_x-8:.2f} {item.center_y+8:.2f}" stroke="white" stroke-width="4"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_bmp(
    source: Path,
    destination: Path,
    top: Sequence[Object],
    bottom: Sequence[Object],
    pairs: Sequence[tuple[Object, Object]],
) -> None:
    """Draw match lines into an uncompressed 24-bit BMP without dependencies."""
    bitmap = bytearray(source.read_bytes())
    pixel_offset = struct.unpack_from("<I", bitmap, 10)[0]
    width, signed_height = struct.unpack_from("<ii", bitmap, 18)
    bits_per_pixel = struct.unpack_from("<H", bitmap, 28)[0]
    compression = struct.unpack_from("<I", bitmap, 30)[0]
    if bits_per_pixel != 24 or compression != 0:
        raise ValueError(f"visualization requires an uncompressed 24-bit BMP: {source}")
    height = abs(signed_height)
    row_stride = (width * 3 + 3) & ~3

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if not (0 <= x < width and 0 <= y < height):
            return
        storage_y = height - 1 - y if signed_height > 0 else y
        offset = pixel_offset + storage_y * row_stride + x * 3
        red, green, blue = color
        bitmap[offset : offset + 3] = bytes((blue, green, red))

    def disk(center_x: int, center_y: int, radius: int, color: tuple[int, int, int]) -> None:
        radius_squared = radius * radius
        for delta_y in range(-radius, radius + 1):
            extent = int(math.sqrt(max(0, radius_squared - delta_y * delta_y)))
            for delta_x in range(-extent, extent + 1):
                set_pixel(center_x + delta_x, center_y + delta_y, color)

    def line(x1: float, y1: float, x2: float, y2: float, color: tuple[int, int, int], thickness: int = 5) -> None:
        start_x, start_y = int(round(x1)), int(round(y1))
        end_x, end_y = int(round(x2)), int(round(y2))
        delta_x, delta_y = end_x - start_x, end_y - start_y
        steps = max(abs(delta_x), abs(delta_y), 1)
        radius = max(1, thickness // 2)
        for step in range(steps + 1):
            fraction = step / steps
            disk(
                int(round(start_x + delta_x * fraction)),
                int(round(start_y + delta_y * fraction)),
                radius,
                color,
            )

    matched_ids = {item.id for pair in pairs for item in pair}
    for match_index, (top_item, bottom_item) in enumerate(pairs):
        color = MATCH_COLORS[match_index % len(MATCH_COLORS)]
        line(top_item.center_x, top_item.center_y, bottom_item.center_x, bottom_item.center_y, color)
        disk(round(top_item.center_x), round(top_item.center_y), 9, color)
        disk(round(bottom_item.center_x), round(bottom_item.center_y), 9, color)

    # White X markers show detections that could not be paired.
    for item in [*top, *bottom]:
        if item.id in matched_ids:
            continue
        line(item.center_x - 10, item.center_y - 10, item.center_x + 10, item.center_y + 10, (255, 255, 255), 4)
        line(item.center_x + 10, item.center_y - 10, item.center_x - 10, item.center_y + 10, (255, 255, 255), 4)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(bitmap)
    os.replace(temporary, destination)


def save_matched_crops(
    source: Path,
    destination_dir: Path,
    pairs: Sequence[tuple[Object, Object]],
    padding: int = 12,
) -> list[Path]:
    """Save each top/bottom object pair as one side-by-side BMP image."""
    bitmap = source.read_bytes()
    pixel_offset = struct.unpack_from("<I", bitmap, 10)[0]
    source_width, signed_height = struct.unpack_from("<ii", bitmap, 18)
    bits_per_pixel = struct.unpack_from("<H", bitmap, 28)[0]
    compression = struct.unpack_from("<I", bitmap, 30)[0]
    if bits_per_pixel != 24 or compression != 0:
        raise ValueError(f"crop export requires an uncompressed 24-bit BMP: {source}")
    source_height = abs(signed_height)
    source_stride = (source_width * 3 + 3) & ~3

    def source_pixel(x: int, y: int) -> bytes:
        if not (0 <= x < source_width and 0 <= y < source_height):
            return b"\x00\x00\x00"
        storage_y = source_height - 1 - y if signed_height > 0 else y
        offset = pixel_offset + storage_y * source_stride + x * 3
        return bitmap[offset : offset + 3]

    def extract(item: Object, crop_width: int, crop_height: int) -> list[bytearray]:
        left = int(round(item.center_x - crop_width / 2))
        top = int(round(item.center_y - crop_height / 2))
        return [
            bytearray().join(source_pixel(left + x, top + y) for x in range(crop_width))
            for y in range(crop_height)
        ]

    def write_bmp(path: Path, rows: Sequence[bytearray], width: int, height: int) -> None:
        row_stride = (width * 3 + 3) & ~3
        padding_bytes = b"\x00" * (row_stride - width * 3)
        pixel_data = b"".join(bytes(row) + padding_bytes for row in reversed(rows))
        file_header = struct.pack("<2sIHHI", b"BM", 54 + len(pixel_data), 0, 0, 54)
        dib_header = struct.pack(
            "<IiiHHIIiiII",
            40,
            width,
            height,
            1,
            24,
            0,
            len(pixel_data),
            2835,
            2835,
            0,
            0,
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(file_header + dib_header + pixel_data)
        os.replace(temporary, path)

    destination_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    divider_width = 4
    for match_index, (top_item, bottom_item) in enumerate(pairs):
        object_width = max(
            top_item.bbox[2] - top_item.bbox[0],
            bottom_item.bbox[2] - bottom_item.bbox[0],
        )
        object_height = max(
            top_item.bbox[3] - top_item.bbox[1],
            bottom_item.bbox[3] - bottom_item.bbox[1],
        )
        crop_width = max(1, int(math.ceil(object_width)) + 2 * padding)
        crop_height = max(1, int(math.ceil(object_height)) + 2 * padding)
        top_crop = extract(top_item, crop_width, crop_height)
        bottom_crop = extract(bottom_item, crop_width, crop_height)
        # BMP stores BGR bytes. The divider is white in either ordering.
        divider = b"\xff\xff\xff" * divider_width
        rows = [
            bytearray(bytes(top_row) + divider + bytes(bottom_row))
            for top_row, bottom_row in zip(top_crop, bottom_crop)
        ]
        destination = destination_dir / f"{source.stem}__match_{match_index:03d}.bmp"
        write_bmp(destination, rows, crop_width * 2 + divider_width, crop_height)
        written.append(destination)
    return written


def annotation_folders(dataset: Path) -> list[Path]:
    """Return annotation folders for a data root in deterministic order.

    LabelMe JSON files live in sibling directories whose names start with an
    underscore. The original ``dastchin``/``goo`` layout remains supported as
    a fallback so existing datasets continue to work.
    """
    folders = sorted(
        (
            child
            for child in dataset.iterdir()
            if child.is_dir() and child.name.startswith("_")
        ),
        key=lambda child: child.name.casefold(),
    )
    if folders:
        return folders
    return [
        child
        for child in (dataset / "dastchin", dataset / "goo")
        if child.is_dir()
    ]


def load_image_objects(
    dataset: Path, image_path: Path, iou_threshold: float
) -> tuple[dict, list[Path], int, int, list[Object], list[Object]]:
    annotation_paths = [
        folder / f"{image_path.stem}.json"
        for folder in annotation_folders(dataset)
    ]
    annotation_paths = [path for path in annotation_paths if path.exists()]
    if not annotation_paths:
        raise FileNotFoundError(
            f"no annotations found for {image_path.name} in label folders "
            f"under {dataset}"
        )
    metadata, objects = merge_annotations(annotation_paths, iou_threshold)
    bmp_width, bmp_height = bmp_dimensions(image_path)
    width = int(metadata.get("imageWidth") or bmp_width)
    height = int(metadata.get("imageHeight") or bmp_height)
    split_y = height / 2.0
    top = [item for item in objects if item.center_y < split_y]
    bottom = [item for item in objects if item.center_y >= split_y]
    return metadata, annotation_paths, width, height, top, bottom


def calibration_key(image_path: Path, width: int, height: int) -> str:
    parts = image_path.stem.split("  ", 1)
    session = parts[1] if len(parts) == 2 else "default"
    return f"{width}x{height} {session}"


def calibrate_transforms(
    dataset: Path,
    paths: Sequence[Path],
    iou_threshold: float,
) -> dict[str, LinearTransform]:
    loaded: dict[Path, tuple[int, int, list[Object], list[Object]]] = {}
    for image_path in paths:
        _, _, width, height, top, bottom = load_image_objects(dataset, image_path, iou_threshold)
        loaded[image_path] = (width, height, top, bottom)

    transforms: dict[str, LinearTransform] = {}
    # Start from geometry-only matches, then alternate assignment/calibration.
    for _ in range(3):
        samples: dict[str, list[tuple[Object, Object]]] = {}
        widths: dict[str, int] = {}
        for image_path, (width, height, top, bottom) in loaded.items():
            key = calibration_key(image_path, width, height)
            pairs = transform_guided_match(top, bottom, width, transforms.get(key))
            samples.setdefault(key, []).extend(pairs)
            widths[key] = width
        transforms = {
            key: fit_linear_transform(pairs, widths[key], seed=sum(map(ord, key)))
            for key, pairs in samples.items()
        }
    return transforms


def match_image(
    dataset: Path,
    image_path: Path,
    output_dir: Path,
    visual_output_dir: Path,
    crop_output_dir: Path,
    iou_threshold: float,
    transform: LinearTransform | None = None,
) -> dict:
    _, annotation_paths, width, height, top, bottom = load_image_objects(dataset, image_path, iou_threshold)
    split_y = height / 2.0
    transform = transform or LinearTransform()
    pairs = transform_guided_match(top, bottom, width, transform)
    paired_ids = {item.id for pair in pairs for item in pair}

    result = {
        "image": image_path.name,
        "width": width,
        "height": height,
        "split_y": split_y,
        "annotation_sources": [str(path) for path in annotation_paths],
        "calibration_key": calibration_key(image_path, width, height),
        "linear_transform": asdict(transform),
        "top_count": len(top),
        "bottom_count": len(bottom),
        "intersection_count": count_intersections(pairs),
        "matches": [
            {
                "match_id": index,
                "transformation_error_pixels": pair_cost(top_item, bottom_item, width, transform) * width,
                "predicted_bottom_center": list(transform.apply(top_item.center_x, top_item.center_y)),
                "top": object_json(top_item),
                "bottom": object_json(bottom_item),
            }
            for index, (top_item, bottom_item) in enumerate(pairs)
        ],
        "unmatched_top": [object_json(item) for item in top if item.id not in paired_ids],
        "unmatched_bottom": [object_json(item) for item in bottom if item.id not in paired_ids],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{image_path.stem}.matches.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    image_href = Path(os.path.relpath(image_path, output_dir)).as_posix()
    (output_dir / f"{image_path.stem}.matches.svg").write_text(
        svg_visualization(image_href, width, height, top, bottom, pairs), encoding="utf-8"
    )
    render_bmp(image_path, visual_output_dir / image_path.name, top, bottom, pairs)
    save_matched_crops(image_path, crop_output_dir, pairs)
    return result


def image_files(dataset: Path, requested_image: str | None) -> Iterable[Path]:
    image_dir = dataset / "images"
    if requested_image:
        requested = Path(requested_image)
        if requested.exists():
            yield requested
            return
        candidate = image_dir / requested_image
        if not candidate.suffix:
            candidate = candidate.with_suffix(".bmp")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        yield candidate
        return
    yield from sorted(image_dir.glob("*.bmp"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="image filename/path; omit to process the complete dataset")
    parser.add_argument("--dataset", type=Path, default=Path("images_dastchin_goo"))
    parser.add_argument("--output", type=Path, default=None, help="directory for JSON and SVG results")
    parser.add_argument("--visual-output", type=Path, default=Path("output"), help="directory for rendered BMP images")
    parser.add_argument("--crop-output", type=Path, default=Path("crops"), help="directory for side-by-side matched crops")
    parser.add_argument("--merge-iou", type=float, default=0.5, help="IoU used to merge duplicate cross-folder outlines")
    args = parser.parse_args()
    output_dir = args.output or args.dataset / "matches"

    paths = list(image_files(args.dataset, args.image))
    transforms = calibrate_transforms(args.dataset, paths, args.merge_iou)
    args.visual_output.mkdir(parents=True, exist_ok=True)
    (args.visual_output / "linear_transforms.json").write_text(
        json.dumps({key: asdict(value) for key, value in transforms.items()}, indent=2),
        encoding="utf-8",
    )

    succeeded = 0
    for image_path in paths:
        width, height = bmp_dimensions(image_path)
        key = calibration_key(image_path, width, height)
        result = match_image(
            args.dataset,
            image_path,
            output_dir,
            args.visual_output,
            args.crop_output,
            args.merge_iou,
            transforms[key],
        )
        print(
            f"{image_path.name}: top={result['top_count']} bottom={result['bottom_count']} "
            f"matches={len(result['matches'])} intersections={result['intersection_count']}"
        )
        succeeded += 1
    print(f"Wrote {succeeded} data result(s) to {output_dir}")
    print(f"Wrote {succeeded} visualized image(s) to {args.visual_output}")
    print(f"Wrote matched top/bottom crop pairs to {args.crop_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
