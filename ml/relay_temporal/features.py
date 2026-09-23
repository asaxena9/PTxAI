"""Pose-derived features mirrored by ``src/motion/temporal-model.ts``."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .constants import DERIVATIVE_FEATURE_INDICES, WINDOW_SIZE

Landmark = Mapping[str, float]
Landmarks = Mapping[str, Landmark]


def _safe_landmark(
    landmarks: Landmarks, name: str
) -> tuple[float, float, float, float]:
    point = landmarks.get(name)
    if point is None:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(point.get("x", 0.0)),
        float(point.get("y", 0.0)),
        float(point.get("z", 0.0)),
        min(
            float(point.get("visibility", 0.0)),
            float(point.get("presence", 0.0)),
        ),
    )


def _angle(
    a: tuple[float, float, float],
    vertex: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    first = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    second = np.asarray(c, dtype=np.float64) - np.asarray(
        vertex, dtype=np.float64
    )
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def extract_base_features(
    image: Landmarks,
    world: Landmarks,
    side: str,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> np.ndarray:
    """Extract scale- and side-normalized features from one pose frame.

    Missing joints become zeros and are represented explicitly by visibility
    features. Image geometry is converted to source pixels before angles are
    calculated so landscape aspect ratio does not distort the estimate.
    """

    prefix = f"{side}-"
    required = ("shoulder", "hip", "knee", "ankle")
    image_points = {
        joint: _safe_landmark(image, prefix + joint) for joint in required
    }
    world_points = {
        joint: _safe_landmark(world, prefix + joint) for joint in required
    }
    direction = -1.0 if side == "left" else 1.0

    image_xyz = {
        name: (
            point[0] * frame_width,
            point[1] * frame_height,
            point[2] * frame_width,
        )
        for name, point in image_points.items()
    }
    world_xyz = {name: point[:3] for name, point in world_points.items()}

    image_lengths = (
        math.dist(image_xyz["hip"], image_xyz["knee"])
        + math.dist(image_xyz["knee"], image_xyz["ankle"])
    )
    world_lengths = (
        math.dist(world_xyz["hip"], world_xyz["knee"])
        + math.dist(world_xyz["knee"], world_xyz["ankle"])
    )
    image_scale = max(image_lengths, 1e-6)
    world_scale = max(world_lengths, 1e-6)

    def image_vector(start: str, end: str) -> tuple[float, float]:
        return (
            direction
            * (image_xyz[end][0] - image_xyz[start][0])
            / image_scale,
            (image_xyz[end][1] - image_xyz[start][1]) / image_scale,
        )

    def world_vector(start: str, end: str) -> tuple[float, float, float]:
        return (
            direction
            * (world_xyz[end][0] - world_xyz[start][0])
            / world_scale,
            (world_xyz[end][1] - world_xyz[start][1]) / world_scale,
            direction
            * (world_xyz[end][2] - world_xyz[start][2])
            / world_scale,
        )

    image_flexion = (
        180.0
        - _angle(
            image_xyz["hip"],
            image_xyz["knee"],
            image_xyz["ankle"],
        )
    ) / 180.0
    world_flexion = (
        180.0
        - _angle(
            world_xyz["hip"],
            world_xyz["knee"],
            world_xyz["ankle"],
        )
    ) / 180.0
    image_hip_knee = image_vector("hip", "knee")
    image_knee_ankle = image_vector("knee", "ankle")
    world_hip_knee = world_vector("hip", "knee")
    world_knee_ankle = world_vector("knee", "ankle")
    image_shoulder_hip = image_vector("shoulder", "hip")

    visibilities = np.asarray(
        [image_points[joint][3] for joint in required], dtype=np.float32
    )
    all_image_points = np.asarray(
        [[image_points[joint][0], image_points[joint][1]] for joint in required],
        dtype=np.float32,
    )
    framing_margin = float(
        np.clip(
            np.min(
                np.concatenate(
                    [
                        all_image_points[:, 0],
                        1.0 - all_image_points[:, 0],
                        all_image_points[:, 1],
                        1.0 - all_image_points[:, 1],
                    ]
                )
            )
            / 0.12,
            0.0,
            1.0,
        )
    )

    return np.asarray(
        [
            image_flexion,
            world_flexion,
            *image_hip_knee,
            *image_knee_ankle,
            *world_hip_knee,
            *world_knee_ankle,
            *image_shoulder_hip,
            float(np.min(visibilities)),
            float(np.mean(visibilities)),
            abs(image_flexion - world_flexion),
            framing_margin,
        ],
        dtype=np.float32,
    )


def window_with_derivatives(
    sequence: np.ndarray, end_index: int, window_size: int = WINDOW_SIZE
) -> np.ndarray:
    """Build a trailing window, padding only within the current sequence."""

    start = max(0, end_index - window_size + 1)
    base = sequence[start : end_index + 1]
    if len(base) < window_size:
        padding = np.repeat(base[:1], window_size - len(base), axis=0)
        base = np.concatenate([padding, base], axis=0)
    derivatives = np.zeros(
        (window_size, len(DERIVATIVE_FEATURE_INDICES)), dtype=np.float32
    )
    derivatives[1:] = (
        base[1:, DERIVATIVE_FEATURE_INDICES]
        - base[:-1, DERIVATIVE_FEATURE_INDICES]
    )
    return np.concatenate([base, derivatives], axis=1).astype(np.float32)
