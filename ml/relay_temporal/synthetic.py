"""Controlled synthetic landmark generator.

This generator is deliberately a development scaffold, not evidence of
performance on people. Splits use disjoint virtual subjects and camera
conditions to prevent exact-trajectory leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .constants import DATASET_SEED, EXERCISES, GENERATOR_VERSION, PHASES
from .features import extract_base_features


@dataclass(frozen=True)
class SplitConfig:
    name: str
    subjects: int
    sequences_per_exercise: int
    seed_offset: int
    yaw_limit: float
    noise_limit: float
    occlusion_probability: float


SPLIT_CONFIGS = (
    SplitConfig("train", 30, 4, 0, 20.0, 0.006, 0.10),
    SplitConfig("validation", 8, 4, 10_000, 27.0, 0.009, 0.16),
    SplitConfig("held_out", 12, 5, 20_000, 35.0, 0.013, 0.24),
)


@dataclass
class GeneratedSequence:
    features: np.ndarray
    exercise: np.ndarray
    phase: np.ndarray
    joint: np.ndarray
    view: np.ndarray
    speed: np.ndarray
    quality: np.ndarray
    subject: np.ndarray
    sequence_key: str


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    yaw_r, pitch_r, roll_r = np.radians([yaw, pitch, roll])
    cy, sy = math.cos(yaw_r), math.sin(yaw_r)
    cp, sp = math.cos(pitch_r), math.sin(pitch_r)
    cr, sr = math.cos(roll_r), math.sin(roll_r)
    yaw_m = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    pitch_m = np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    roll_m = np.asarray([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return roll_m @ pitch_m @ yaw_m


def _smooth(start: float, stop: float, count: int) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, count, dtype=np.float64)
    eased = 0.5 - 0.5 * np.cos(np.pi * progress)
    return start + (stop - start) * eased


def _motion_profile(
    exercise: str, rng: np.random.Generator
) -> tuple[np.ndarray, list[str], str]:
    speed_index = int(rng.choice(3, p=[0.24, 0.54, 0.22]))
    speed = ("fast", "normal", "slow")[speed_index]
    outbound_base = (8, 16, 26)[speed_index]
    return_base = (8, 16, 26)[speed_index]
    outbound_count = max(6, outbound_base + int(rng.integers(-3, 4)))
    return_count = max(6, return_base + int(rng.integers(-3, 4)))
    hold_count = int(rng.integers(4, 24))
    ready_before = int(rng.integers(7, 16))
    ready_after = int(rng.integers(7, 16))

    if exercise == "heel-slide":
        start = float(rng.uniform(2, 11))
        peak = float(rng.uniform(48, 108))
    elif exercise == "straight-leg-raise":
        start = float(rng.uniform(0, 6))
        peak = float(rng.uniform(27, 55))
    else:
        start = float(rng.uniform(82, 103))
        peak = float(rng.uniform(3, 24))

    values = np.concatenate(
        [
            np.full(ready_before, start),
            _smooth(start, peak, outbound_count),
            np.full(hold_count, peak),
            _smooth(peak, start, return_count),
            np.full(ready_after, start),
        ]
    )
    phases = (
        ["ready"] * ready_before
        + ["moving"] * outbound_count
        + ["holding"] * hold_count
        + ["returning"] * return_count
        + ["ready"] * ready_after
    )
    return values, phases, speed


def _skeleton(
    exercise: str,
    value: float,
    thigh: float,
    shank: float,
    torso: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    hip = np.zeros(3)
    if exercise == "heel-slide":
        flexion = math.radians(value)
        shoulder = np.asarray([-torso, -0.02, 0.0])
        knee = np.asarray([thigh, 0.02, 0.0])
        ankle = knee + np.asarray(
            [shank * math.cos(flexion), shank * math.sin(flexion), 0.0]
        )
    elif exercise == "straight-leg-raise":
        lift = math.radians(value)
        knee_bend = math.radians(float(rng.uniform(2, 9)))
        shoulder = np.asarray([-torso, -0.015, 0.0])
        knee = np.asarray(
            [thigh * math.cos(lift), -thigh * math.sin(lift), 0.0]
        )
        lower_angle = -lift + knee_bend
        ankle = knee + np.asarray(
            [
                shank * math.cos(lower_angle),
                shank * math.sin(lower_angle),
                0.0,
            ]
        )
    else:
        flexion = math.radians(value)
        shoulder = np.asarray([0.015, -torso, 0.0])
        # In a supported side view the seated thigh is approximately
        # horizontal; the lower leg moves from down (about 90°) to forward.
        knee = np.asarray([thigh, 0.03, 0.0])
        ankle = knee + np.asarray(
            [shank * math.cos(flexion), shank * math.sin(flexion), 0.0]
        )
    return {
        "shoulder": shoulder,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
    }


def _landmarks_for_frame(
    skeleton: dict[str, np.ndarray],
    side: str,
    rotation: np.ndarray,
    scale: float,
    center: tuple[float, float],
    noise: float,
    visibility: float,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    image: dict[str, dict[str, float]] = {}
    world: dict[str, dict[str, float]] = {}
    mirror = -1.0 if side == "left" else 1.0
    for name, raw_point in skeleton.items():
        point = raw_point.copy()
        point[0] *= mirror
        rotated = rotation @ point
        # Mild perspective approximates normalized camera coordinates.
        perspective = 1.0 / max(0.72, 1.0 + rotated[2] * 0.22)
        # One world unit projects to the same number of source pixels on both
        # axes. Normalized x therefore carries the 720/1280 aspect correction.
        image_x = (
            center[0] + rotated[0] * scale * (720.0 / 1280.0) * perspective
        )
        image_y = center[1] + rotated[1] * scale * perspective
        image_x += float(rng.normal(0.0, noise))
        image_y += float(rng.normal(0.0, noise))
        world_noise = noise * 0.45
        noisy_world = rotated + rng.normal(0.0, world_noise, size=3)
        key = f"{side}-{name}"
        image[key] = {
            "x": float(image_x),
            "y": float(image_y),
            "z": float(rotated[2] * scale),
            "visibility": visibility,
            "presence": visibility,
        }
        world[key] = {
            "x": float(noisy_world[0]),
            "y": float(noisy_world[1]),
            "z": float(noisy_world[2]),
            "visibility": visibility,
            "presence": visibility,
        }
    return image, world


def generate_sequence(
    split: SplitConfig,
    subject_index: int,
    exercise_index: int,
    repetition_index: int,
) -> GeneratedSequence:
    seed = (
        DATASET_SEED
        + split.seed_offset
        + subject_index * 997
        + exercise_index * 101
        + repetition_index * 17
    )
    rng = np.random.default_rng(seed)
    exercise = EXERCISES[exercise_index]
    side = "right" if rng.random() > 0.5 else "left"
    thigh = float(rng.uniform(0.39, 0.52))
    shank = float(rng.uniform(0.37, 0.51))
    torso = float(rng.uniform(0.40, 0.60))
    yaw = float(rng.uniform(-split.yaw_limit, split.yaw_limit))
    pitch = float(rng.uniform(-8.0, 8.0))
    roll = float(rng.uniform(-5.0, 5.0))
    scale = float(rng.uniform(0.38, 0.62))
    distance = "far" if scale < 0.45 else "close" if scale > 0.56 else "standard"
    center = (float(rng.uniform(0.31, 0.45)), float(rng.uniform(0.38, 0.58)))
    noise = float(rng.uniform(0.001, split.noise_limit))
    rotation = _rotation_matrix(yaw, pitch, roll)
    values, phases, speed_name = _motion_profile(exercise, rng)
    view_name = (
        "side"
        if abs(yaw) <= 12
        else "oblique"
        if abs(yaw) <= 25
        else "very-oblique"
    )

    occlusion = rng.random() < split.occlusion_probability
    occlusion_start = int(rng.integers(10, max(11, len(values) - 8)))
    occlusion_length = int(rng.integers(3, 9))
    noisy_condition = noise > split.noise_limit * 0.72
    features: list[np.ndarray] = []
    quality_labels: list[str] = []
    actual_phases: list[str] = []

    for frame_index, (value, phase) in enumerate(zip(values, phases)):
        in_occlusion = (
            occlusion
            and occlusion_start <= frame_index < occlusion_start + occlusion_length
        )
        visibility = (
            float(rng.uniform(0.12, 0.38))
            if in_occlusion
            else float(rng.uniform(0.86, 0.99))
        )
        skeleton = _skeleton(
            exercise,
            float(value + rng.normal(0.0, 0.35)),
            thigh,
            shank,
            torso,
            rng,
        )
        image, world = _landmarks_for_frame(
            skeleton,
            side,
            rotation,
            scale,
            center,
            noise * (2.2 if in_occlusion else 1.0),
            visibility,
            rng,
        )
        features.append(extract_base_features(image, world, side))
        quality_labels.append(
            "occluded" if in_occlusion else "noisy" if noisy_condition else "clean"
        )
        actual_phases.append("paused" if in_occlusion else phase)

    exercise_labels = np.full(len(features), exercise_index, dtype=np.int64)
    phase_labels = np.asarray(
        [PHASES.index(phase) for phase in actual_phases], dtype=np.int64
    )
    joint_labels = exercise_labels * len(PHASES) + phase_labels
    subject_id = split.seed_offset + subject_index
    return GeneratedSequence(
        features=np.stack(features).astype(np.float32),
        exercise=exercise_labels,
        phase=phase_labels,
        joint=joint_labels,
        view=np.full(len(features), view_name),
        speed=np.full(len(features), speed_name),
        quality=np.asarray(quality_labels),
        subject=np.full(len(features), subject_id, dtype=np.int64),
        sequence_key=(
            f"{split.name}-s{subject_id:05d}-{exercise}-r{repetition_index:02d}"
            f"-{side}-{view_name}-{distance}"
        ),
    )


def generate_split(config: SplitConfig) -> list[GeneratedSequence]:
    return [
        generate_sequence(config, subject, exercise, repetition)
        for subject in range(config.subjects)
        for exercise in range(len(EXERCISES))
        for repetition in range(config.sequences_per_exercise)
    ]


def provenance() -> dict[str, Any]:
    return {
        "generatorVersion": GENERATOR_VERSION,
        "seed": DATASET_SEED,
        "origin": "synthetic",
        "containsRealPeople": False,
        "intendedEvidence": (
            "software plumbing, split discipline, temporal learning feasibility, "
            "and deterministic regression coverage"
        ),
        "notValidEvidenceFor": (
            "accuracy on people, clinical validity, camera-pose SDK error, "
            "safety at a prescribed range limit, or demographic generalization"
        ),
        "splits": [
            {
                "name": config.name,
                "subjects": config.subjects,
                "sequencesPerExercise": config.sequences_per_exercise,
                "subjectSeedRange": [
                    config.seed_offset,
                    config.seed_offset + config.subjects - 1,
                ],
                "yawLimitDegrees": config.yaw_limit,
                "noiseLimitNormalized": config.noise_limit,
                "occlusionProbability": config.occlusion_probability,
            }
            for config in SPLIT_CONFIGS
        ],
        "splitUnit": "virtual subject; no subject appears in multiple splits",
    }
