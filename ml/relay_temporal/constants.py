"""Shared constants for training, export, and browser parity."""

from __future__ import annotations

EXERCISES = (
    "heel-slide",
    "straight-leg-raise",
    "seated-knee-extension",
)
PHASES = ("ready", "moving", "holding", "returning", "paused")
JOINT_LABELS = tuple(
    f"{exercise}:{phase}" for exercise in EXERCISES for phase in PHASES
)

BASE_FEATURE_NAMES = (
    "knee_flexion_image",
    "knee_flexion_world",
    "image_hip_to_knee_x",
    "image_hip_to_knee_y",
    "image_knee_to_ankle_x",
    "image_knee_to_ankle_y",
    "world_hip_to_knee_x",
    "world_hip_to_knee_y",
    "world_hip_to_knee_z",
    "world_knee_to_ankle_x",
    "world_knee_to_ankle_y",
    "world_knee_to_ankle_z",
    "image_shoulder_to_hip_x",
    "image_shoulder_to_hip_y",
    "minimum_visibility",
    "mean_visibility",
    "angle_agreement",
    "framing_margin",
)

DERIVATIVE_FEATURE_INDICES = (0, 1, 4, 5)
DERIVATIVE_FEATURE_NAMES = tuple(
    f"delta_{BASE_FEATURE_NAMES[index]}" for index in DERIVATIVE_FEATURE_INDICES
)
FEATURE_NAMES = BASE_FEATURE_NAMES + DERIVATIVE_FEATURE_NAMES

WINDOW_SIZE = 24
SAMPLE_RATE_HZ = 10
SCHEMA_VERSION = 1
GENERATOR_VERSION = "relay-synthetic-pose-v1"
DATASET_SEED = 73021
TRAINING_SEED = 9173
