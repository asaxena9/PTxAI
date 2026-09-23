"""Auditable kinematic baseline for comparison with the learned model."""

from __future__ import annotations

import numpy as np

from .constants import EXERCISES, PHASES


def predict_baseline(
    windows: np.ndarray, expected_exercise: np.ndarray | None = None
) -> np.ndarray:
    """Classify exercise and current phase using explicit geometry thresholds."""

    current = windows[:, -1, :]
    knee_flexion = current[:, 0] * 180.0
    knee_velocity = current[:, 18] * 180.0
    shank_x = current[:, 4]
    shank_y = current[:, 5]
    shoulder_x = current[:, 12]
    shoulder_y = current[:, 13]
    minimum_visibility = current[:, 14]

    seated = np.abs(shoulder_y) > np.maximum(0.25, np.abs(shoulder_x) * 1.4)
    leg_elevation = np.degrees(
        np.arctan2(
            -(current[:, 3] + current[:, 5]),
            np.maximum(1e-4, current[:, 2] + current[:, 4]),
        )
    )
    straight_raise = (~seated) & (
        (leg_elevation > 10.0)
        | ((np.abs(knee_velocity) < 0.8) & (knee_flexion < 15.0))
    )
    exercise = (
        expected_exercise.astype(np.int64)
        if expected_exercise is not None
        else np.where(
            seated,
            EXERCISES.index("seated-knee-extension"),
            np.where(
                straight_raise,
                EXERCISES.index("straight-leg-raise"),
                EXERCISES.index("heel-slide"),
            ),
        )
    )

    phase = np.full(len(windows), PHASES.index("ready"), dtype=np.int64)
    phase[minimum_visibility < 0.45] = PHASES.index("paused")
    usable = minimum_visibility >= 0.45

    heel = usable & (exercise == EXERCISES.index("heel-slide"))
    phase[heel & (knee_velocity > 0.8)] = PHASES.index("moving")
    phase[heel & (knee_velocity < -0.8)] = PHASES.index("returning")
    phase[
        heel & (knee_flexion > 35.0) & (np.abs(knee_velocity) <= 0.8)
    ] = PHASES.index("holding")

    raise_mask = usable & (
        exercise == EXERCISES.index("straight-leg-raise")
    )
    elevation_velocity = -(current[:, 20] + current[:, 21]) * 90.0
    phase[raise_mask & (elevation_velocity > 0.35)] = PHASES.index("moving")
    phase[raise_mask & (elevation_velocity < -0.35)] = PHASES.index(
        "returning"
    )
    phase[
        raise_mask & (leg_elevation > 18.0) & (np.abs(elevation_velocity) <= 0.35)
    ] = PHASES.index("holding")

    seated_mask = usable & (
        exercise == EXERCISES.index("seated-knee-extension")
    )
    phase[seated_mask & (knee_velocity < -0.8)] = PHASES.index("moving")
    phase[seated_mask & (knee_velocity > 0.8)] = PHASES.index("returning")
    phase[
        seated_mask & (knee_flexion < 35.0) & (np.abs(knee_velocity) <= 0.8)
    ] = PHASES.index("holding")

    return exercise * len(PHASES) + phase
