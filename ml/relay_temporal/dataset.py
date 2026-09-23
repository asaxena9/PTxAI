"""Dataset serialization and trailing-window construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    BASE_FEATURE_NAMES,
    DATASET_SEED,
    GENERATOR_VERSION,
    WINDOW_SIZE,
)
from .features import window_with_derivatives
from .synthetic import GeneratedSequence, SPLIT_CONFIGS, generate_split, provenance


def _flatten_sequences(sequences: list[GeneratedSequence]) -> dict[str, np.ndarray]:
    offsets = [0]
    keys: list[str] = []
    arrays: dict[str, list[np.ndarray]] = {
        "features": [],
        "exercise": [],
        "phase": [],
        "joint": [],
        "view": [],
        "speed": [],
        "quality": [],
        "subject": [],
    }
    for sequence in sequences:
        keys.append(sequence.sequence_key)
        for name in arrays:
            arrays[name].append(getattr(sequence, name))
        offsets.append(offsets[-1] + len(sequence.features))
    flattened = {
        name: np.concatenate(values, axis=0) for name, values in arrays.items()
    }
    flattened["offsets"] = np.asarray(offsets, dtype=np.int64)
    flattened["sequence_keys"] = np.asarray(keys)
    return flattened


def prepare_dataset(output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "schemaVersion": 1,
        "generatorVersion": GENERATOR_VERSION,
        "seed": DATASET_SEED,
        "origin": "synthetic",
        "containsRealPeople": False,
        "baseFeatureNames": list(BASE_FEATURE_NAMES),
        "windowSize": WINDOW_SIZE,
        "splits": {},
        "provenance": provenance(),
    }
    for config in SPLIT_CONFIGS:
        flattened = _flatten_sequences(generate_split(config))
        prefix = config.name
        for name, values in flattened.items():
            payload[f"{prefix}_{name}"] = values
        summary["splits"][prefix] = {
            "frames": int(len(flattened["features"])),
            "sequences": int(len(flattened["offsets"]) - 1),
            "subjects": int(len(np.unique(flattened["subject"]))),
        }
    np.savez_compressed(output, **payload)
    summary["archiveBytes"] = output.stat().st_size
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_split(
    archive: np.lib.npyio.NpzFile, name: str
) -> dict[str, np.ndarray]:
    return {
        key: archive[f"{name}_{key}"]
        for key in (
            "features",
            "exercise",
            "phase",
            "joint",
            "view",
            "speed",
            "quality",
            "subject",
            "offsets",
            "sequence_keys",
        )
    }


def build_windows(split: dict[str, np.ndarray]) -> np.ndarray:
    base = split["features"]
    offsets = split["offsets"]
    windows = np.empty(
        (len(base), WINDOW_SIZE, base.shape[1] + 4), dtype=np.float32
    )
    for sequence_index in range(len(offsets) - 1):
        start, stop = int(offsets[sequence_index]), int(offsets[sequence_index + 1])
        sequence = base[start:stop]
        for local_index in range(len(sequence)):
            windows[start + local_index] = window_with_derivatives(
                sequence, local_index
            )
    return windows
