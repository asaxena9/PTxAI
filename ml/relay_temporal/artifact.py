"""Custom int8 artifact export and parity runtime."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import (
    EXERCISES,
    FEATURE_NAMES,
    GENERATOR_VERSION,
    JOINT_LABELS,
    PHASES,
    SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
    TRAINING_SEED,
    WINDOW_SIZE,
)
from .model import RelayTemporalNet


def _quantize(values: np.ndarray) -> dict[str, Any]:
    maximum = float(np.max(np.abs(values)))
    scale = maximum / 127.0 if maximum > 0 else 1.0
    quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return {
        "shape": list(values.shape),
        "scale": scale,
        "encoding": "base64-int8",
        "data": base64.b64encode(quantized.tobytes()).decode("ascii"),
    }


def export_artifact(
    model: RelayTemporalNet,
    output: Path,
    mean: np.ndarray,
    scale: np.ndarray,
    temperature: float,
    minimum_confidence: float,
    minimum_margin: float,
    validation_summary: dict[str, float],
) -> dict[str, Any]:
    state = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in model.state_dict().items()
    }
    artifact: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "modelId": "relay-temporal-synthetic-v1",
        "modelKind": "causal-temporal-convolution",
        "intendedRuntime": "browser-main-thread-or-worker",
        "input": {
            "windowSize": WINDOW_SIZE,
            "sampleRateHz": SAMPLE_RATE_HZ,
            "featureNames": list(FEATURE_NAMES),
        },
        "labels": {
            "exercises": list(EXERCISES),
            "phases": list(PHASES),
            "joint": list(JOINT_LABELS),
        },
        "normalization": {
            "mean": mean.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
        },
        "architecture": {
            "hiddenChannels": int(state["conv1.weight"].shape[0]),
            "kernelSize": int(state["conv1.weight"].shape[2]),
            "pooling": "temporal-mean-plus-last",
            "activation": "relu",
        },
        "tensors": {
            "conv1.weight": _quantize(state["conv1.weight"]),
            "conv1.bias": state["conv1.bias"].astype(float).tolist(),
            "conv2.weight": _quantize(state["conv2.weight"]),
            "conv2.bias": state["conv2.bias"].astype(float).tolist(),
            "head.weight": _quantize(state["head.weight"]),
            "head.bias": state["head.bias"].astype(float).tolist(),
        },
        "calibration": {
            "temperature": float(temperature),
            "selectionSource": "synthetic validation split",
        },
        "abstention": {
            "minimumConfidence": float(minimum_confidence),
            "minimumMargin": float(minimum_margin),
            "minimumCurrentVisibility": 0.45,
            "minimumWarmupFrames": 6,
            "selectionSource": "synthetic validation split plus explicit quality gate",
        },
        "training": {
            "seed": TRAINING_SEED,
            "dataOrigin": "synthetic",
            "containsRealPeople": False,
            "generatorVersion": GENERATOR_VERSION,
            "validationSummary": validation_summary,
        },
        "limitations": [
            "Trained only on generated landmarks, not camera recordings of people.",
            "Does not validate clinical accuracy, safety, or pose-SDK landmark error.",
            "Exercise identity is inherently ambiguous before visible movement.",
            "Outputs are supporting evidence; explicit geometry and confirmed rules remain authoritative.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _decode_tensor(tensor: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(tensor["data"])
    quantized = np.frombuffer(raw, dtype=np.int8)
    return (
        quantized.astype(np.float32).reshape(tensor["shape"])
        * float(tensor["scale"])
    )


def load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _causal_conv(
    inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    kernel = weights.shape[2]
    padded = np.pad(inputs, ((0, 0), (kernel - 1, 0), (0, 0)))
    output = np.broadcast_to(
        bias[None, None, :], (len(inputs), inputs.shape[1], len(bias))
    ).copy()
    for kernel_index in range(kernel):
        output += np.einsum(
            "btf,of->bto",
            padded[:, kernel_index : kernel_index + inputs.shape[1], :],
            weights[:, :, kernel_index],
            optimize=True,
        )
    return output


def artifact_logits(artifact: dict[str, Any], windows: np.ndarray) -> np.ndarray:
    mean = np.asarray(artifact["normalization"]["mean"], dtype=np.float32)
    scale = np.asarray(artifact["normalization"]["scale"], dtype=np.float32)
    normalized = (windows - mean[None, None, :]) / scale[None, None, :]
    tensors = artifact["tensors"]
    first = np.maximum(
        0.0,
        _causal_conv(
            normalized,
            _decode_tensor(tensors["conv1.weight"]),
            np.asarray(tensors["conv1.bias"], dtype=np.float32),
        ),
    )
    second = np.maximum(
        0.0,
        _causal_conv(
            first,
            _decode_tensor(tensors["conv2.weight"]),
            np.asarray(tensors["conv2.bias"], dtype=np.float32),
        ),
    )
    pooled = np.concatenate([second.mean(axis=1), second[:, -1, :]], axis=1)
    head_weight = _decode_tensor(tensors["head.weight"])
    head_bias = np.asarray(tensors["head.bias"], dtype=np.float32)
    return pooled @ head_weight.T + head_bias


def artifact_predict(artifact: dict[str, Any], windows: np.ndarray) -> np.ndarray:
    return np.argmax(artifact_logits(artifact, windows), axis=1)
