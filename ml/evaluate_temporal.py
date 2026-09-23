#!/usr/bin/env python3
"""Compare the quantized temporal artifact with a kinematic baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from relay_temporal.artifact import (
    artifact_logits,
    artifact_predict,
    load_artifact,
)
from relay_temporal.baseline import predict_baseline
from relay_temporal.constants import FEATURE_NAMES, JOINT_LABELS, PHASES
from relay_temporal.dataset import build_windows, load_split
from relay_temporal.metrics import (
    boundary_metrics,
    condition_breakdown,
    evaluate_predictions,
    expected_calibration_error,
    softmax,
)
from relay_temporal.model import RelayTemporalNet


def float_checkpoint_logits(
    checkpoint_path: Path, windows: np.ndarray
) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = RelayTemporalNet(len(FEATURE_NAMES), len(JOINT_LABELS))
    model.load_state_dict(checkpoint["stateDict"])
    model.eval()
    mean = checkpoint["mean"].numpy()
    scale = checkpoint["scale"].numpy()
    normalized = (windows - mean[None, None, :]) / scale[None, None, :]
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 1024):
            output.append(
                model(
                    torch.from_numpy(normalized[start : start + 1024])
                ).numpy()
            )
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml/.generated/relay-temporal-synthetic-v1.npz"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("public/models/temporal/relay-temporal-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/results/held-out-evaluation.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml/.generated/relay-temporal-v1.pt"),
    )
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    with np.load(args.data) as archive:
        held_out = load_split(archive, "held_out")
        windows = build_windows(held_out)
    learned_logits = artifact_logits(artifact, windows)
    learned = np.argmax(learned_logits, axis=1)
    float_logits = float_checkpoint_logits(args.checkpoint, windows)
    float_prediction = np.argmax(float_logits, axis=1)
    baseline = predict_baseline(windows)
    expected_exercise = held_out["exercise"].astype(np.int64)
    expected_phase_logits = learned_logits.reshape(
        len(windows), -1, len(PHASES)
    )[np.arange(len(windows)), expected_exercise]
    learned_expected_phase = np.argmax(expected_phase_logits, axis=1)
    learned_expected_joint = (
        expected_exercise * len(PHASES) + learned_expected_phase
    )
    baseline_expected = predict_baseline(windows, expected_exercise)
    temperature = float(artifact["calibration"]["temperature"])
    probabilities = softmax(learned_logits, temperature)
    probability_cube = probabilities.reshape(
        len(probabilities), -1, len(PHASES)
    )
    confirmed_phase_probabilities = probability_cube[
        np.arange(len(probabilities)), expected_exercise
    ]
    confirmed_phase_probabilities /= np.sum(
        confirmed_phase_probabilities, axis=1, keepdims=True
    )
    confidence = np.max(confirmed_phase_probabilities, axis=1)
    sorted_probabilities = np.sort(confirmed_phase_probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    exercise_probabilities = np.sum(probability_cube, axis=2)
    predicted_exercise = np.argmax(exercise_probabilities, axis=1)
    exercise_confidence = np.max(exercise_probabilities, axis=1)
    abstention = artifact["abstention"]
    accepted = (
        (confidence >= float(abstention["minimumConfidence"]))
        & (margin >= float(abstention["minimumMargin"]))
        & (windows[:, -1, 14] >= float(abstention["minimumCurrentVisibility"]))
        & ~(
            (predicted_exercise != expected_exercise)
            & (
                exercise_confidence
                >= float(abstention["minimumConfidence"])
            )
        )
    )
    correct = learned_expected_phase == held_out["phase"]

    learned_metrics = evaluate_predictions(held_out["joint"], learned)
    learned_metrics["boundaries"] = boundary_metrics(
        held_out["joint"], learned, held_out["offsets"]
    )
    learned_metrics["calibration"] = {
        "temperature": temperature,
        "jointExpectedCalibrationError": expected_calibration_error(
            held_out["joint"], probabilities
        ),
        "confirmedExercisePhaseExpectedCalibrationError": (
            expected_calibration_error(
                held_out["phase"], confirmed_phase_probabilities
            )
        ),
        "meanConfidence": float(np.mean(confidence)),
        "coverageAfterAbstention": float(np.mean(accepted)),
        "selectivePhaseAccuracy": (
            float(np.mean(correct[accepted])) if np.any(accepted) else 0.0
        ),
        "task": "phase classification conditioned on confirmed exercise",
    }
    learned_metrics["breakdown"] = {
        "view": condition_breakdown(
            held_out["joint"], learned, held_out["view"]
        ),
        "speed": condition_breakdown(
            held_out["joint"], learned, held_out["speed"]
        ),
        "trackingQuality": condition_breakdown(
            held_out["joint"], learned, held_out["quality"]
        ),
    }
    learned_metrics["phaseGivenConfirmedExercise"] = evaluate_predictions(
        held_out["joint"], learned_expected_joint
    )["phase"]

    baseline_metrics = evaluate_predictions(held_out["joint"], baseline)
    baseline_metrics["boundaries"] = boundary_metrics(
        held_out["joint"], baseline, held_out["offsets"]
    )
    baseline_metrics["breakdown"] = {
        "view": condition_breakdown(
            held_out["joint"], baseline, held_out["view"]
        ),
        "speed": condition_breakdown(
            held_out["joint"], baseline, held_out["speed"]
        ),
        "trackingQuality": condition_breakdown(
            held_out["joint"], baseline, held_out["quality"]
        ),
    }
    baseline_metrics["phaseGivenConfirmedExercise"] = evaluate_predictions(
        held_out["joint"], baseline_expected
    )["phase"]
    artifact_bytes = args.artifact.read_bytes()
    result = {
        "evaluationVersion": 1,
        "evidenceScope": "synthetic held-out virtual subjects only",
        "containsRealPeople": False,
        "warning": (
            "These results validate software behavior and feasibility, not "
            "accuracy, safety, or generalization on human camera recordings."
        ),
        "heldOut": {
            "frames": int(len(windows)),
            "sequences": int(len(held_out["offsets"]) - 1),
            "subjects": int(len(np.unique(held_out["subject"]))),
            "splitUnit": "virtual subject",
        },
        "learnedQuantizedTemporalModel": learned_metrics,
        "kinematicBaseline": baseline_metrics,
        "comparison": {
            "jointMacroF1Delta": (
                learned_metrics["joint"]["macroF1"]
                - baseline_metrics["joint"]["macroF1"]
            ),
            "exerciseMacroF1Delta": (
                learned_metrics["exercise"]["macroF1"]
                - baseline_metrics["exercise"]["macroF1"]
            ),
            "phaseMacroF1Delta": (
                learned_metrics["phase"]["macroF1"]
                - baseline_metrics["phase"]["macroF1"]
            ),
            "phaseGivenConfirmedExerciseMacroF1Delta": (
                learned_metrics["phaseGivenConfirmedExercise"]["macroF1"]
                - baseline_metrics["phaseGivenConfirmedExercise"]["macroF1"]
            ),
        },
        "quantization": {
            "floatCheckpointJointMacroF1": evaluate_predictions(
                held_out["joint"], float_prediction
            )["joint"]["macroF1"],
            "int8ArtifactJointMacroF1": learned_metrics["joint"]["macroF1"],
            "predictionAgreement": float(
                np.mean(float_prediction == learned)
            ),
        },
        "artifact": {
            "path": str(args.artifact),
            "bytes": len(artifact_bytes),
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))
    print(
        "learned joint/phase/exercise macro-F1:",
        f"{learned_metrics['joint']['macroF1']:.4f}",
        f"{learned_metrics['phase']['macroF1']:.4f}",
        f"{learned_metrics['exercise']['macroF1']:.4f}",
    )
    print(
        "baseline joint/phase/exercise macro-F1:",
        f"{baseline_metrics['joint']['macroF1']:.4f}",
        f"{baseline_metrics['phase']['macroF1']:.4f}",
        f"{baseline_metrics['exercise']['macroF1']:.4f}",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
