"""Evaluation helpers with per-condition and transition reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .constants import EXERCISES, JOINT_LABELS, PHASES


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, labels: tuple[str, ...]
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    supported = support > 0
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macroPrecision": float(np.mean(precision[supported])),
        "macroRecall": float(np.mean(recall[supported])),
        "macroF1": float(np.mean(f1[supported])),
        "perClass": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def evaluate_predictions(
    truth_joint: np.ndarray,
    prediction_joint: np.ndarray,
) -> dict[str, Any]:
    truth_exercise = truth_joint // len(PHASES)
    prediction_exercise = prediction_joint // len(PHASES)
    truth_phase = truth_joint % len(PHASES)
    prediction_phase = prediction_joint % len(PHASES)
    active = truth_phase != PHASES.index("ready")
    return {
        "joint": classification_metrics(
            truth_joint, prediction_joint, JOINT_LABELS
        ),
        "exercise": classification_metrics(
            truth_exercise, prediction_exercise, EXERCISES
        ),
        "exerciseActiveFrames": classification_metrics(
            truth_exercise[active],
            prediction_exercise[active],
            EXERCISES,
        ),
        "phase": classification_metrics(truth_phase, prediction_phase, PHASES),
    }


def condition_breakdown(
    truth_joint: np.ndarray,
    prediction_joint: np.ndarray,
    condition: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in np.unique(condition):
        mask = condition == value
        evaluation = evaluate_predictions(
            truth_joint[mask], prediction_joint[mask]
        )
        result[str(value)] = {
            "frames": int(np.sum(mask)),
            "jointMacroF1": evaluation["joint"]["macroF1"],
            "exerciseMacroF1": evaluation["exercise"]["macroF1"],
            "phaseMacroF1": evaluation["phase"]["macroF1"],
        }
    return result


def boundary_metrics(
    truth_joint: np.ndarray,
    prediction_joint: np.ndarray,
    offsets: np.ndarray,
    tolerance_frames: int = 2,
) -> dict[str, float | int]:
    matched = 0
    truth_count = 0
    predicted_count = 0
    absolute_errors: list[int] = []
    for sequence_index in range(len(offsets) - 1):
        start, stop = int(offsets[sequence_index]), int(offsets[sequence_index + 1])
        truth_phase = truth_joint[start:stop] % len(PHASES)
        prediction_phase = prediction_joint[start:stop] % len(PHASES)
        truth_boundaries = np.flatnonzero(truth_phase[1:] != truth_phase[:-1]) + 1
        predicted_boundaries = (
            np.flatnonzero(prediction_phase[1:] != prediction_phase[:-1]) + 1
        )
        truth_count += len(truth_boundaries)
        predicted_count += len(predicted_boundaries)
        available = set(range(len(predicted_boundaries)))
        for boundary in truth_boundaries:
            candidates = [
                (abs(int(predicted_boundaries[index]) - int(boundary)), index)
                for index in available
                if abs(int(predicted_boundaries[index]) - int(boundary))
                <= tolerance_frames
            ]
            if candidates:
                error, selected = min(candidates)
                available.remove(selected)
                matched += 1
                absolute_errors.append(error)
    precision = matched / predicted_count if predicted_count else 0.0
    recall = matched / truth_count if truth_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "toleranceFrames": tolerance_frames,
        "matched": matched,
        "truthBoundaries": truth_count,
        "predictedBoundaries": predicted_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "meanAbsoluteFrameError": (
            float(np.mean(absolute_errors)) if absolute_errors else 0.0
        ),
    }


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    adjusted = logits / temperature
    adjusted -= np.max(adjusted, axis=1, keepdims=True)
    exponent = np.exp(adjusted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def expected_calibration_error(
    truth: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = np.argmax(probabilities, axis=1) == truth
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask])) - float(np.mean(correct[mask]))
            )
    return result


def fit_temperature(truth: np.ndarray, logits: np.ndarray) -> float:
    candidates = np.linspace(0.5, 3.0, 101)
    losses = []
    for temperature in candidates:
        probability = softmax(logits, float(temperature))
        losses.append(
            -float(
                np.mean(
                    np.log(
                        np.clip(
                            probability[np.arange(len(truth)), truth],
                            1e-8,
                            1.0,
                        )
                    )
                )
            )
        )
    return float(candidates[int(np.argmin(losses))])


def select_confidence_threshold(
    truth: np.ndarray,
    probabilities: np.ndarray,
    target_selective_accuracy: float = 0.90,
) -> tuple[float, dict[str, float]]:
    confidence = np.max(probabilities, axis=1)
    prediction = np.argmax(probabilities, axis=1)
    correct = prediction == truth
    selected_threshold = 0.70
    selected = {
        "coverage": 0.0,
        "selectiveAccuracy": 0.0,
        "targetSelectiveAccuracy": target_selective_accuracy,
    }
    for threshold in np.linspace(0.35, 0.90, 112):
        mask = confidence >= threshold
        if not np.any(mask):
            continue
        coverage = float(np.mean(mask))
        accuracy = float(np.mean(correct[mask]))
        if accuracy >= target_selective_accuracy and coverage > selected["coverage"]:
            selected_threshold = float(threshold)
            selected = {
                "coverage": coverage,
                "selectiveAccuracy": accuracy,
                "targetSelectiveAccuracy": target_selective_accuracy,
            }
    return selected_threshold, selected
