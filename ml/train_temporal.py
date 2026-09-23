#!/usr/bin/env python3
"""Train and export Relay's compact temporal movement component."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from relay_temporal.artifact import export_artifact
from relay_temporal.constants import (
    FEATURE_NAMES,
    JOINT_LABELS,
    PHASES,
    TRAINING_SEED,
)
from relay_temporal.dataset import build_windows, load_split
from relay_temporal.metrics import (
    expected_calibration_error,
    fit_temperature,
    select_confidence_threshold,
    softmax,
)
from relay_temporal.model import RelayTemporalNet


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def logits_for(
    model: RelayTemporalNet, windows: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size])
            output.append(model(batch).cpu().numpy())
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
        "--history",
        type=Path,
        default=Path("ml/results/training-history.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml/.generated/relay-temporal-v1.pt"),
    )
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    set_deterministic(TRAINING_SEED)

    with np.load(args.data) as archive:
        train = load_split(archive, "train")
        validation = load_split(archive, "validation")
        train_windows = build_windows(train)
        validation_windows = build_windows(validation)

    mean = train_windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = train_windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    train_windows = (train_windows - mean[None, None, :]) / scale[
        None, None, :
    ]
    validation_windows = (
        validation_windows - mean[None, None, :]
    ) / scale[None, None, :]

    labels = train["joint"].astype(np.int64)
    counts = np.bincount(labels, minlength=len(JOINT_LABELS))
    weights = np.sqrt(len(labels) / np.maximum(counts, 1))
    weights /= np.mean(weights)

    dataset = TensorDataset(
        torch.from_numpy(train_windows), torch.from_numpy(labels)
    )
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = RelayTemporalNet(len(FEATURE_NAMES), len(JOINT_LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights.astype(np.float32))
    )
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_logits = logits_for(model, validation_windows)
        prediction = np.argmax(validation_logits, axis=1)
        macro_f1 = float(
            f1_score(
                validation["joint"],
                prediction,
                labels=np.arange(len(JOINT_LABELS)),
                average="macro",
                zero_division=0,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "trainingLoss": float(np.mean(losses)),
                "validationJointMacroF1": macro_f1,
            }
        )
        print(
            f"epoch={epoch:02d} loss={np.mean(losses):.4f} "
            f"validation_joint_macro_f1={macro_f1:.4f}"
        )
        if macro_f1 > best_f1 + 1e-4:
            best_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break

    model.load_state_dict(best_state)
    validation_logits = logits_for(model, validation_windows)
    temperature = fit_temperature(validation["joint"], validation_logits)
    probabilities = softmax(validation_logits, temperature)
    joint_probabilities = probabilities.reshape(
        len(probabilities), -1, len(PHASES)
    )
    confirmed_phase_probabilities = joint_probabilities[
        np.arange(len(probabilities)), validation["exercise"]
    ]
    confirmed_phase_probabilities /= np.sum(
        confirmed_phase_probabilities, axis=1, keepdims=True
    )
    confidence_threshold, threshold_metrics = select_confidence_threshold(
        validation["phase"], confirmed_phase_probabilities
    )
    sorted_probabilities = np.sort(confirmed_phase_probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    correct = (
        np.argmax(confirmed_phase_probabilities, axis=1)
        == validation["phase"]
    )
    margin_threshold = float(
        np.clip(np.percentile(margins[correct], 10), 0.05, 0.25)
    )
    validation_summary = {
        "bestJointMacroF1": best_f1,
        "temperature": temperature,
        "jointExpectedCalibrationError": expected_calibration_error(
            validation["joint"], probabilities
        ),
        "confirmedExercisePhaseExpectedCalibrationError": (
            expected_calibration_error(
                validation["phase"], confirmed_phase_probabilities
            )
        ),
        "minimumConfidence": confidence_threshold,
        "minimumMargin": margin_threshold,
        "thresholdTask": "phase classification conditioned on confirmed exercise",
        **threshold_metrics,
    }
    export_artifact(
        model,
        args.artifact,
        mean,
        scale,
        temperature,
        confidence_threshold,
        margin_threshold,
        validation_summary,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stateDict": model.state_dict(),
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "featureCount": len(FEATURE_NAMES),
            "classCount": len(JOINT_LABELS),
            "seed": TRAINING_SEED,
        },
        args.checkpoint,
    )
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(
        json.dumps(
            {
                "seed": TRAINING_SEED,
                "epochsRequested": args.epochs,
                "epochsCompleted": len(history),
                "bestEpoch": int(
                    max(
                        range(len(history)),
                        key=lambda index: history[index][
                            "validationJointMacroF1"
                        ],
                    )
                    + 1
                ),
                "bestValidationJointMacroF1": best_f1,
                "calibration": validation_summary,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.artifact} ({args.artifact.stat().st_size} bytes)")
    print(f"Wrote disposable float checkpoint {args.checkpoint}")
    print(json.dumps(validation_summary, indent=2))


if __name__ == "__main__":
    main()
