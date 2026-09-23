#!/usr/bin/env python3
"""Audit which temporal information drives the existing synthetic result."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from relay_temporal.constants import FEATURE_NAMES, JOINT_LABELS, PHASES  # noqa: E402
from relay_temporal.dataset import build_windows, load_split  # noqa: E402
from relay_temporal.model import RelayTemporalNet  # noqa: E402


DATA = ROOT / "ml/.generated/relay-temporal-synthetic-v1.npz"
OUTPUT = ROOT / "experiments/synthetic-temporal-audit/results/2026-08-07"
SEEDS = (1103, 2207, 3301, 4409, 5519)
MAX_EPOCHS = 35
PATIENCE = 6
BATCH_SIZE = 256
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class LastMLP(nn.Module):
    def __init__(self, input_width: int, class_count: int, hidden: int = 32):
        super().__init__()
        self.input_width = input_width
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.network(windows[:, -1, : self.input_width])


class MeanLastMLP(nn.Module):
    def __init__(self, feature_count: int, class_count: int, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        representation = torch.cat(
            (windows.mean(dim=1), windows[:, -1, :]), dim=1
        )
        return self.network(representation)


@dataclass(frozen=True)
class ModelSpec:
    name: str

    def build(self) -> nn.Module:
        if self.name == "last_mlp_no_derivatives":
            return LastMLP(18, len(JOINT_LABELS))
        if self.name == "last_mlp_full":
            return LastMLP(len(FEATURE_NAMES), len(JOINT_LABELS))
        if self.name == "mean_last_mlp":
            return MeanLastMLP(len(FEATURE_NAMES), len(JOINT_LABELS))
        if self.name == "causal_tcn":
            return RelayTemporalNet(len(FEATURE_NAMES), len(JOINT_LABELS))
        raise ValueError(self.name)


SPECS = tuple(
    ModelSpec(name)
    for name in (
        "last_mlp_no_derivatives",
        "last_mlp_full",
        "mean_last_mlp",
        "causal_tcn",
    )
)


def conditioned_phase_predictions(
    logits: np.ndarray, exercise: np.ndarray
) -> np.ndarray:
    cube = logits.reshape(len(logits), -1, len(PHASES))
    return np.argmax(cube[np.arange(len(logits)), exercise], axis=1)


def phase_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    per_class = f1_score(
        target,
        prediction,
        labels=np.arange(len(PHASES)),
        average=None,
        zero_division=0,
    )
    return {
        "macroF1": float(np.mean(per_class)),
        "accuracy": float(np.mean(target == prediction)),
        "perClassF1": {
            phase: float(per_class[index]) for index, phase in enumerate(PHASES)
        },
        "confusion": confusion_matrix(
            target, prediction, labels=np.arange(len(PHASES))
        ).tolist(),
    }


@torch.inference_mode()
def logits_for(model: nn.Module, windows: np.ndarray) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(windows), 2048):
        outputs.append(model(torch.from_numpy(windows[start : start + 2048])).numpy())
    return np.concatenate(outputs)


def train_one(
    spec: ModelSpec,
    seed: int,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    validation_windows: np.ndarray,
    validation_phase: np.ndarray,
    validation_exercise: np.ndarray,
    class_weights: np.ndarray,
) -> tuple[nn.Module, dict]:
    seed_everything(seed)
    model = spec.build()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows), torch.from_numpy(train_labels)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_score = -math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for windows, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(windows), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_logits = logits_for(model, validation_windows)
        validation_prediction = conditioned_phase_predictions(
            validation_logits, validation_exercise
        )
        score = float(
            f1_score(
                validation_phase,
                validation_prediction,
                labels=np.arange(len(PHASES)),
                average="macro",
                zero_division=0,
            )
        )
        history.append(
            {"epoch": epoch, "trainingLoss": float(np.mean(losses)), "validationPhaseMacroF1": score}
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {
        "bestEpoch": best_epoch,
        "bestValidationPhaseMacroF1": best_score,
        "epochsCompleted": len(history),
        "history": history,
    }


def aggregate(rows: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    tcn_by_seed = {
        row["seed"]: row["heldOut"]["macroF1"] for row in by_model["causal_tcn"]
    }
    result = {}
    for model, model_rows in by_model.items():
        values = np.asarray(
            [row["heldOut"]["macroF1"] for row in model_rows], dtype=np.float64
        )
        differences = np.asarray(
            [row["heldOut"]["macroF1"] - tcn_by_seed[row["seed"]] for row in model_rows],
            dtype=np.float64,
        )
        result[model] = {
            "parameters": model_rows[0]["parameters"],
            "heldOutPhaseMacroF1Mean": float(np.mean(values)),
            "heldOutPhaseMacroF1Std": float(np.std(values, ddof=1)),
            "pairedDifferenceFromTcnMean": float(np.mean(differences)),
            "pairedDifferenceFromTcnStd": float(np.std(differences, ddof=1)),
            "seeds": len(values),
        }
    return result


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    with np.load(DATA) as archive:
        train = load_split(archive, "train")
        validation = load_split(archive, "validation")
        held_out = load_split(archive, "held_out")
        train_windows = build_windows(train)
        validation_windows = build_windows(validation)
        held_out_windows = build_windows(held_out)

    mean = train_windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = train_windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    train_windows = (train_windows - mean[None, None, :]) / scale[None, None, :]
    validation_windows = (
        validation_windows - mean[None, None, :]
    ) / scale[None, None, :]
    held_out_windows = (held_out_windows - mean[None, None, :]) / scale[None, None, :]

    counts = np.bincount(train["joint"], minlength=len(JOINT_LABELS))
    weights = np.sqrt(len(train["joint"]) / np.maximum(counts, 1)).astype(np.float32)
    weights /= np.mean(weights)

    rows = []
    for spec in SPECS:
        for seed in SEEDS:
            print(f"training model={spec.name} seed={seed}", flush=True)
            model, training = train_one(
                spec,
                seed,
                train_windows,
                train["joint"].astype(np.int64),
                validation_windows,
                validation["phase"].astype(np.int64),
                validation["exercise"].astype(np.int64),
                weights,
            )
            held_logits = logits_for(model, held_out_windows)
            held_prediction = conditioned_phase_predictions(
                held_logits, held_out["exercise"].astype(np.int64)
            )
            rows.append(
                {
                    "model": spec.name,
                    "seed": seed,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "training": training,
                    "heldOut": phase_metrics(
                        held_out["phase"].astype(np.int64), held_prediction
                    ),
                }
            )

    result = {
        "experimentId": "synthetic-temporal-information-audit-2026-08-07",
        "evidenceClass": "MEASURED-PILOT",
        "scope": "Existing synthetic generator and disjoint virtual subjects only",
        "containsRealPeople": False,
        "primaryEstimand": "paired held-out phase macro-F1 difference from causal TCN, exercise conditioned",
        "data": {"path": str(DATA), "sha256": sha256(DATA)},
        "seeds": list(SEEDS),
        "optimization": {
            "loss": "sqrt-inverse-frequency-weighted joint exercise-phase cross entropy",
            "learningRate": LEARNING_RATE,
            "weightDecay": WEIGHT_DECAY,
            "batchSize": BATCH_SIZE,
            "maximumEpochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "selectionMetric": "validation phase macro-F1 conditioned on known exercise",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "aggregate": aggregate(rows),
        "runs": rows,
        "limitations": [
            "Synthetic generator labels are deterministic and may contain shortcuts.",
            "This is not evidence of human, camera, clinical, or deployment behavior.",
            "Models are not exactly parameter or compute matched; counts are reported.",
            "Known-exercise conditioning reflects the current product prescription flow.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
