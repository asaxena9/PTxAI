#!/usr/bin/env python3
"""Parameter-matched order-sensitive TCN versus order-invariant set follow-up."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = Path(__file__).with_name("run.py")
BASE_RESULT = Path(__file__).parent / "results/2026-08-07/results.json"
OUTPUT = Path(__file__).parent / "results/2026-08-07-order-matched/results.json"

spec = importlib.util.spec_from_file_location("synthetic_temporal_audit_v1", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load base audit")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class DeepSetMeanLast(nn.Module):
    """Nonlinear per-frame encoding; historical order is removed by mean pooling."""

    def __init__(self, feature_count: int, class_count: int, hidden: int = 44):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(feature_count, hidden), nn.ReLU())
        self.head = nn.Linear(hidden * 2, class_count)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        encoded = self.frame(windows)
        return self.head(torch.cat((encoded.mean(dim=1), encoded[:, -1, :]), dim=1))


class DeepSetSpec:
    name = "deepset_mean_last_parameter_matched"

    @staticmethod
    def build() -> nn.Module:
        return DeepSetMeanLast(len(base.FEATURE_NAMES), len(base.JOINT_LABELS))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    previous = json.loads(BASE_RESULT.read_text())
    tcn_rows = {
        row["seed"]: row
        for row in previous["runs"]
        if row["model"] == "causal_tcn"
    }
    if set(tcn_rows) != set(base.SEEDS):
        raise RuntimeError("base result does not contain every declared TCN seed")

    with np.load(base.DATA) as archive:
        train = base.load_split(archive, "train")
        validation = base.load_split(archive, "validation")
        held_out = base.load_split(archive, "held_out")
        train_windows = base.build_windows(train)
        validation_windows = base.build_windows(validation)
        held_out_windows = base.build_windows(held_out)

    mean = train_windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = train_windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    train_windows = (train_windows - mean[None, None, :]) / scale[None, None, :]
    validation_windows = (validation_windows - mean[None, None, :]) / scale[None, None, :]
    held_out_windows = (held_out_windows - mean[None, None, :]) / scale[None, None, :]
    counts = np.bincount(train["joint"], minlength=len(base.JOINT_LABELS))
    weights = np.sqrt(len(train["joint"]) / np.maximum(counts, 1)).astype(np.float32)
    weights /= np.mean(weights)

    rows = []
    for seed in base.SEEDS:
        print(f"training parameter-matched DeepSet seed={seed}", flush=True)
        model, training = base.train_one(
            DeepSetSpec(),
            seed,
            train_windows,
            train["joint"].astype(np.int64),
            validation_windows,
            validation["phase"].astype(np.int64),
            validation["exercise"].astype(np.int64),
            weights,
        )
        logits = base.logits_for(model, held_out_windows)
        prediction = base.conditioned_phase_predictions(
            logits, held_out["exercise"].astype(np.int64)
        )
        metrics = base.phase_metrics(held_out["phase"].astype(np.int64), prediction)
        difference = metrics["macroF1"] - tcn_rows[seed]["heldOut"]["macroF1"]
        rows.append(
            {
                "seed": seed,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "training": training,
                "heldOut": metrics,
                "pairedDifferenceFromTcn": difference,
            }
        )

    values = np.asarray([row["heldOut"]["macroF1"] for row in rows])
    differences = np.asarray([row["pairedDifferenceFromTcn"] for row in rows])
    # Exact two-sided 95% Student-t multiplier for 4 degrees of freedom.
    t_critical = 2.7764451051977987
    half_width = t_critical * float(np.std(differences, ddof=1)) / math.sqrt(len(rows))
    result = {
        "experimentId": "synthetic-order-matched-followup-2026-08-07",
        "evidenceClass": "MEASURED-PILOT",
        "scope": "Existing synthetic generator and disjoint nominal virtual-subject IDs only",
        "question": "Does ordered causal history outperform an order-invariant nonlinear set representation at matched parameter count?",
        "dataSha256": file_sha256(base.DATA),
        "baseResultSha256": file_sha256(BASE_RESULT),
        "seeds": list(base.SEEDS),
        "models": {
            "candidate": "DeepSets mean(encoded frames) plus encoded current frame, 2347 parameters",
            "reference": "causal TCN, 2351 parameters, values loaded from immutable base result",
        },
        "aggregate": {
            "candidatePhaseMacroF1Mean": float(np.mean(values)),
            "candidatePhaseMacroF1Std": float(np.std(values, ddof=1)),
            "pairedDifferenceFromTcnMean": float(np.mean(differences)),
            "pairedDifferenceFromTcnStd": float(np.std(differences, ddof=1)),
            "pairedDifferenceFromTcn95PercentTCI": [
                float(np.mean(differences) - half_width),
                float(np.mean(differences) + half_width),
            ],
        },
        "runs": rows,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Five training seeds do not estimate uncertainty over real subjects or generator families.",
            "The same deterministic generator family produces training and held-out data.",
            "The TCN values are paired to the prior base run rather than retrained in this process.",
            "Order sensitivity in this synthetic task does not establish useful long-context reasoning on people.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
