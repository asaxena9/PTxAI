#!/usr/bin/env python3
"""Render high-disagreement frames for qualitative inspection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from benchmark import DATA_DIR, PRIMARY_SIDE, ROOT


COLORS = {
    "mediapipe-full": (255, 220, 20),
    "mediapipe-lite": (20, 220, 255),
    "movenet-lightning-f16": (255, 40, 210),
}


def boolean(value: str) -> bool:
    return value.lower() == "true"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "latest",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader((args.results / "frame_metrics.csv").open()))
    keyed: dict[tuple[str, int, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        row["frame_index"] = int(row["frame_index"])
        row["usable"] = boolean(row["usable"])
        for field in (
            "chain_confidence",
            "knee_flexion_deg",
            "hip_x",
            "hip_y",
            "knee_x",
            "knee_y",
            "ankle_x",
            "ankle_y",
        ):
            row[field] = float(row[field]) if row[field] else None
        keyed[(row["clip"], row["frame_index"], row["side"])][row["model"]] = row

    chosen = []
    for clip, side in PRIMARY_SIDE.items():
        candidates = []
        for (candidate_clip, frame_index, candidate_side), models in keyed.items():
            if candidate_clip != clip or candidate_side != side:
                continue
            full = models.get("mediapipe-full")
            lite = models.get("mediapipe-lite")
            if not full or not lite or not full["usable"] or not lite["usable"]:
                continue
            difference = abs(full["knee_flexion_deg"] - lite["knee_flexion_deg"])
            movenet = models.get("movenet-lightning-f16")
            all_three = bool(movenet and movenet["usable"])
            candidates.append((all_three, difference, frame_index, models))
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        if candidates:
            chosen.append((clip, side, *candidates[0][1:]))

    panels = []
    diagnostics = []
    for clip, side, difference, frame_index, models in chosen:
        capture = cv2.VideoCapture(str(DATA_DIR / clip))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            continue

        scale = min(640 / frame.shape[0], 520 / frame.shape[1])
        frame = cv2.resize(
            frame,
            (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

        y = 28
        cv2.putText(
            frame,
            f"{clip} frame {frame_index} ({side})",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
        model_values = {}
        for model in (
            "mediapipe-full",
            "mediapipe-lite",
            "movenet-lightning-f16",
        ):
            row = models.get(model)
            if not row:
                continue
            color = COLORS[model]
            points = [
                (round(row["hip_x"] * scale), round(row["hip_y"] * scale)),
                (round(row["knee_x"] * scale), round(row["knee_y"] * scale)),
                (round(row["ankle_x"] * scale), round(row["ankle_y"] * scale)),
            ]
            if row["usable"]:
                cv2.polylines(
                    frame,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    color,
                    3,
                )
                for point in points:
                    cv2.circle(frame, point, 5, color, -1)
            angle = (
                f"{row['knee_flexion_deg']:.1f}deg"
                if row["knee_flexion_deg"] is not None
                else "unavailable"
            )
            cv2.putText(
                frame,
                f"{model}: {angle}, q={row['chain_confidence']:.2f}",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 21
            model_values[model] = {
                "usable": row["usable"],
                "angle_deg": row["knee_flexion_deg"],
                "chain_confidence": row["chain_confidence"],
            }
        panels.append(frame)
        diagnostics.append(
            {
                "clip": clip,
                "side": side,
                "frame_index": frame_index,
                "full_lite_difference_deg": difference,
                "models": model_values,
            }
        )

    if panels:
        target_height = max(panel.shape[0] for panel in panels)
        normalized = []
        for panel in panels:
            if panel.shape[0] < target_height:
                panel = cv2.copyMakeBorder(
                    panel,
                    0,
                    target_height - panel.shape[0],
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            normalized.append(panel)
        output = cv2.hconcat(normalized)
        cv2.imwrite(str(args.results / "disagreement_examples.jpg"), output)
    with (args.results / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
