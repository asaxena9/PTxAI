#!/usr/bin/env python3
"""Measure loss from independently sampled pose streams.

This script intentionally treats the highest-rate pose output as a proxy
reference. It does not claim biomechanical ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PRIMARY_SIDE = {
    "E07_P02_AF_VFL_GM.mp4": "left",
    "E07_P02_AR_VFL_GM.mp4": "left",
    "E07_P17_AR_VML_GM.mp4": "right",
}
MODEL = "mediapipe-full"
BOUNDARY_P95_THRESHOLD_MS = 100.0
MATCH_TOLERANCE_MS = 350.0
SMOOTHING_WINDOW_MS = 200.0
MIN_BOUNDARY_GAP_MS = 400.0


@dataclass(frozen=True)
class Series:
    timestamp_ms: np.ndarray
    angle_deg: np.ndarray


@dataclass(frozen=True)
class Boundary:
    timestamp_ms: float
    direction: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_series(value: str) -> tuple[float, Path]:
    try:
        raw_hz, raw_path = value.split("=", 1)
        return float(raw_hz), Path(raw_path).resolve()
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected RATE=CSV_PATH") from error


def percentile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return float(np.percentile(finite, q))


def rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def load_csv(path: Path) -> dict[str, Series]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip = row["clip"]
            if (
                row["model"] != MODEL
                or clip not in PRIMARY_SIDE
                or row["side"] != PRIMARY_SIDE[clip]
                or row["usable"].lower() != "true"
                or not row["knee_flexion_deg"]
            ):
                continue
            grouped.setdefault(clip, []).append(
                (float(row["timestamp_ms"]), float(row["knee_flexion_deg"]))
            )
    result: dict[str, Series] = {}
    for clip, pairs in grouped.items():
        pairs.sort()
        timestamps = np.asarray([item[0] for item in pairs], dtype=np.float64)
        angles = np.asarray([item[1] for item in pairs], dtype=np.float64)
        unique = np.concatenate(([True], np.diff(timestamps) > 0))
        result[clip] = Series(timestamps[unique], angles[unique])
    return result


def smooth(series: Series, window_ms: float) -> Series:
    if len(series.timestamp_ms) < 3:
        return series
    median_step = float(np.median(np.diff(series.timestamp_ms)))
    width = max(1, int(round(window_ms / median_step)))
    if width % 2 == 0:
        width += 1
    width = min(width, len(series.angle_deg) if len(series.angle_deg) % 2 else len(series.angle_deg) - 1)
    if width <= 1:
        return series
    pad = width // 2
    padded = np.pad(series.angle_deg, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return Series(series.timestamp_ms, np.convolve(padded, kernel, mode="valid"))


def boundaries(series: Series, threshold: float) -> list[Boundary]:
    events: list[Boundary] = []
    last_timestamp = -math.inf
    for index in range(1, len(series.timestamp_ms)):
        before = float(series.angle_deg[index - 1] - threshold)
        after = float(series.angle_deg[index] - threshold)
        if before == after or before * after > 0:
            continue
        direction = "up" if after > before else "down"
        denominator = abs(before) + abs(after)
        fraction = 0.5 if denominator == 0 else abs(before) / denominator
        timestamp = float(
            series.timestamp_ms[index - 1]
            + fraction
            * (series.timestamp_ms[index] - series.timestamp_ms[index - 1])
        )
        if timestamp - last_timestamp < MIN_BOUNDARY_GAP_MS:
            continue
        events.append(Boundary(timestamp, direction))
        last_timestamp = timestamp
    return events


def match_boundaries(
    reference: list[Boundary], candidate: list[Boundary]
) -> tuple[list[dict[str, float | str]], int, int]:
    used: set[int] = set()
    matches: list[dict[str, float | str]] = []
    for ref in reference:
        eligible = [
            (abs(item.timestamp_ms - ref.timestamp_ms), index, item)
            for index, item in enumerate(candidate)
            if index not in used
            and item.direction == ref.direction
            and abs(item.timestamp_ms - ref.timestamp_ms) <= MATCH_TOLERANCE_MS
        ]
        if not eligible:
            continue
        error, index, item = min(eligible, key=lambda entry: entry[0])
        used.add(index)
        matches.append(
            {
                "direction": ref.direction,
                "referenceTimestampMs": rounded(ref.timestamp_ms),
                "candidateTimestampMs": rounded(item.timestamp_ms),
                "absoluteErrorMs": rounded(error),
            }
        )
    return matches, len(reference) - len(matches), len(candidate) - len(matches)


def compare_clip(reference: Series, candidate: Series) -> tuple[dict, list[dict]]:
    reference_smoothed = smooth(reference, SMOOTHING_WINDOW_MS)
    candidate_smoothed = smooth(candidate, SMOOTHING_WINDOW_MS)
    low = percentile(reference_smoothed.angle_deg, 20)
    high = percentile(reference_smoothed.angle_deg, 80)
    assert low is not None and high is not None
    threshold = (low + high) / 2.0

    overlap = (
        (reference_smoothed.timestamp_ms >= candidate_smoothed.timestamp_ms[0])
        & (reference_smoothed.timestamp_ms <= candidate_smoothed.timestamp_ms[-1])
    )
    ref_time = reference_smoothed.timestamp_ms[overlap]
    ref_angle = reference_smoothed.angle_deg[overlap]
    interpolated_angle = np.interp(
        ref_time, candidate_smoothed.timestamp_ms, candidate_smoothed.angle_deg
    )
    angle_error = np.abs(ref_angle - interpolated_angle)

    ref_velocity = np.gradient(ref_angle, ref_time / 1000.0)
    candidate_velocity = np.gradient(
        candidate_smoothed.angle_deg,
        candidate_smoothed.timestamp_ms / 1000.0,
    )
    interpolated_velocity = np.interp(
        ref_time, candidate_smoothed.timestamp_ms, candidate_velocity
    )
    velocity_error = np.abs(ref_velocity - interpolated_velocity)

    reference_events = boundaries(reference_smoothed, threshold)
    candidate_events = boundaries(candidate_smoothed, threshold)
    matches, missed, extra = match_boundaries(reference_events, candidate_events)
    event_errors = np.asarray(
        [float(item["absoluteErrorMs"]) for item in matches], dtype=np.float64
    )
    intervals = np.diff(candidate.timestamp_ms)
    metrics = {
        "referenceSamples": int(len(reference.timestamp_ms)),
        "candidateSamples": int(len(candidate.timestamp_ms)),
        "candidateIntervalMsP50": rounded(percentile(intervals, 50)),
        "candidateIntervalMsP95": rounded(percentile(intervals, 95)),
        "midrangeThresholdDeg": rounded(threshold),
        "referenceBoundaries": len(reference_events),
        "candidateBoundaries": len(candidate_events),
        "matchedBoundaries": len(matches),
        "missedReferenceBoundaries": missed,
        "extraCandidateBoundaries": extra,
        "boundaryAbsoluteErrorMsP50": rounded(percentile(event_errors, 50)),
        "boundaryAbsoluteErrorMsP95": rounded(percentile(event_errors, 95)),
        "angleReconstructionMaeDeg": rounded(float(np.mean(angle_error))),
        "angleReconstructionErrorDegP95": rounded(percentile(angle_error, 95)),
        "angularVelocityErrorDegPerSecMae": rounded(float(np.mean(velocity_error))),
        "angularVelocityErrorDegPerSecP95": rounded(percentile(velocity_error, 95)),
    }
    return metrics, matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", type=parse_series, required=True)
    parser.add_argument("--reference-hz", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.series)
    if args.reference_hz not in paths:
        raise SystemExit("reference rate must be included in --series")
    if len(paths) != len(args.series):
        raise SystemExit("duplicate --series rate")
    for path in paths.values():
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    loaded = {rate: load_csv(path) for rate, path in paths.items()}
    reference = loaded[args.reference_hz]
    expected_clips = set(PRIMARY_SIDE)
    for rate, clips in loaded.items():
        if set(clips) != expected_clips:
            raise SystemExit(
                f"rate {rate:g} has clips {sorted(clips)}, expected {sorted(expected_clips)}"
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_clip: list[dict] = []
    all_matches: list[dict] = []
    dispositions: dict[str, dict] = {}

    for rate in sorted(loaded):
        if rate == args.reference_hz:
            continue
        rate_rows: list[dict] = []
        for clip in sorted(expected_clips):
            metrics, matches = compare_clip(reference[clip], loaded[rate][clip])
            row = {"rateHz": rate, "clip": clip, **metrics}
            per_clip.append(row)
            rate_rows.append(row)
            all_matches.extend(
                {"rateHz": rate, "clip": clip, **item} for item in matches
            )

        errors = np.asarray(
            [
                float(item["absoluteErrorMs"])
                for item in all_matches
                if item["rateHz"] == rate
            ],
            dtype=np.float64,
        )
        missed = sum(int(item["missedReferenceBoundaries"]) for item in rate_rows)
        extra = sum(int(item["extraCandidateBoundaries"]) for item in rate_rows)
        boundary_p95 = percentile(errors, 95)
        passes = (
            boundary_p95 is not None
            and boundary_p95 <= BOUNDARY_P95_THRESHOLD_MS
            and missed == 0
            and extra == 0
        )
        dispositions[f"{rate:g}"] = {
            "rateHz": rate,
            "boundaryAbsoluteErrorMsP95": rounded(boundary_p95),
            "missedReferenceBoundaries": missed,
            "extraCandidateBoundaries": extra,
            "predeclaredThresholdMs": BOUNDARY_P95_THRESHOLD_MS,
            "disposition": "keep-hypothesis-for-replication" if passes else "reject-hypothesis",
        }

    manifest = {
        "experimentId": "temporal-sampling-audit-2026-08-07-pilot",
        "evidenceClass": "MEASURED-PILOT",
        "claim": (
            "On three slow MobiPhysio E07 clips, independent 10/15 Hz "
            "MediaPipe Full streams preserve mid-range crossings relative "
            "to an independent 30 Hz stream with p95 <= 100 ms and no "
            "unmatched boundaries."
        ),
        "nullHypothesis": "Any tested rate violates a predeclared boundary criterion.",
        "primaryMetric": {
            "name": "boundaryAbsoluteErrorMsP95",
            "direction": "min",
            "threshold": BOUNDARY_P95_THRESHOLD_MS,
        },
        "guardrails": {
            "missedReferenceBoundaries": 0,
            "extraCandidateBoundaries": 0,
        },
        "method": {
            "model": MODEL,
            "referenceHz": args.reference_hz,
            "smoothingWindowMs": SMOOTHING_WINDOW_MS,
            "minimumBoundaryGapMs": MIN_BOUNDARY_GAP_MS,
            "matchingToleranceMs": MATCH_TOLERANCE_MS,
            "threshold": "midpoint of reference p20 and p80 per clip",
            "warning": (
                "The reference is 30 Hz model output, not biomechanical ground truth. "
                "Centered smoothing is non-causal."
            ),
        },
        "inputs": [
            {"rateHz": rate, "path": str(path), "sha256": sha256(path)}
            for rate, path in sorted(paths.items())
        ],
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "metrics.json").write_text(
        json.dumps({"dispositions": dispositions, "perClip": per_clip}, indent=2) + "\n"
    )
    with (output / "boundary_matches.csv").open("w", newline="") as handle:
        fieldnames = [
            "rateHz",
            "clip",
            "direction",
            "referenceTimestampMs",
            "candidateTimestampMs",
            "absoluteErrorMs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_matches)
    print(json.dumps(dispositions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
