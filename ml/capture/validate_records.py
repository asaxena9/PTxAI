#!/usr/bin/env python3
"""Validate capture metadata, files, timestamps, and participant split hygiene."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ALLOWED_EXERCISES = {
    "heel-slide",
    "straight-leg-raise",
    "seated-knee-extension",
}
ALLOWED_PHASES = {"ready", "moving", "holding", "returning", "paused"}


def validate_record(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"{path}: cannot read JSON: {error}"]
    prefix = str(path)
    for field in (
        "recordingId",
        "origin",
        "participant",
        "consent",
        "exercise",
        "side",
        "camera",
        "poseProvider",
        "files",
        "annotations",
    ):
        if field not in record:
            errors.append(f"{prefix}: missing {field}")
    if record.get("origin") == "recorded-human":
        consent = record.get("consent", {})
        if consent.get("obtained") is not True:
            errors.append(f"{prefix}: recorded-human data requires consent")
        if not consent.get("protocolId"):
            errors.append(f"{prefix}: consent protocolId is required")
    if record.get("exercise") not in ALLOWED_EXERCISES:
        errors.append(f"{prefix}: unsupported exercise")
    for segment in record.get("annotations", {}).get("phaseSegments", []):
        if segment.get("phase") not in ALLOWED_PHASES:
            errors.append(f"{prefix}: invalid phase {segment.get('phase')}")
        if segment.get("endMs", -1) <= segment.get("startMs", -1):
            errors.append(f"{prefix}: invalid phase interval")

    files = record.get("files", {})
    landmark_name = files.get("landmarks")
    if landmark_name:
        landmark_path = (path.parent / landmark_name).resolve()
        if not landmark_path.is_file():
            errors.append(f"{prefix}: missing landmark file {landmark_path}")
        else:
            digest = hashlib.sha256(landmark_path.read_bytes()).hexdigest()
            if digest != files.get("landmarksSha256"):
                errors.append(f"{prefix}: landmark SHA-256 mismatch")
            try:
                landmarks = json.loads(landmark_path.read_text(encoding="utf-8"))
                timestamps = [
                    float(frame["timestampMs"])
                    for frame in landmarks.get("frames", [])
                ]
                if not timestamps:
                    errors.append(f"{prefix}: landmark sequence is empty")
                elif any(
                    current <= previous
                    for previous, current in zip(timestamps, timestamps[1:])
                ):
                    errors.append(
                        f"{prefix}: landmark timestamps must strictly increase"
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                errors.append(f"{prefix}: malformed landmarks: {error}")
    return record, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    args = parser.parse_args()
    all_errors: list[str] = []
    participant_splits: dict[str, set[str]] = {}
    for path in args.records:
        record, errors = validate_record(path)
        all_errors.extend(errors)
        participant = record.get("participant", {}).get("pseudonymousId")
        split = record.get("split", "unassigned")
        if participant and split != "unassigned":
            participant_splits.setdefault(participant, set()).add(split)
    for participant, splits in participant_splits.items():
        if len(splits) > 1:
            all_errors.append(
                f"participant {participant} leaks across splits: {sorted(splits)}"
            )
    if all_errors:
        print("\n".join(all_errors), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"Validated {len(args.records)} capture record(s); "
        f"{len(participant_splits)} assigned participant(s); no split leakage."
    )


if __name__ == "__main__":
    main()
