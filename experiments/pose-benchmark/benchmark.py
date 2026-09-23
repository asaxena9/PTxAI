#!/usr/bin/env python3
"""Benchmark MediaPipe Pose Lite/Full and MoveNet on common rehab footage.

The benchmark intentionally measures observable engineering behavior, not
clinical angle accuracy. MobiPhysio contains no synchronized angle ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "mobiphysio"
MODEL_DIR = ROOT / "models"

MODEL_PATH_BY_NAME = {
    "mediapipe-lite": MODEL_DIR / "pose_landmarker_lite.task",
    "mediapipe-full": MODEL_DIR / "pose_landmarker_full.task",
    "movenet-lightning-f16": (
        MODEL_DIR / "movenet_singlepose_lightning_f16.tflite"
    ),
}

SAMPLES = [
    "E07_P02_AF_VFL_GM.mp4",
    "E07_P02_AR_VFL_GM.mp4",
    "E07_P17_AR_VML_GM.mp4",
]

# The active/more observable chain was selected once by visual review and is
# applied identically to every model. The other leg is often overlapped in the
# side views and would otherwise turn label swaps into a misleading "stability"
# comparison.
PRIMARY_SIDE = {
    "E07_P02_AF_VFL_GM.mp4": "left",
    "E07_P02_AR_VFL_GM.mp4": "left",
    "E07_P17_AR_VML_GM.mp4": "right",
}

SIDE_JOINTS = {
    "left": {
        "mediapipe": (23, 25, 27),
        "movenet": (11, 13, 15),
    },
    "right": {
        "mediapipe": (24, 26, 28),
        "movenet": (12, 14, 16),
    },
}

MOVENET_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def percentile(values: Iterable[float], q: float) -> float | None:
    materialized = np.asarray(list(values), dtype=float)
    materialized = materialized[np.isfinite(materialized)]
    if not len(materialized):
        return None
    return float(np.percentile(materialized, q))


def median(values: Iterable[float]) -> float | None:
    return percentile(values, 50)


def safe_round(value: float | None, places: int = 3) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, places)


def knee_flexion_deg(
    hip: tuple[float, float],
    knee: tuple[float, float],
    ankle: tuple[float, float],
) -> float | None:
    first = np.asarray(hip, dtype=float) - np.asarray(knee, dtype=float)
    second = np.asarray(ankle, dtype=float) - np.asarray(knee, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    interior = math.degrees(math.acos(cosine))
    return 180.0 - interior


@dataclass
class PoseOutput:
    points: list[tuple[float, float]]
    scores: list[float]
    preprocess_ms: float
    inference_ms: float


class MediaPipeRunner:
    family = "mediapipe"

    def __init__(self, name: str, model_path: Path):
        self.name = name
        start = time.perf_counter()
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.load_ms = (time.perf_counter() - start) * 1000.0

    def infer(self, rgb: np.ndarray, timestamp_ms: int) -> PoseOutput | None:
        preprocess_start = time.perf_counter()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        inference_start = time.perf_counter()
        preprocess_ms = (inference_start - preprocess_start) * 1000.0
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        if not result.pose_landmarks:
            return None
        height, width = rgb.shape[:2]
        landmarks = result.pose_landmarks[0]
        points = [(float(item.x * width), float(item.y * height)) for item in landmarks]
        scores = [
            float(min(item.visibility or 0.0, item.presence or 0.0))
            for item in landmarks
        ]
        return PoseOutput(points, scores, preprocess_ms, inference_ms)

    def close(self) -> None:
        self.landmarker.close()


@dataclass
class CropRegion:
    x0: float
    y0: float
    size: float


class MoveNetRunner:
    family = "movenet"
    name = "movenet-lightning-f16"
    input_size = 192

    def __init__(self, model_path: Path):
        start = time.perf_counter()
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            num_threads=4,
        )
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.load_ms = (time.perf_counter() - start) * 1000.0
        self.crop: CropRegion | None = None

    def reset(self, width: int, height: int) -> None:
        size = float(max(width, height))
        self.crop = CropRegion((width - size) / 2.0, (height - size) / 2.0, size)

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        if self.crop is None:
            self.reset(width, height)
        assert self.crop is not None
        scale = self.input_size / self.crop.size
        transform = np.asarray(
            [
                [scale, 0.0, -self.crop.x0 * scale],
                [0.0, scale, -self.crop.y0 * scale],
            ],
            dtype=np.float32,
        )
        resized = cv2.warpAffine(
            rgb,
            transform,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return np.ascontiguousarray(resized[None, ...], dtype=np.uint8)

    def _update_crop(
        self,
        points: list[tuple[float, float]],
        scores: list[float],
        width: int,
        height: int,
    ) -> None:
        torso_indices = (5, 6, 11, 12)
        if min(scores[index] for index in torso_indices) < 0.2:
            self.reset(width, height)
            return

        center_x = (points[11][0] + points[12][0]) / 2.0
        center_y = (points[11][1] + points[12][1]) / 2.0
        confident = [
            point for point, score in zip(points, scores, strict=True) if score >= 0.15
        ]
        if not confident:
            self.reset(width, height)
            return

        half_size = max(
            max(abs(point[0] - center_x), abs(point[1] - center_y))
            for point in confident
        )
        half_size = max(half_size * 1.35, 0.22 * max(width, height))
        half_size = min(half_size, float(max(width, height)))
        self.crop = CropRegion(center_x - half_size, center_y - half_size, 2 * half_size)

    def infer(self, rgb: np.ndarray, _timestamp_ms: int) -> PoseOutput | None:
        height, width = rgb.shape[:2]
        preprocess_start = time.perf_counter()
        tensor = self._preprocess(rgb)
        inference_start = time.perf_counter()
        preprocess_ms = (inference_start - preprocess_start) * 1000.0
        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)[0, 0]
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        assert self.crop is not None
        points = [
            (
                float(self.crop.x0 + item[1] * self.crop.size),
                float(self.crop.y0 + item[0] * self.crop.size),
            )
            for item in output
        ]
        scores = [float(item[2]) for item in output]
        self._update_crop(points, scores, width, height)
        return PoseOutput(points, scores, preprocess_ms, inference_ms)

    def close(self) -> None:
        return


def thumbnail_motion(
    bgr: np.ndarray, previous: np.ndarray | None
) -> tuple[float | None, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(gray, (90, 160), interpolation=cv2.INTER_AREA)
    if previous is None:
        return None, thumbnail
    return float(np.mean(cv2.absdiff(thumbnail, previous)) / 255.0), thumbnail


def rows_for_output(
    *,
    output: PoseOutput | None,
    runner: Any,
    clip: str,
    frame_index: int,
    timestamp_ms: int,
    frame_motion: float | None,
    width: int,
    height: int,
    latency_recorded: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_ms = (
        output.preprocess_ms + output.inference_ms if output is not None else None
    )
    for side in ("left", "right"):
        indices = SIDE_JOINTS[side][runner.family]
        if output is None:
            points = [(None, None)] * 3
            scores = [0.0] * 3
        else:
            points = [output.points[index] for index in indices]
            scores = [output.scores[index] for index in indices]

        valid_points = all(
            x is not None
            and y is not None
            and -0.05 * width <= x <= 1.05 * width
            and -0.05 * height <= y <= 1.05 * height
            for x, y in points
        )
        chain_confidence = min(scores)
        usable = bool(valid_points and chain_confidence >= 0.5)
        angle = (
            knee_flexion_deg(points[0], points[1], points[2]) if usable else None
        )
        thigh_length = (
            math.dist(points[0], points[1]) if usable else None
        )
        shank_length = (
            math.dist(points[1], points[2]) if usable else None
        )
        rows.append(
            {
                "model": runner.name,
                "family": runner.family,
                "clip": clip,
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "side": side,
                "frame_motion": frame_motion,
                "pose_detected": output is not None,
                "chain_confidence": chain_confidence,
                "points_in_frame": valid_points,
                "usable": usable,
                "knee_flexion_deg": angle,
                "thigh_length_px": thigh_length,
                "shank_length_px": shank_length,
                "hip_x": points[0][0],
                "hip_y": points[0][1],
                "knee_x": points[1][0],
                "knee_y": points[1][1],
                "ankle_x": points[2][0],
                "ankle_y": points[2][1],
                "preprocess_ms": output.preprocess_ms if output else None,
                "inference_ms": output.inference_ms if output else None,
                "total_pose_ms": total_ms,
                "latency_recorded": latency_recorded,
            }
        )
    return rows


def process_clip(
    runner: Any,
    clip_path: Path,
    sample_hz: float,
    warmup_frames: int,
    input_rotation_deg: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(clip_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or source_frames <= 0:
        raise RuntimeError(f"could not read video metadata: {clip_path}")

    stride = max(1, round(fps / sample_hz))
    rows: list[dict[str, Any]] = []
    frame_index = 0
    sampled_index = 0
    previous_thumbnail: np.ndarray | None = None
    inference_width = width
    inference_height = height
    runner_dimensions_initialized = False
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue

        motion, previous_thumbnail = thumbnail_motion(bgr, previous_thumbnail)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if input_rotation_deg == 90:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        elif input_rotation_deg == -90:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        inference_height, inference_width = rgb.shape[:2]
        if isinstance(runner, MoveNetRunner) and not runner_dimensions_initialized:
            runner.reset(inference_width, inference_height)
            runner_dimensions_initialized = True
        timestamp_ms = round(frame_index / fps * 1000.0)
        output = runner.infer(rgb, timestamp_ms)
        rows.extend(
            rows_for_output(
                output=output,
                runner=runner,
                clip=clip_path.name,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                frame_motion=motion,
                width=inference_width,
                height=inference_height,
                latency_recorded=sampled_index >= warmup_frames,
            )
        )
        sampled_index += 1
        frame_index += 1

    capture.release()
    metadata = {
        "clip": clip_path.name,
        "source_fps": fps,
        "source_frames": source_frames,
        "duration_sec": source_frames / fps,
        "width": width,
        "height": height,
        "inference_width": inference_width,
        "inference_height": inference_height,
        "input_rotation_deg": input_rotation_deg,
        "stride": stride,
        "sampled_frames": sampled_index,
        "effective_sample_hz": fps / stride,
    }
    return rows, metadata


def consecutive_pairs(rows: list[dict[str, Any]]) -> Iterable[tuple[dict, dict]]:
    ordered = sorted(rows, key=lambda item: item["timestamp_ms"])
    for previous, current in zip(ordered, ordered[1:]):
        if (
            previous["usable"]
            and current["usable"]
            and current["frame_index"] > previous["frame_index"]
        ):
            yield previous, current


def summarize_model(
    model: str,
    rows: list[dict[str, Any]],
    load_times: list[float],
) -> dict[str, Any]:
    subset = [row for row in rows if row["model"] == model]
    frame_keys = {(row["clip"], row["frame_index"]) for row in subset}
    detected_keys = {
        (row["clip"], row["frame_index"])
        for row in subset
        if row["pose_detected"]
    }
    any_leg_keys = {
        (row["clip"], row["frame_index"]) for row in subset if row["usable"]
    }
    usable = [row for row in subset if row["usable"]]
    primary = [
        row for row in subset if row["side"] == PRIMARY_SIDE[row["clip"]]
    ]
    usable_primary = [row for row in primary if row["usable"]]
    latency = [
        row
        for row in subset
        if row["latency_recorded"] and row["side"] == "left"
    ]

    static_steps: list[float] = []
    all_steps: list[float] = []
    second_differences: list[float] = []
    active_clips = sorted({row["clip"] for row in subset})
    for clip in active_clips:
        clip_motion = [
            row["frame_motion"]
            for row in subset
            if row["clip"] == clip
            and row["side"] == "left"
            and row["frame_motion"] is not None
        ]
        low_motion_threshold = percentile(clip_motion, 25)
        side = PRIMARY_SIDE[clip]
        sequence = [
            row
            for row in subset
            if row["clip"] == clip and row["side"] == side
        ]
        pairs = list(consecutive_pairs(sequence))
        for previous, current in pairs:
            delta = abs(
                current["knee_flexion_deg"] - previous["knee_flexion_deg"]
            )
            all_steps.append(delta)
            if (
                low_motion_threshold is not None
                and current["frame_motion"] is not None
                and current["frame_motion"] <= low_motion_threshold
            ):
                static_steps.append(delta)
        usable_sequence = [row for row in sequence if row["usable"]]
        for first, middle, last in zip(
            usable_sequence, usable_sequence[1:], usable_sequence[2:]
        ):
            if (
                middle["frame_index"] - first["frame_index"]
                == last["frame_index"] - middle["frame_index"]
            ):
                second_differences.append(
                    abs(
                        last["knee_flexion_deg"]
                        - 2 * middle["knee_flexion_deg"]
                        + first["knee_flexion_deg"]
                    )
                )

    def coefficient_of_variation(field: str) -> float | None:
        values = np.asarray([row[field] for row in usable_primary], dtype=float)
        if not len(values) or float(np.mean(values)) == 0:
            return None
        return float(np.std(values) / np.mean(values) * 100.0)

    total_pose_ms = [row["total_pose_ms"] for row in latency]
    inference_ms = [row["inference_ms"] for row in latency]
    preprocess_ms = [row["preprocess_ms"] for row in latency]
    total_seconds = sum(total_pose_ms) / 1000.0
    confidence_coverage = {
        f"{threshold:.1f}": safe_round(
            100.0
            * sum(
                row["points_in_frame"]
                and row["chain_confidence"] >= threshold
                for row in primary
            )
            / len(primary)
        )
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    }
    return {
        "model": model,
        "model_asset_bytes": MODEL_PATH_BY_NAME[model].stat().st_size,
        "cold_load_ms_median": safe_round(median(load_times)),
        "cold_load_ms_all": [safe_round(item) for item in load_times],
        "evaluated_frames": len(frame_keys),
        "pose_detection_coverage_pct": safe_round(
            100.0 * len(detected_keys) / len(frame_keys)
        ),
        "any_leg_chain_coverage_pct": safe_round(
            100.0 * len(any_leg_keys) / len(frame_keys)
        ),
        "all_leg_chain_coverage_pct": safe_round(
            100.0 * len(usable) / len(subset)
        ),
        "primary_leg_chain_coverage_pct": safe_round(
            100.0 * len(usable_primary) / len(primary)
        ),
        "primary_median_chain_confidence": safe_round(
            median(row["chain_confidence"] for row in primary)
        ),
        "primary_chain_coverage_by_raw_confidence_pct": confidence_coverage,
        "total_pose_ms_p50": safe_round(median(total_pose_ms)),
        "total_pose_ms_p95": safe_round(percentile(total_pose_ms, 95)),
        "inference_ms_p50": safe_round(median(inference_ms)),
        "inference_ms_p95": safe_round(percentile(inference_ms, 95)),
        "preprocess_ms_p50": safe_round(median(preprocess_ms)),
        "sequential_throughput_fps": safe_round(
            len(total_pose_ms) / total_seconds if total_seconds else None
        ),
        "low_motion_angle_step_deg_p50": safe_round(median(static_steps)),
        "low_motion_angle_step_deg_p95": safe_round(
            percentile(static_steps, 95)
        ),
        "all_motion_angle_step_deg_p50": safe_round(median(all_steps)),
        "angle_second_difference_deg_p50": safe_round(
            median(second_differences)
        ),
        "thigh_length_cv_pct": safe_round(
            coefficient_of_variation("thigh_length_px")
        ),
        "shank_length_cv_pct": safe_round(
            coefficient_of_variation("shank_length_px")
        ),
        "knee_angle_p05": safe_round(
            percentile(
                (row["knee_flexion_deg"] for row in usable_primary),
                5,
            )
        ),
        "knee_angle_p95": safe_round(
            percentile(
                (row["knee_flexion_deg"] for row in usable_primary),
                95,
            )
        ),
        "per_clip_primary_chain": [
            summarize_clip_primary(model, clip, subset) for clip in active_clips
        ],
    }


def summarize_clip_primary(
    model: str,
    clip: str,
    model_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    side = PRIMARY_SIDE[clip]
    subset = [
        row
        for row in model_rows
        if row["clip"] == clip and row["side"] == side
    ]
    usable = [row for row in subset if row["usable"]]
    angles = [row["knee_flexion_deg"] for row in usable]
    return {
        "model": model,
        "clip": clip,
        "primary_side": side,
        "coverage_pct": safe_round(100.0 * len(usable) / len(subset)),
        "median_chain_confidence": safe_round(
            median(row["chain_confidence"] for row in subset)
        ),
        "angle_p05": safe_round(percentile(angles, 5)),
        "angle_p50": safe_round(percentile(angles, 50)),
        "angle_p95": safe_round(percentile(angles, 95)),
    }


def pairwise_agreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["side"] != PRIMARY_SIDE[row["clip"]]:
            continue
        key = (row["clip"], row["frame_index"], row["side"])
        keyed.setdefault(key, {})[row["model"]] = row
    models = sorted({row["model"] for row in rows})
    results = []
    for index, first in enumerate(models):
        for second in models[index + 1 :]:
            differences = []
            signed = []
            for observations in keyed.values():
                if first not in observations or second not in observations:
                    continue
                first_row, second_row = observations[first], observations[second]
                if not first_row["usable"] or not second_row["usable"]:
                    continue
                difference = (
                    first_row["knee_flexion_deg"]
                    - second_row["knee_flexion_deg"]
                )
                differences.append(abs(difference))
                signed.append(difference)
            results.append(
                {
                    "first": first,
                    "second": second,
                    "common_angle_samples": len(differences),
                    "median_absolute_difference_deg": safe_round(
                        median(differences)
                    ),
                    "p95_absolute_difference_deg": safe_round(
                        percentile(differences, 95)
                    ),
                    "median_signed_first_minus_second_deg": safe_round(
                        median(signed)
                    ),
                }
            )
    return results


def factory_for(model_name: str) -> Any:
    if model_name == "mediapipe-lite":
        return MediaPipeRunner(
            "mediapipe-lite", MODEL_DIR / "pose_landmarker_lite.task"
        )
    if model_name == "mediapipe-full":
        return MediaPipeRunner(
            "mediapipe-full", MODEL_DIR / "pose_landmarker_full.task"
        )
    if model_name == "movenet-lightning-f16":
        return MoveNetRunner(
            MODEL_DIR / "movenet_singlepose_lightning_f16.tflite"
        )
    raise ValueError(model_name)


def cold_load_probes(model_name: str, repetitions: int) -> list[float]:
    values = []
    for _ in range(repetitions):
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--probe-model-load",
                model_name,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        values.append(float(json.loads(completed.stdout.strip())["load_ms"]))
    return values


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--cold-load-probes", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "mediapipe-lite",
            "mediapipe-full",
            "movenet-lightning-f16",
        ],
        default=[
            "mediapipe-lite",
            "mediapipe-full",
            "movenet-lightning-f16",
        ],
        help="Run only the selected pose pipelines. This is useful for "
        "bounded single-model sampling experiments.",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        choices=SAMPLES,
        default=SAMPLES,
    )
    parser.add_argument(
        "--input-rotation-deg",
        type=int,
        choices=(-90, 0, 90),
        default=0,
        help="Rotate decoded frames before pose inference; -90 makes the "
        "head-up orientation for the selected side-view proxy clips.",
    )
    parser.add_argument(
        "--probe-model-load",
        choices=[
            "mediapipe-lite",
            "mediapipe-full",
            "movenet-lightning-f16",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "latest",
    )
    args = parser.parse_args()

    if args.probe_model_load:
        runner = factory_for(args.probe_model_load)
        print(json.dumps({"model": runner.name, "load_ms": runner.load_ms}))
        runner.close()
        return 0

    selected_samples = args.clips
    missing = [
        str(DATA_DIR / sample)
        for sample in selected_samples
        if not (DATA_DIR / sample).exists()
    ]
    if missing:
        raise SystemExit(
            "Missing benchmark samples. Run download_assets.py first:\n"
            + "\n".join(missing)
        )

    model_names = args.models

    rows: list[dict[str, Any]] = []
    clip_metadata: dict[str, Any] = {}
    load_times = {
        name: cold_load_probes(name, args.cold_load_probes)
        for name in model_names
    }
    in_process_create_times: dict[str, list[float]] = {
        name: [] for name in model_names
    }

    for model_name in model_names:
        for sample in selected_samples:
            print(f"running {model_name} on {sample}", flush=True)
            runner = factory_for(model_name)
            in_process_create_times[model_name].append(runner.load_ms)
            clip_rows, metadata = process_clip(
                runner,
                DATA_DIR / sample,
                sample_hz=args.sample_hz,
                warmup_frames=args.warmup_frames,
                input_rotation_deg=args.input_rotation_deg,
            )
            runner.close()
            rows.extend(clip_rows)
            clip_metadata[sample] = metadata

    summaries = [
        {
            **summarize_model(model_name, rows, load_times[model_name]),
            "in_process_runner_create_ms_all": [
                safe_round(item)
                for item in in_process_create_times[model_name]
            ],
        }
        for model_name in model_names
    ]
    report = {
        "methodology": {
            "sample_hz_requested": args.sample_hz,
            "warmup_frames_excluded_from_latency": args.warmup_frames,
            "input_rotation_deg": args.input_rotation_deg,
            "selected_clips": selected_samples,
            "chain_confidence_threshold": 0.5,
            "primary_analysis_side": PRIMARY_SIDE,
            "static_transition_definition": (
                "bottom quartile of model-independent thumbnail frame change "
                "within each clip"
            ),
            "angle_definition": (
                "180 degrees minus the aspect-correct image-space "
                "hip-knee-ankle interior angle"
            ),
            "movenet_preprocessing": (
                "192x192 square crop; first frame is full-frame padded, then "
                "a confidence-gated hip-centered crop tracks the subject"
            ),
            "warning": (
                "No synchronized anatomical angle ground truth is available. "
                "Angle stability and pairwise agreement are not accuracy."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "opencv": cv2.__version__,
            "mediapipe": mp.__version__,
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
        "clips": clip_metadata,
        "models": summaries,
        "pairwise_angle_agreement": pairwise_agreement(rows),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "frame_metrics.csv", rows)
    with (args.output / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
