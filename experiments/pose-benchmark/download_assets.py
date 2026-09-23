#!/usr/bin/env python3
"""Download the exact public samples and pose models used by the benchmark."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SAMPLES = [
    {
        "filename": "E07_P02_AF_VFL_GM.mp4",
        "url": "https://dataverse.harvard.edu/api/access/datafile/11718308",
        "sha256": "3a69a3f21547a0994328d4fbd627c699ef3ccb43a0324279d7cd1f7e17be156c",
        "exercise": "MobiPhysio E07, hip joint flexion",
        "participant": "P02",
        "view": "front",
        "condition": "full light",
    },
    {
        "filename": "E07_P02_AR_VFL_GM.mp4",
        "url": "https://dataverse.harvard.edu/api/access/datafile/11718300",
        "sha256": "06e0cbcb6adb371961889d9bd3b7a9c193460426606a2091dcc673cd1bc9ca2d",
        "exercise": "MobiPhysio E07, hip joint flexion",
        "participant": "P02",
        "view": "right",
        "condition": "full light",
    },
    {
        "filename": "E07_P17_AR_VML_GM.mp4",
        "url": "https://dataverse.harvard.edu/api/access/datafile/11788645",
        "sha256": "e07908ad9ea7211d994c9473936f4d3781e912a41fe595cf0193a6d360179ba8",
        "exercise": "MobiPhysio E07, hip joint flexion",
        "participant": "P17",
        "view": "right",
        "condition": "medium light",
    },
]

MODELS = [
    {
        "filename": "pose_landmarker_lite.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            "?generation=1682624738331272"
        ),
        "sha256": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    },
    {
        "filename": "pose_landmarker_full.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            "?generation=1682642787774579"
        ),
        "sha256": "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    },
    {
        "filename": "movenet_singlepose_lightning_f16.tflite",
        "url": (
            "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/"
            "tflite/float16/4?lite-format=tflite"
        ),
        "sha256": "0fac2226112d0371903ca86e3853cec24ef603a0b2f96f589b180f0ebdd135ab",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(item: dict[str, str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == item["sha256"]:
        print(f"verified {destination.relative_to(ROOT)}")
        return

    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        item["url"], headers={"User-Agent": "RelayPoseBenchmark/1.0"}
    )
    print(f"downloading {item['filename']}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)

    actual = sha256(temporary)
    if actual != item["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {item['filename']}: "
            f"expected {item['sha256']}, got {actual}"
        )
    temporary.replace(destination)
    print(f"verified {destination.relative_to(ROOT)}")


def main() -> int:
    for sample in SAMPLES:
        download(sample, ROOT / "data" / "raw" / "mobiphysio" / sample["filename"])
    for model in MODELS:
        download(model, ROOT / "models" / model["filename"])

    manifest = {
        "dataset": {
            "title": "MobiPhysio",
            "doi": "10.7910/DVN/XSI0QN",
            "landingPage": "https://doi.org/10.7910/DVN/XSI0QN",
            "license": "CC0 1.0",
            "articleDoi": "10.1016/j.dib.2026.112635",
            "samples": SAMPLES,
        },
        "models": MODELS,
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
