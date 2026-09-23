#!/usr/bin/env python3
"""Run data preparation, training, and held-out evaluation in one command."""

from __future__ import annotations

import subprocess
import sys


def run(script: str) -> None:
    command = [sys.executable, f"ml/{script}"]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    run("prepare_data.py")
    run("train_temporal.py")
    run("evaluate_temporal.py")
