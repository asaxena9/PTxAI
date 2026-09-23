#!/usr/bin/env python3
"""Generate the disjoint synthetic pose-landmark dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relay_temporal.dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/.generated/relay-temporal-synthetic-v1.npz"),
    )
    args = parser.parse_args()
    summary = prepare_dataset(args.output)
    print(json.dumps(summary["splits"], indent=2))
    print(f"Wrote {args.output} ({summary['archiveBytes']} bytes)")


if __name__ == "__main__":
    main()
