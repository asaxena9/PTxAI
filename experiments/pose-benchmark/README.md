# Relay pose benchmark

This isolated experiment compares MediaPipe Pose Landmarker Lite and Full with
MoveNet SinglePose Lightning on identical public rehabilitation footage.

Read `REPORT.md` for the measured results, provenance, recommendation, and
limitations. `provenance.json` records exact source files, model URLs, sizes,
and SHA-256 hashes.

## Reproduce

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
./run.sh --sample-hz 10 --warmup-frames 10 --cold-load-probes 3
```

`download_assets.py` fetches only three MobiPhysio clips (about 36 MB total)
and the three model assets, then verifies every checksum. Raw video, model
files, and the mutable `results/latest/` directory are gitignored.

Run the focused input-orientation experiment with:

```bash
.venv/bin/python benchmark.py \
  --sample-hz 10 \
  --warmup-frames 10 \
  --cold-load-probes 0 \
  --clips E07_P02_AR_VFL_GM.mp4 \
  --input-rotation-deg -90 \
  --output results/orientation-head-up
```

Render disagreement evidence:

```bash
.venv/bin/python render_diagnostics.py --results results/latest
```

Run geometry tests:

```bash
.venv/bin/python -m unittest -v test_benchmark.py
```

## Committed results

- `results/2026-07-27-m2-pro/summary.json`
- `results/2026-07-27-m2-pro/frame_metrics.csv`
- `results/2026-07-27-m2-pro/disagreement_examples.jpg`
- `results/2026-07-27-orientation-head-up/summary.json`
- `results/2026-07-27-orientation-head-up/frame_metrics.csv`
