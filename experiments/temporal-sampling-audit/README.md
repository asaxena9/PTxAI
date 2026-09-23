# Temporal sampling audit

This is the first bounded falsification experiment under `AGENTS.md`.

## Predeclared claim

`HYPOTHESIS`: for the slow MobiPhysio E07 hip-flexion recordings already present in this repository, a separately executed MediaPipe Full pipeline at 10 or 15 Hz preserves repetition-level mid-range crossing events relative to a 30 Hz pipeline with:

- boundary-time p95 at or below 100 ms;
- zero unmatched reference boundaries;
- zero unmatched candidate boundaries.

The null is accepted for any rate that violates one of those thresholds.

This experiment does **not** test jumps, landings, force, anatomical angle accuracy, decoder learnability, phone performance, or patient generalization. The 30 Hz pose output is a higher-rate proxy reference, not biomechanical ground truth.

## Method

The same three CC0 MobiPhysio clips and the same MediaPipe Full asset are run independently at 10, 15, and 30 Hz. For the active leg fixed by the pose benchmark, the audit:

1. applies a 200 ms centered moving average to the image-space knee-flexion proxy;
2. defines a per-clip midpoint from the 20th and 80th percentiles of the 30 Hz reference;
3. detects signed threshold crossings separated by at least 400 ms;
4. matches candidate and reference crossings of the same direction within 350 ms;
5. measures boundary error, missing/extra crossings, angle reconstruction error, and angular-velocity error.

Centered smoothing and use of a 30 Hz reference make this an information-loss audit, not a deployable causal decoder evaluation.

## Commands

Generate independent pose streams:

```bash
experiments/pose-benchmark/.venv/bin/python experiments/pose-benchmark/benchmark.py \
  --models mediapipe-full --sample-hz 15 --cold-load-probes 0 \
  --output experiments/temporal-sampling-audit/runs/pose-15hz

experiments/pose-benchmark/.venv/bin/python experiments/pose-benchmark/benchmark.py \
  --models mediapipe-full --sample-hz 30 --cold-load-probes 0 \
  --output experiments/temporal-sampling-audit/runs/pose-30hz
```

Then compare the existing independently-run 10 Hz stream with the new streams:

```bash
experiments/pose-benchmark/.venv/bin/python \
  experiments/temporal-sampling-audit/audit.py \
  --series 10=experiments/pose-benchmark/results/2026-07-27-m2-pro/frame_metrics.csv \
  --series 15=experiments/temporal-sampling-audit/runs/pose-15hz/frame_metrics.csv \
  --series 30=experiments/temporal-sampling-audit/runs/pose-30hz/frame_metrics.csv \
  --reference-hz 30 \
  --output experiments/temporal-sampling-audit/results/2026-08-07-pilot
```

The output directory contains a manifest, aggregate metrics, per-clip metrics, matched boundary events, and a disposition for every tested rate.
