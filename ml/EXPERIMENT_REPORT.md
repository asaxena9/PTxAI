# Temporal movement experiment

## Decision

Relay uses a hybrid architecture:

1. The pose provider supplies image and inferred-world landmarks.
2. Explicit quality gates decide whether requested geometry is observable.
3. The temporal model contributes exercise and phase evidence.
4. Exercise-specific kinematics and hysteresis decide repetitions and holds.
5. Confirmed structured rules decide targets and restrictions.

This preserves explainability for safety-relevant range behavior while adding a
genuinely learned temporal representation where sequences are more informative
than one-frame thresholds.

## Compared approaches

The learned candidate is a small causal temporal convolution over normalized
landmark geometry, velocities, and tracking quality. The comparison baseline is
deliberately transparent: trunk orientation distinguishes seated from lying;
leg elevation and knee flexion distinguish the two lying exercises; direction,
top-position, and visibility thresholds assign the current phase.

Both approaches receive the same trailing pose-derived feature windows. The
model was selected because it materially outperformed the baseline on the
separate generated held-out split, its artifact is small, and its pure
TypeScript runtime avoids a second browser inference framework.

## Reproducibility

The run used:

- Python 3.11.5
- PyTorch 2.6.0
- NumPy 1.26.4
- scikit-learn 1.6.1
- Dataset seed 73021
- Training seed 9173
- AdamW, learning rate 0.003, weight decay 0.0001
- Weighted cross-entropy, batch 256
- Early stopping after six non-improving epochs

Training stopped after 23 epochs; epoch 17 was selected with validation joint
macro-F1 0.8997. Full epoch history is checked in at
`ml/results/training-history.json`.

The generated splits and complete held-out metrics are in
`ml/results/held-out-evaluation.json`. The data archive and float checkpoint
are reproducible but ignored to avoid committing derived binaries.

## Results

On 12 disjoint held-out virtual subjects (180 sequences, 12,339 frames), the
deployed quantized model achieved:

- Joint exercise-phase accuracy 0.852 and macro-F1 0.868.
- Exercise accuracy 0.978 and macro-F1 0.978.
- Phase accuracy 0.864 and macro-F1 0.883.
- Phase macro-F1 0.887 when conditioned on the confirmed exercise.

The baseline reached joint macro-F1 0.404, exercise macro-F1 0.833, phase
macro-F1 0.452, and confirmed-exercise phase macro-F1 0.475.

Raw transition evaluation shows why the integrated choice is hybrid. Within a
two-frame tolerance, the learned model had boundary recall 0.913, precision
0.365, and F1 0.521; the noisier baseline had recall 0.997, precision 0.104,
and F1 0.189. Relay should require geometrically plausible progression and
hysteresis before converting phase evidence into rep events.

## Edge export and benchmark

The 2,351-parameter artifact uses int8 weights and occupies 7,813 bytes. The
TypeScript edge performs feature extraction, tensor decode, causal convolution,
softmax calibration, expected-exercise conditioning, and abstention without a
runtime dependency.

On this development host (arm64 macOS, Node 24.11.0/V8), one 5,000-iteration
microbenchmark measured:

- Artifact file read + JSON parse: 0.167 ms.
- Tensor decode + validation: 0.647 ms.
- Feature extraction + inference median: 0.083 ms.
- p95: 0.186 ms.
- p99: 0.367 ms.
- Mean: 0.101 ms.

This excludes camera capture, pose inference, rendering, and speech. It is not
a phone-browser result. Target iPhone Safari and Android Chrome measurements
remain required before making on-device latency or power claims. The raw
benchmark record is `ml/results/edge-benchmark.json`.

## Verification performed

- Full data preparation, training, quantization, and held-out evaluation ran
  successfully.
- Int8 and float decisions agreed on 98.76% of held-out generated frames.
- Seven edge tests passed, including 15 Hz provider decimation to the trained
  10 Hz temporal cadence.
- Shared product replay traces for all three exercises produced ready, moving,
  holding, returning, and final ready behavior.
- Warm-up and tracking-loss traces abstained and recovered.
- Targeted TypeScript compilation passed.

## Evidence boundary

No real videos or human landmark recordings were available. Consequently this
experiment does not evaluate phone pose latency, real angle error, rep-count
error on people, demographic generalization, constraint-warning misses, or
clinical validity. Those claims remain blocked, not estimated from generated
data.
