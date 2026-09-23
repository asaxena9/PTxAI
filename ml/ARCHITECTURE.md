# ADR: hybrid temporal movement intelligence

**Status:** accepted for the MVP
**Evidence date:** 2026-07-27

## Context

Relay knows the confirmed exercise, must respond locally, and must keep hard
range rules explainable. A pose SDK alone has no concept of exercise phase.
Single-frame thresholds are easy to audit but were unstable under generated
noise and camera variation. No representative human training set was available.

## Decision

Use a compact causal temporal convolution as non-authoritative phase evidence,
inside an exercise-specific hybrid engine.

- Normalize provider landmarks into side-canonical 2D/world geometry.
- Run a 24-frame, 10 Hz temporal window locally.
- Condition phase probabilities on the confirmed exercise.
- Abstain during warm-up, low current visibility, or ambiguous probability.
- Keep camera quality, knee angle, repetition hysteresis, holds, tempo, targets,
  and restrictions in explicit product logic.
- Export int8 weights to a custom JSON artifact and run them with a small
  TypeScript implementation behind a dedicated class.

## Why

On disjoint generated virtual subjects, the quantized model improved phase
macro-F1 from 0.452 to 0.883 and joint exercise-phase macro-F1 from 0.404 to
0.868 versus the transparent baseline. The 7,813-byte artifact adds negligible
compute next to pose inference and does not add TensorFlow, ONNX, or PyTorch to
the browser bundle.

The model is not allowed to own range-limit enforcement because there is no
human-camera evidence for that use. Raw transition precision is also too low
for direct rep segmentation, so deterministic hysteresis remains necessary.

## Consequences

The MVP demonstrates a trained learned component without obscuring its evidence
boundary. Future artifacts can replace the current weights without changing the
camera/session UI if they preserve or version the feature contract. The next
artifact should be trained and frozen against participant-level, consented
camera splits and compared with this version before receiving more authority.
