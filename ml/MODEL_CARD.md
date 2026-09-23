# Model card: Relay Temporal Synthetic v1

## Summary

Relay Temporal Synthetic v1 is a 2,351-parameter causal temporal convolution
that consumes a trailing pose-feature window and returns joint probabilities
over three exercises and five phases.

- Exercises: heel slide, straight-leg raise, seated knee extension.
- Phases: ready, moving, holding, returning, paused.
- Context: 24 frames at a nominal 10 Hz (2.4 seconds).
- Features: 18 normalized 2D/world kinematic and tracking-quality features,
  plus four first differences.
- Network: two 16-channel causal convolutions (kernel 3, ReLU), temporal mean
  plus last-state pooling, and a 15-class linear head.
- Export: symmetric per-tensor int8 weights in JSON; float normalization,
  biases, and temperature.
- Artifact: 7,813 bytes, SHA-256
  `8312867bd19bc6d531eb2f2b8ad415ba01cc5a47950b8f29f1bd06611954e23f`.

## Intended use

The model supplies exercise/phase evidence to Relay’s exercise-specific hybrid
motion engine after an off-the-shelf pose provider has produced landmarks. The
confirmed exercise can condition the five phase probabilities. A high-quality,
high-confidence result may support the state machine. Ambiguous, low-visibility,
or warm-up results abstain.

Explicit pose quality, joint geometry, user-confirmed targets/restrictions,
timestamp calculations, hysteresis, and repetition completion remain
deterministic and inspectable. The model does not decide diagnosis, exercise
appropriateness, pain, tissue safety, or readiness to progress.

## Training data

There are no human recordings in this version.

| Split | Virtual subjects | Sequences | Frames |
|---|---:|---:|---:|
| Train | 30 | 360 | 24,051 |
| Validation | 8 | 96 | 6,404 |
| Held out | 12 | 180 | 12,339 |

The split unit is virtual subject. Body proportions and subject seeds never
cross splits. The held-out generator increases camera-yaw range, landmark
noise, and occlusion probability. It varies side, limb proportions, camera
position, speed, hold duration, and short tracking loss.

This generated variation is a controlled test scaffold. It does not reproduce
pose-SDK mistakes, real anatomy, assistive devices, clothing, pain-limited
movement, atypical movement strategies, homes, cameras, or population
diversity.

## Held-out synthetic behavior

The deployed int8 artifact was evaluated once on the untouched held-out
virtual-subject split:

| Metric | Temporal model | Kinematic baseline |
|---|---:|---:|
| Joint exercise-phase macro-F1 | 0.868 | 0.404 |
| Exercise macro-F1 | 0.978 | 0.833 |
| Exercise macro-F1, active frames | 0.991 | 0.917 |
| Phase macro-F1 | 0.883 | 0.452 |
| Phase macro-F1, confirmed exercise | 0.887 | 0.475 |

Temporal phase F1 by class was 0.887 ready, 0.845 moving, 0.842 holding,
0.862 returning, and 0.980 paused. Joint macro-F1 was 0.876 for synthetic
side views, 0.876 for oblique views, and 0.848 for the hardest very-oblique
bucket. It was 0.839 fast, 0.875 normal, and 0.866 slow.

The float checkpoint reached 0.869 joint macro-F1. Int8 export reached 0.868,
with 98.76% class-decision agreement.

These numbers are **synthetic feasibility metrics only** and must never be
described as accuracy on patients, users, people, or camera video.

## Confidence and abstention

Temperature 1.05 and decision thresholds were selected only on the synthetic
validation split. On the harder synthetic held-out split:

- Joint expected calibration error: 0.0229.
- Confirmed-exercise phase expected calibration error: 0.0182.
- Coverage after confidence, margin, visibility, and mismatch gates: 88.05%.
- Phase accuracy among accepted predictions: 89.87%.
- Current required-landmark visibility below 0.45 abstains immediately.
- The first six frames abstain while accumulating temporal context.

These are not calibrated real-world probabilities. Product language should use
“tracking good,” “use with caution,” or “not measuring,” not a pseudo-clinical
confidence percentage.

## Important failure modes

- Pre-movement lying heel slides and straight-leg raises can be visually
  indistinguishable; the confirmed prescription is more trustworthy.
- Raw phase decisions still jitter near transitions. With a two-frame matching
  tolerance, raw boundary recall was 0.913 but precision was 0.365. Do not use
  a single class transition as a repetition boundary; retain state-machine
  hysteresis and geometry.
- Sampling materially faster or slower than the nominal 10 Hz changes the
  temporal distribution.
- Missing or incorrectly mirrored side semantics can corrupt canonicalized
  features.
- Synthetic visibility drops are simpler than real occlusion and pose-model
  recovery.
- Very oblique views underperform supported side views even in generated data.

## Ethical and privacy notes

The model consumes pose landmarks and needs no raw video persistence. That
does not remove the need for informed consent, privacy review, access control,
retention limits, or device-network auditing during human data collection.
Capture schemas deliberately require consent provenance and pseudonymous IDs.

## Next validation

Collect consented, synchronized phone-camera recordings from varied people and
devices, with a held-out participant split. The highest-priority subset is
right heel slides around a 90-degree reference measured by an independent
goniometric or optical system, including camera yaw and deliberate tracking
loss. Retrain only after the frozen artifact has first been evaluated as-is.
