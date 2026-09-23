# Pose-native decoder capability and sampling audit

**Evidence cutoff:** 2026-08-07  
**Purpose:** correct the earlier conflation of `a head can be attached` with `the
physical signal supports the capability`.

## Corrected conclusion

Seven heads can fit mathematically beside a 25.83M Nexus decoder. That does not make
their outputs valid. The limiting questions are:

1. Was the event sampled often enough?
2. Is the target observable from monocular kinematics?
3. Is a causal label available at that instant?
4. Is there enough independent real data to learn and calibrate it?
5. Does the model output a distribution or pretend an ambiguous future is certain?

The honest first model should implement only outputs whose answers to those questions
are demonstrated for a named exercise, camera setup and rate.

## What the movement timescales imply

Published biomechanics provides useful anchor values:

- A deliberately slow belt squat protocol used a 4-second eccentric and 2-second
  concentric condition; its traditional condition used 2 seconds eccentric and 1
  second concentric ([Wilk et al. protocol](https://pubmed.ncbi.nlm.nih.gov/35394146/)).
- The combined countermovement and takeoff phases of a vertical jump are approximately
  0.5 seconds ([Struzik et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441889/)).
- A representative countermovement-jump flight phase was 402 ± 34 ms, while peak
  landing force occurred 86 ± 18 ms after touchdown
  ([Le Pellec and Maton](https://pmc.ncbi.nlm.nih.gov/articles/PMC4624769/)).
- Force-platform jump studies commonly collect kinetics at much higher rates; one
  recent protocol used 2,000 Hz force data while its landing-analysis interval was 250
  ms ([Makaruk et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12654891/)).
- Smartphone flight-time work has validated 60/120/240 Hz video settings, with 240 Hz
  recommended when frame-rate timing error must be negligible relative to other error
  sources ([Pueo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10108745/)).

The number of sample intervals inside an event is simply:

\[
N=f_sT,
\]

where \(f_s\) is the *effective pose-output* rate, not the camera's advertised rate.

| Event/phase | Duration used | 10 Hz | 15 Hz | 30 Hz | 60 Hz | 120 Hz |
|---|---:|---:|---:|---:|---:|---:|
| squat descent, traditional tempo | 2.0 s | 20 | 30 | 60 | 120 | 240 |
| squat ascent, traditional tempo | 1.0 s | 10 | 15 | 30 | 60 | 120 |
| short bottom hold | 0.2 s | 2 | 3 | 6 | 12 | 24 |
| jump countermovement + takeoff | 0.5 s | 5 | 7.5 | 15 | 30 | 60 |
| representative flight | 0.402 s | 4.0 | 6.0 | 12.1 | 24.1 | 48.2 |
| touchdown to representative force peak | 0.086 s | 0.86 | 1.29 | 2.58 | 5.16 | 10.32 |
| landing analysis interval | 0.25 s | 2.5 | 3.75 | 7.5 | 15 | 30 |

This is before dropped frames, pose failures, temporal smoothing and inference
latency. Having two samples per cycle satisfies an idealized Nyquist condition for a
sinusoid; it is not enough to localize several nonlinear phases or estimate a sharp
derivative robustly.

## Boundary timing has a hard lower bound

The frame grid alone gives a worst-case nearest-sample error of \(1/(2f_s)\). A causal
system that waits for \(n\) confirming observations adds approximately \(n/f_s\), plus
pose and decoder latency:

\[
L_{decision}\gtrsim \frac{1}{2f_s}+\frac{n}{f_s}
+L_{pose}+L_{decoder}+L_{output}.
\]

| Effective pose rate | Frame period | Worst nearest-sample error | 2-frame confirmation | 3-frame confirmation |
|---:|---:|---:|---:|---:|
| 10 Hz | 100 ms | 50 ms | 200 ms | 300 ms |
| 15 Hz | 66.7 ms | 33.3 ms | 133 ms | 200 ms |
| 30 Hz | 33.3 ms | 16.7 ms | 66.7 ms | 100 ms |
| 60 Hz | 16.7 ms | 8.3 ms | 33.3 ms | 50 ms |
| 120 Hz | 8.3 ms | 4.2 ms | 16.7 ms | 25 ms |

At 10 Hz, a three-observation causal confirmation consumes 300 ms of a 500 ms
countermovement/takeoff. It can label the event retrospectively; it cannot provide a
meaningful mid-jump correction. The cited jump-control work also notes that movement
modification within the approximately 0.5-second ballistic phase is effectively
preplanned. A useful jump cue is therefore generally delivered before the next
attempt, not during the current takeoff.

## Capability-by-capability verdict

| Proposed output | Parameter-feasible? | Signal-feasible? | Correct scope |
|---|---|---|---|
| coarse phase | Yes | For slow known exercises at 10-15 Hz; faster movements need 30-60+ Hz | Exercise-specific phases only |
| phase boundary | Yes | Timing limited by sample grid, smoothing and causal confirmation | Report latency/tolerance; jump contact may need foot view or contact sensor |
| progress | Yes | Only for a defined trajectory; pauses, reversals and partial reps make one scalar ambiguous | Predict cyclic phase plus direction/state, not universal percent complete |
| form error | Yes | Only for predeclared, visible kinematic deviations | One validated head per named error/view; no generic “bad form” |
| observability | Yes | Joint visibility can be estimated; semantic observability cannot be proven from softmax | Combine view geometry, required joints, confidence and explicit abstention |
| uncertainty | Yes | A number can be emitted; trustworthy uncertainty requires held-out calibration/OOD tests | Call it calibrated predictive confidence only after validation |
| action | Yes | Output is easy; action quality needs therapist decisions and intervention outcomes | Constrained approved actions; imitation/offline RL later |
| future motion | Yes | Future is multimodal and horizon-dependent | Short-horizon distribution; not deterministic 2-second trajectory |
| landing force/load | No from pose alone | Not observable from monocular kinematics | Add force/contact/IMU sensing or decline the output |
| pain/tissue safety | No | Not visually identifiable | User report/clinical input; never infer from pose |

### Important corrections

- `Observability` is not another magical neural capability. The system can know that a
  joint is cropped or confidence is low. It cannot know from the same image that an
  out-of-plane compensation definitely did not occur.
- `Uncertainty` is not automatically meaningful because a head emits variance. It must
  be calibrated by participant/device/view and tested under distribution shift.
- `Error` is not one output. `Visible right-knee medial displacement in a supported
  frontal squat view` is a learnable target. `Unsafe squat` is not.
- `Future motion` should begin as a probability distribution over a low-dimensional
  100-500 ms latent. A deterministic full-skeleton forecast will average multiple
  plausible futures or appear confidently wrong.

## The heads fit; the labels are the hard part

For Nexus width \(d=512\), simple linear heads have these parameter counts:

| Head | Output dimensions | Parameters including bias |
|---|---:|---:|
| six coarse phases | 6 | 3,078 |
| three boundary states | 3 | 1,539 |
| cyclic progress `(sin, cos)` | 2 | 1,026 |
| twelve named deviations | 12 | 6,156 |
| per-joint visibility/quality | 33 | 16,929 |
| twenty constrained actions | 20 | 10,260 |
| sixteen calibration/scale values | 16 | 8,208 |
| deterministic 10×33×3 pose forecast | 990 | 507,870 |
| **Total** | 1,092 | **555,066 (2.15% of Nexus)** |

This proves only that memory and multiplication for the readouts are small. It does
not prove that one 512-D hidden state, one dataset, and one monocular stream can learn
all targets without negative transfer. Every head needs an ablation against its
single-task model, gradient-conflict measurement, real held-out evaluation and a
decision about whether its label is even legitimate.

The full-skeleton forecast above is also an intentionally naive upper bound: at 30 Hz
it covers only 333 ms, is deterministic, and would need more outputs or a latent
probabilistic decoder to represent multiple futures.

## Context changes with the required sampling rate

With 256 of 1,024 positions reserved for instruction, memory, events and output, 768
remain for pose:

| Pose-token rate | Fine-grained history |
|---:|---:|
| 10 Hz | 76.8 s |
| 15 Hz | 51.2 s |
| 30 Hz | 25.6 s |
| 60 Hz | 12.8 s |
| 120 Hz | 6.4 s |

Therefore 1,024 is not “the best context” independently of movement. A jump model at
60 Hz gets only 12.8 seconds of raw pose history; a slow rehabilitation model at 10 Hz
gets 76.8 seconds. Adding multiple joint/part tokens per frame reduces it further.

## Correct runtime design: multirate, not one universal FPS

The defensible architecture is:

```text
camera 30/60/120 fps depending task
  -> pose/optical features at the highest sustained rate
  -> short causal high-rate buffer/stem
  -> adaptive decoder tokens
       idle/slow movement: 5-15 tokens/s
       squat: 15-30 tokens/s if fine boundaries matter
       jump: 30-60 tokens/s for kinematics
       sharp contact/force: separate sensor, not hallucinated from pose
```

The high-rate stem must not discard events when compressing. It can emit one normal
token per interval plus an additional boundary/detail token when acceleration,
uncertainty or foot-contact evidence changes sharply. Train and evaluate the entire
stem-plus-decoder causally.

For current Nexus, approximate cached block work at a full 1,024 context is about 31M
MAC per accepted pose token (22.55M matrix MAC plus up to roughly 8.39M attention MAC).
That gives approximately:

| Decoder input rate | Approximate Nexus work |
|---:|---:|
| 10 tokens/s | 0.31 GMAC/s |
| 15 tokens/s | 0.47 GMAC/s |
| 30 tokens/s | 0.93 GMAC/s |
| 60 tokens/s | 1.86 GMAC/s |

These are arithmetic estimates, not phone benchmarks. Pose inference, cache movement,
delegate dispatch and thermal throttling remain unmeasured. At high token rates, the
short front end may need to make the fastest boundary decision while Nexus handles
longer contextual interpretation. That is a shared trained system, but it is not the
claim that every fast event is decoded by eight full Nexus layers.

## Minimum experiment before claiming the architecture

Run two separate studies rather than a universal demo.

### Slow known exercise

- effective pose rates: 10, 15, 30 Hz;
- outputs: coarse phase, rep completion, one or two explicitly visible deviations;
- compare Nexus against TCN/GRU/rules;
- measure boundary F1 at ±50/100/250 ms, confirmation delay, rep errors, calibration,
  and participant/device/view splits.

### Jump

- camera: 60/120/240 Hz reference capture;
- evaluated pose paths: 30 and 60 Hz;
- synchronized force/contact reference;
- outputs: kinematic phases and flight/touchdown timing only;
- explicitly exclude force/load inference from pose;
- test whether a pre-jump or next-attempt cue is useful—do not claim mid-ballistic
  correction.

Only retain a head if it beats a single-task baseline on real held-out people without
degrading the other accepted heads or calibration. The result may be two models or two
rate profiles. That is a scientifically acceptable outcome.
