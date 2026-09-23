# Movement evaluation

## What has and has not been measured

Relay has four distinct evidence layers. They must not be blended into a claim
of clinical or real-world accuracy.

| Layer | Input | What it supports | What it does not support |
|---|---|---|---|
| Product replay | Deterministic simulated landmarks | State, constraint, pause, failure, summary, and journey behavior | Performance on camera video or people |
| Geometry sensitivity | Idealized projection plus seeded pixel noise | Aspect correction, side-view guidance, minimum projected size | Pose-model or anatomical accuracy |
| Pose-provider benchmark | 1,028 frames from three public rehabilitation clips | Relative provider coverage, latency, temporal consistency, visible identity failures | Target-exercise, phone, or anatomical angle accuracy |
| Temporal evaluation | 12 held-out synthetic virtual subjects | Learned-sequence feasibility, export fidelity, abstention/runtime behavior | Human exercise recognition or phase accuracy |

There is no synchronized anatomical ground truth, no exact supported-exercise
human dataset, and no target-phone performance study in the repository.

## Pose-provider selection

The benchmark used three checksum-verified CC0
[MobiPhysio](https://doi.org/10.7910/DVN/XSI0QN) clips: 1,028 common timestamps
sampled at about 10 Hz from two people, front/right views, full/medium light.
The movement is supine hip flexion with a bent knee—a useful horizontal-body
and occlusion proxy, but not a heel slide, straight-leg raise, or seated knee
extension.

Host: 10-core Apple M2 Pro MacBook Pro, 16 GB RAM. Video decode, rendering,
application logic, and speech were excluded. MediaPipe used Python Tasks
0.10.31; MoveNet Lightning used TensorFlow 2.20/XNNPACK with four threads.

Primary result at each provider’s raw 0.5 active hip–knee–ankle chain threshold:

| Metric | MediaPipe Lite | MediaPipe Full | MoveNet Lightning |
|---|---:|---:|---:|
| Active-chain coverage | 100.0% | 100.0% | 28.4% |
| Pose time p50 | 13.65 ms | 19.31 ms | 5.40 ms |
| Pose time p95 | 30.09 ms | 30.36 ms | 11.19 ms |
| Derived sequential throughput | 62.2 FPS | 47.1 FPS | 149.8 FPS |
| Low-motion angle-step p50 | 1.56° | 1.32° | 2.17° |
| Low-motion angle-step p95 | 10.70° | 8.23° | 12.65° |

MediaPipe Full was selected because it retained the 33-point image/world
contract, heel/foot landmarks, and full selected-chain coverage while showing
lower low-motion jitter than Lite. MoveNet was faster but its high-threshold
active-chain availability on the horizontal proxy was insufficient.

This choice does not imply that Full is anatomically accurate. Full-versus-Lite
image-angle disagreement was 9.42° median and 68.72° p95. Reviewed outliers
showed Lite following the inactive leg with high raw confidence. The result
motivates selected-side continuity and abstention; it is not ground truth.

Rotating one lying clip to put the head at the top did not consistently help and
reduced MoveNet coverage. Relay therefore keeps the native camera orientation
rather than forcing a generic upright transform.

Full details and annotated disagreement examples are in
[`experiments/pose-benchmark/REPORT.md`](../experiments/pose-benchmark/REPORT.md).

## Geometry sensitivity

The seeded deterministic experiment isolates projection and coordinate risks:

| Constructed condition | Result |
|---|---:|
| Treating normalized x/y as equal-scale in 16:9 | 13.877° median absolute error; 31.493° maximum |
| Aspect-correct source-pixel calculation in the same ideal cases | 0° maximum |
| 15° out-of-plane yaw | 1.735° p95 absolute error |
| 30° out-of-plane yaw | 7.380° p95 |
| 45° out-of-plane yaw | 18.463° p95 |

At a true constructed 90° angle with independent 2-pixel landmark noise:

| Projected segment length | Error SD | p95 absolute error |
|---:|---:|---:|
| 50 px | 4.539° | 8.844° |
| 75 px | 3.064° | 5.951° |
| 100 px | 2.303° | 4.533° |
| 150 px | 1.534° | 3.001° |

These results explain the aspect correction, side-on setup, and current
75-pixel minimum segment gate. They contain no pose model or human anatomy.
Inputs and committed output are in
[`experiments/geometry/`](../experiments/geometry/).

## Temporal movement model

Relay Temporal Synthetic v1 is a 2,351-parameter causal temporal convolution:

- 24 frames at nominal 10 Hz;
- 22 inputs: 18 normalized 2D/world geometry and tracking features plus four
  first differences;
- two 16-channel causal convolutions and a 15-class exercise-phase head;
- int8 weights in a 7,813-byte JSON artifact;
- SHA-256
  `8312867bd19bc6d531eb2f2b8ad415ba01cc5a47950b8f29f1bd06611954e23f`.

Training used 30 synthetic subjects/24,051 frames; validation used 8/6,404;
held-out evaluation used 12/12,339. Virtual subjects do not cross splits.

| Held-out synthetic metric | Quantized temporal model | Kinematic baseline |
|---|---:|---:|
| Joint exercise-phase macro-F1 | 0.868 | 0.404 |
| Exercise macro-F1 | 0.978 | 0.833 |
| Phase macro-F1 | 0.883 | 0.452 |
| Phase macro-F1 conditioned on confirmed exercise | 0.887 | 0.475 |

Float and int8 class decisions agreed on 98.76% of held-out frames. After the
synthetic confidence, margin, visibility, and mismatch gates, coverage was
88.05% and accepted phase accuracy was 89.87%. These are synthetic-only
statistics, not calibrated probabilities for people.

Raw learned transition precision was only 0.365 at 0.913 recall within two
frames (F1 0.521). For that reason the product does not convert a single
temporal class change into a rep or restriction event. Applicable evidence can
corroborate a hold after the geometric top has already been reached, or a return
after a smaller but still explicit geometric reversal. It cannot establish the
endpoint, complete a rep, or trigger a constraint; the explicit exercise engine
remains authoritative.

### Measured edge runtime

The checked-in 5,000-iteration microbenchmark ran on arm64 Darwin 25.5 with
Node v24.11.0 and V8 13.6.233.10:

| Operation | Measured time |
|---|---:|
| Artifact file read + JSON parse | 0.167 ms |
| Tensor decode + validation | 0.647 ms |
| Feature extraction + inference median | 0.083 ms |
| Inference p95 / p99 | 0.186 / 0.367 ms |

It excludes camera capture, pose inference, React rendering, and speech. It is
not a phone-browser result. See [`ml/MODEL_CARD.md`](../ml/MODEL_CARD.md),
[`ml/EXPERIMENT_REPORT.md`](../ml/EXPERIMENT_REPORT.md), and the raw
[`edge-benchmark.json`](../ml/results/edge-benchmark.json).

## Shared replay evaluation

The same movement engine used by the camera path is exercised with these
published scenarios:

| Scenario | Expected behavior under test |
|---|---|
| Clean repetitions | Calibrate, recognize phases, hold, return, count |
| Incomplete movement | Do not count an excursion that misses the endpoint |
| Incomplete return | Prompt for return and do not count |
| Too fast / slow | Separate missed and met minimum tempo |
| Tracking loss / occlusion | Pause assessment and exclude hidden time from hold |
| Poor camera position | Do not count unusable movement |
| Visible compensation | Emit a broad posture observation with good feature quality |
| Approach / reach maximum | Separate advance warning from stop-limit event |
| Uncertain angle | Abstain from angle constraints while geometry disagrees |

Automated suites also cover instruction semantics, generic constraints,
pause/resume cursor and timing behavior, summaries/exports, and the required
sentence through editable confirmation and camera setup.

## Reproduce

From the repository root:

```bash
npm test
npm run test:ml
node experiments/geometry/angle-sensitivity.mjs
python3 -m pip install -r ml/requirements.txt
python3 ml/run_experiment.py
node --experimental-strip-types ml/benchmark_edge.mjs --iterations 5000
```

Recorded provider benchmark:

```bash
cd experiments/pose-benchmark
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
./run.sh --sample-hz 10 --warmup-frames 10 --cold-load-probes 3
.venv/bin/python -m unittest -v test_benchmark.py
```

The benchmark download script verifies exact clip and model checksums recorded
in `provenance.json`.

## Evaluation conclusion

The evidence is sufficient to select an MVP provider, demonstrate a generalized
pose→feature→constraint pipeline, and test its product behavior repeatably. It
is insufficient to publish camera-angle accuracy, rep-count reliability on
people, warning sensitivity, demographic generalization, target-phone latency,
or clinical validity.

The next decisive experiment is a frozen-pipeline, participant-held-out,
synchronized phone study of right heel slides around the 90° boundary on
representative iPhone and Android devices. It should measure reference error,
selected-leg identity, abstention, late/missed warnings, rep/hold error, and
end-to-end cue latency under planned camera and environment variation.
