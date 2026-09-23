# Relay

Relay turns a confirmed home-exercise plan into a camera-guided movement
session. The MVP is anonymous, mobile-first, local-first, and supports heel
slides, straight-leg raises, and seated knee extensions.

Relay is a **body-pose system**, not a knee-angle tool. Its runtime follows the
same pattern for every supported exercise:

```text
pose landmarks → observable body features → scoped constraints
→ confidence-aware events → visual and spoken guidance
```

Camera-derived values are estimates, not clinical measurements. Relay follows
the session the user confirms; it does not diagnose, interpret pain, determine
exercise appropriateness, or decide readiness to progress.

## What is implemented

- Typed, spoken, uploaded, manual, and template-based session creation.
- Focused clarification for a missing side, missing dose, or an angle whose
  meaning as a target versus hard maximum is unclear.
- An editable confirmation step for exercise, side, sets, reps, hold, tempo,
  targets, and restrictions.
- A real browser-camera provider using a locally served MediaPipe Pose
  Landmarker Full model and a repeatable simulated-landmark provider using the
  same downstream movement pipeline.
- Per-exercise calibration, framing guidance, tracking quality, repetition
  phases, holds, tempo, incomplete movement, and incomplete-return handling.
- Exercise-specific body features:
  - heel slide: hip–knee–ankle flexion and per-repetition peak estimate;
  - straight-leg raise: leg-chain lift, visible knee bend, and broad
    shoulder–hip stability;
  - seated knee extension: extension progress relative to the calibrated
    seated start.
- Generic target, minimum, maximum, range, visible-condition, tempo, and hold
  evaluation over joint, relationship, whole-body, and temporal scopes.
- Spoken and large-format visual coaching, pause, stop, multi-set and
  multi-exercise progression, summaries, local history, deletion, and JSON
  export/share.
- A compact temporal exercise/phase model running locally as cautious
  corroboration for hold/return transitions. Deterministic geometry,
  hysteresis, and feature quality remain authoritative for repetitions and
  constraints.

## Run locally

Prerequisites:

- Node.js 22.13 or newer
- A current browser; camera use requires `localhost` or another secure context

```bash
npm install
npm run dev
```

Open the local address printed by the development server. No account, database,
API key, or cloud model credential is required.

Production build and local production server:

```bash
npm run build
npm run start
```

## Try the product

The fastest complete path is **Try the guided demo**. Review its three
exercises, confirm the plan, choose **Guided replay**, run the setup check, and
start. The replay uses simulated landmarks but passes them through the same
geometry, confidence, constraint, coaching, set, and summary code as the live
camera.

For the personalized hard-limit proof, enter or load:

> Right heel slides, three sets of ten, hold for two seconds, and do not bend
> my knee beyond 90 degrees.

After confirming, switch camera setup to **Guided replay** and choose
**Approach maximum** or **Reach maximum**. For live use, leave **Live camera**
selected, choose the front or rear camera, place the phone landscape about 6–8
feet away, confirm its support and the selected side, keep the requested joints
visible, allow camera access, and hold still through the stability check.

See [the guided product demonstration](docs/PRODUCT_DEMO.md) for exact paths
through replay, live camera, upload, manual configuration, history, and export.

## Verify the repository

```bash
npm run lint
npm run typecheck
npm test
npm run test:ml
npm run build
```

`npm run test:all` runs lint, the TypeScript check, application tests, temporal
edge tests, and the production build.

Verification snapshot on 2026-07-27:

| Check | Result |
|---|---|
| Application tests | 9 files, 39 tests passed |
| Temporal edge tests | 7 tests passed |
| ESLint | Passed |
| TypeScript | Passed with no emit |
| vinext production build | Passed; `/` built for the Worker output |
| Geometry experiment | Reproduced the committed `results.json` exactly |

Focused experiment commands:

```bash
# Deterministic projection and landmark-noise sensitivity
node experiments/geometry/angle-sensitivity.mjs

# Re-run the checked-in temporal training/evaluation pipeline (Python 3.11)
python3 -m pip install -r ml/requirements.txt
python3 ml/run_experiment.py

# Benchmark the deployed temporal edge for 5,000 iterations
node --experimental-strip-types ml/benchmark_edge.mjs --iterations 5000

# Re-run the recorded public-video pose-provider benchmark
cd experiments/pose-benchmark
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
./run.sh --sample-hz 10 --warmup-frames 10 --cold-load-probes 3
```

The pose benchmark downloads checksum-verified public clips and model assets.
Its raw downloads and mutable latest-results directory are intentionally not
committed.

## Evidence at a glance

| Evidence | Actual result | Boundary |
|---|---|---|
| Public-video provider benchmark | MediaPipe Full and Lite each produced the selected active leg chain on 100% of 1,028 sampled proxy frames at the report’s 0.5 raw threshold; Full had lower measured low-motion jitter than Lite | Three clips, two people, non-target exercise, desktop host, no anatomical ground truth |
| Temporal model | Quantized model joint exercise-phase macro-F1 0.868 versus 0.404 for the baseline on 12 held-out synthetic subjects | Generated pose sequences only; not performance on people |
| Geometry sensitivity | Aspect-correct coordinates removed the constructed 16:9 scaling error; yaw and short limb projections produced increasing synthetic error | Idealized projection/noise experiment, not camera or anatomical accuracy |
| Replay suite | Covers complete/incomplete reps, holds, tempo, tracking loss, occlusion, poor framing, posture variation, targets, hard maxima, and uncertain geometry | Deterministic simulated landmarks |

The full numbers, host conditions, and reproduction notes are in
[Movement evaluation](docs/MOVEMENT_EVALUATION.md). The architecture and
extension contract are in [Architecture](docs/ARCHITECTURE.md). Read
[Known limitations](docs/KNOWN_LIMITATIONS.md) before interpreting any result
as real-world performance.

## Local-first behavior

- Live frames are processed in browser memory; Relay does not persist or upload
  continuous camera video.
- The upstream MediaPipe runtime may report performance and utilization metrics
  to Google under its [privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice).
  Relay does not send raw frames; a formal production network audit remains
  required.
- PDF text extraction and photograph OCR run in the browser. Photograph OCR
  downloads Tesseract worker, core, and English language assets on first use;
  Relay does not intentionally upload the selected image. The interpreted text
  must be reviewed before confirmation.
- Confirmed plans and structured results are stored in browser `localStorage`
  until the user deletes them. Relay keeps at most 50 session records. If
  storage is unavailable, the summary still opens and asks the user to export.
- Export/share is user initiated and contains structured JSON, not camera
  video.
- A service worker caches core application and pose assets after they have
  loaded. This improves resilience after first use; it is not a promise that a
  first visit—or first-use photograph OCR—works offline.

## Repository map

| Path | Purpose |
|---|---|
| `src/domain/` | Confirmed session and generic constraint contracts |
| `src/intelligence/` | Local instruction parsing and upload extraction |
| `src/motion/` | Providers, body-feature extraction, confidence, constraints, temporal edge, and exercise engine |
| `src/session/` | Structured results, summaries, export, and local history |
| `tests/` | Product journey, interpretation, engine, replay, constraint, pause/resume, and summary tests |
| `ml/` | Reproducible synthetic temporal-model pipeline, model card, and measured outputs |
| `experiments/` | Recorded pose-provider benchmark and deterministic geometry sensitivity experiment |
| `public/models/` | Locally served pose and temporal artifacts |

## Founding-team workstreams

- Product and integration established the anonymous
  tell → clarify → confirm → position → move → review experience and joined all
  workstreams into one product.
- Motion research compared deployable pose providers on public rehabilitation
  footage and selected MediaPipe Full for the live MVP.
- Movement evaluation built the shared replay matrix and the geometry and
  public-video benchmarks.
- Temporal modeling produced the reproducible synthetic dataset, baseline,
  trained causal model, quantized browser artifact, edge runtime, and model
  card.
- Fresh product, motion, privacy, accessibility, and failure-case review drove
  revisions to calibration, confidence gating, pause/resume timing, parser
  semantics, camera setup, multi-exercise transitions, and local-data controls.

## Most important next experiment

Record consented right heel slides around 70–100° on representative iPhones and
Android phones, synchronized to an independent electrogoniometer or optical
reference. Vary camera yaw, distance, clothing, lighting, occlusion, and
participants; freeze the current pipeline before testing. Measure signed and
peak angle error, selected-leg identity failures, confidence-gate abstention,
missed or late 90° warnings, rep/hold errors, and end-to-end spoken-cue latency.

That experiment is the shortest path from a credible technical MVP to evidence
about whether its most safety-sensitive real-world behavior is usable.
