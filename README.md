# PTxAI: Relay, camera-guided physical therapy in the browser

**Relay** turns a confirmed home-exercise plan into a camera-guided movement session. Everything runs in the browser: no account, no database, no API key and no cloud model. Pose estimation comes from Google's MediaPipe Pose Landmarker running on the device, and camera frames never leave browser memory.

Relay currently supports three knee-rehabilitation exercises: heel slides, straight-leg raises and seated knee extensions.

Next to the product, this repository keeps the research layer behind it: a reproducible synthetic temporal-model pipeline with a model card, benchmark reports on public data, and the design and evidence documents for a compact model that reads body pose directly instead of going through text.

## How it works

Every supported exercise runs through the same pipeline:

```text
pose landmarks -> observable body features -> scoped constraints
-> confidence-aware events -> visual and spoken guidance
```

- **Plan capture.** A session can be typed, spoken, uploaded (PDF text or photo OCR, both processed in the browser), entered by hand, or picked from a template. Relay asks a focused follow-up when the side, the dose, or the meaning of an angle (target versus hard maximum) is unclear, then shows an editable confirmation step.
- **Camera setup.** Per-exercise calibration, framing guidance and tracking-quality checks before the first repetition.
- **Movement engine.** Deterministic geometry with hysteresis decides repetitions, holds, tempo, incomplete movement and incomplete return. A 2,351-parameter temporal model runs locally as cautious corroboration for hold and return transitions; it never overrides the geometry.
- **Coaching and records.** Spoken and large-format visual cues, pause and stop, multi-set and multi-exercise progression, summaries, local history (at most 50 sessions in browser storage) and user-initiated JSON export.

Camera-derived values are estimates, not clinical measurements. Relay follows the session the user confirms. It does not diagnose, interpret pain, judge whether an exercise is appropriate, or decide readiness to progress. Read [Known limitations](docs/KNOWN_LIMITATIONS.md) before treating any number in this repository as real-world performance.

## Evidence

| Evidence | Result | Boundary |
|---|---|---|
| Temporal model ([model card](ml/MODEL_CARD.md)) | A 2,351-parameter causal temporal convolution, shipped as a 7,813-byte int8 JSON artifact. Joint exercise-phase macro-F1 of 0.868 against 0.404 for the kinematic baseline on 12 held-out synthetic subjects (180 sequences, 12,339 frames). | Generated pose sequences only, not people |
| Edge runtime ([benchmark](ml/results/edge-benchmark.json)) | Feature extraction plus int8 inference at a median of 0.083 ms and p95 of 0.186 ms per step over 5,000 iterations | Node desktop microbenchmark on an arm64 laptop, not a phone browser |
| Pose-provider benchmark ([report](experiments/pose-benchmark/REPORT.md)) | MediaPipe Full and Lite each produced the selected active leg chain on 100% of 1,028 sampled frames from three public MobiPhysio clips (CC0), and Full had lower low-motion jitter. MoveNet Lightning was faster but produced the full hip, knee and ankle chain much less often. | Three clips, two people, a non-target exercise, desktop host, no anatomical ground truth |
| Geometry sensitivity ([experiment](experiments/geometry/README.md)) | Aspect-correct coordinates removed the constructed 16:9 scaling error; camera yaw and short limb projections produced increasing synthetic error | Idealized projection and noise experiment |
| Synthetic-data audit ([report](experiments/synthetic-temporal-audit/REPORT.md)) | A parameter-matched order-invariant model reaches 0.853 against the temporal model's 0.887, so most of the score does not depend on ordered history. The audit also records a flaw found in the synthetic generator. | Self-critical ablation on synthetic data |
| Replay suite (`tests/`) | Complete and incomplete reps, holds, tempo, tracking loss, occlusion, poor framing, posture variation, targets, hard maxima and uncertain geometry | Deterministic simulated landmarks |

Full numbers, host conditions and reproduction notes are in [Movement evaluation](docs/MOVEMENT_EVALUATION.md). The architecture and the contract for adding an exercise are in [Architecture](docs/ARCHITECTURE.md).

## Research layer

The product is one proving ground for a larger question: can a compact causal decoder read continuous body-pose signal natively, without an English or JSON intermediate, and support transferable temporal reasoning? The documents below record that work as it happened. Numeric claims carry an evidence label (MEASURED, SOURCED, DERIVED, HYPOTHESIS or UNKNOWN) defined in the [evidence protocol](docs/EVIDENCE_PROTOCOL.md).

- [North star](docs/BODY_NATIVE_INTELLIGENCE_NORTH_STAR.md): the thesis, six proving domains, curriculum and proof ladder.
- [Canonical build](docs/NATIVE_BODY_DECODER_CANONICAL_BUILD_2026-08-07.md): mathematical formulation, capability stages and training ladder.
- [Prior art](docs/COMPACT_CONTINUOUS_POSE_DECODER_PRIOR_ART_2026-08-07.md) and the [streaming skeleton systems audit](docs/STREAMING_SKELETON_SYSTEMS_AUDIT_2026-08-07.md): what published systems actually do, checked against their code at pinned commits.
- [Pose-native capability audit](docs/POSE_NATIVE_CAPABILITY_AUDIT_2026.md): what pose signal can and cannot support.
- [Nexus design](docs/POSE_NATIVE_NEXUS_DESIGN_2026.md) and [streaming decoder design](docs/POSE_NATIVE_STREAMING_DECODER_2026.md): fusing pose tokens into a 25.8M-parameter causal transformer, with parameter counts and phone latency and memory budgets.
- [One-billion backbone frontier](docs/ONE_BILLION_BACKBONE_FRONTIER_2026-08-13.md): candidate 1B-class backbones, their licenses and memory envelopes.
- [First proof proposal](docs/FIRST_PROOF_PROPOSAL_2026-08-10.md), [physical AI research plan](docs/PHYSICAL_AI_RESEARCH_2026.md), [scale and data plan](docs/SCALE_AND_DATA_AUTORESEARCH_2026.md) and [why no one built this](docs/WHY_NO_ONE_BUILT_THIS_2026-08-11.md).
- [Research state, 7 August 2026](docs/RESEARCH_STATE_2026-08-07.md): a superseded snapshot kept as a record.
- [Results ledger](docs/RESULTS_LEDGER.tsv): dated numeric claims with their evidence class and source.
- [Replay demo](experiments/e0-native-fusion/replay_demo_v0/README.md): a self-contained page that replays a rank-8 LoRA adaptation of LFM2.5-1.2B against UI-PRMD ground truth, failures included.

The raw GPU experiment archive (checkpoints, arrays and logs, about 14 GB) is not in this repository. Each experiment's report, manifest and checksums are.

## Run it

Prerequisites: Node.js 22.13 or newer and a current browser. Camera access needs `localhost` or another secure context.

```bash
npm install
npm run dev
```

Open the local address the development server prints. The fastest complete path is **Try the guided demo**: review its three exercises, confirm the plan, choose **Guided replay**, run the setup check and start. The replay uses simulated landmarks but passes them through the same geometry, confidence, constraint, coaching, set and summary code as the live camera. [Product demo](docs/PRODUCT_DEMO.md) walks through replay, live camera, upload, manual configuration, history and export.

For the personalized hard-limit check, enter or load:

> Right heel slides, three sets of ten, hold for two seconds, and do not bend my knee beyond 90 degrees.

Then switch camera setup to **Guided replay** and choose **Approach maximum** or **Reach maximum**. For live use, keep **Live camera** selected, place the phone in landscape about 6 to 8 feet away, confirm its support and the selected side, keep the requested joints visible, allow camera access and hold still through the stability check.

Production build and server:

```bash
npm run build
npm run start
```

## Verify it

```bash
npm run lint
npm run typecheck
npm test
npm run test:ml
npm run build
```

`npm run test:all` runs all five. The Vitest suite has 27 test files and the temporal model has its own Node test file. The last recorded full run, with lint, the type check, both test suites and the production build passing, was on 2026-07-27; the suite has grown since, so run it again before relying on it.

Focused experiments:

```bash
# Deterministic projection and landmark-noise sensitivity
node experiments/geometry/angle-sensitivity.mjs

# Re-run the checked-in temporal training and evaluation pipeline (Python 3.11)
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

The pose benchmark downloads checksum-verified public clips and model assets. Its raw downloads and mutable latest-results directory are not committed.

## Repository map

| Path | Purpose |
|---|---|
| `app/` | Routes, including the consented research-capture, clock-calibration and review tools |
| `src/domain/` | Confirmed session and generic constraint contracts |
| `src/intelligence/` | Local instruction parsing and upload extraction |
| `src/motion/` | Pose providers, body-feature extraction, confidence, constraints, temporal edge and exercise engine |
| `src/session/` | Structured results, summaries, export and local history |
| `src/research/` | Research data-capture protocol and annotation review |
| `tests/` | Product journey, interpretation, engine, replay, constraint, pause and resume, research-capture and summary tests |
| `ml/` | Reproducible synthetic temporal-model pipeline, model card, experiment report and measured outputs |
| `experiments/` | Pose-provider benchmark, geometry sensitivity, temporal sampling and synthetic-data audits, replay demo |
| `docs/` | Architecture, evaluation, limitations, demo script and the research documents listed above |
| `public/models/` | Locally served pose and temporal model artifacts |
| `worker/` | Cloudflare Worker entry |

## Privacy and local-first behavior

- Live frames are processed in browser memory; Relay does not persist or upload continuous camera video.
- The upstream MediaPipe runtime may report performance and utilization metrics to Google under its [privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice). Relay does not send raw frames; a formal production network audit is still required.
- PDF text extraction and photo OCR run in the browser. Photo OCR downloads Tesseract worker, core and English language assets on first use; Relay does not intentionally upload the selected image. The interpreted text must be reviewed before confirmation.
- Confirmed plans and structured results stay in browser storage until the user deletes them, at most 50 session records. If storage is unavailable, the summary still opens and asks the user to export.
- Export and share are user initiated and contain structured JSON, not camera video.
- A service worker caches core application and pose assets after they have loaded. That improves resilience after first use; it is not a promise that a first visit, or first-use photo OCR, works offline.

## Status

Relay is a technical MVP, not a medical device. The most important next experiment is to record consented right heel slides around 70 to 100 degrees on representative iPhones and Android phones, synchronized to an independent electrogoniometer or optical reference, varying camera yaw, distance, clothing, lighting, occlusion and participants, with the pipeline frozen first. It should measure signed and peak angle error, selected-leg identity failures, confidence-gate abstention, missed or late 90-degree warnings, rep and hold errors, and end-to-end spoken-cue latency. That is the shortest path from a credible technical MVP to evidence about whether its most safety-sensitive behavior is usable.

Built by [Aditya Saxena](https://github.com/asaxena9). No open-source license has been applied yet, so the code is shared for reading and evaluation; contact the author about other uses.
