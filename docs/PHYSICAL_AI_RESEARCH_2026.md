# Relay physical-AI architecture and research plan

**Evidence cutoff:** 2026-08-07  
**Scope:** monocular phone video, structured pose, continuous exercise understanding,
rep/set logic, corrective feedback, and ESP32-class deployment.

## Bottom line

This is possible, but the strongest system is not one 25M decoder watching raw
frames and emitting prose continuously. It is a hierarchical, streaming system:

```text
camera (15-30 fps)
  -> pose estimator (10-15 Hz)
  -> normalized body/motion tokens
  -> tiny causal temporal encoder (10-15 Hz)
  -> constrained state decoder + calibrated safety rules
  -> rep/set event record
  -> optional 25M structured-language decoder (only on events)
  -> validated cue or abstention
```

The temporal encoder answers *what is happening now and what just completed*.
The optional language decoder answers *which approved explanation or cue fits the
structured evidence*. This separation is both computationally efficient and much
more scientifically defensible than a small general-purpose VLM. A 2026 empirical
evaluation found that even large vision-language models were only marginally above
chance on several action-quality tasks and that pose/prompting additions did not
produce consistent gains ([Monte e Freitas et al., 2026](https://openaccess.thecvf.com/content/CVPR2026W/SAUAFG/html/Monte_e_Freitas_Can_Vision_Language_Models_Judge_Action_Quality_An_Empirical_Evaluation_CVPRW_2026_paper.html)).

Relay already has the correct seed: an abstaining 2,351-parameter causal temporal
model over a 2.4-second pose-feature window, with the deterministic exercise engine
retaining authority. The next version should preserve that interface while replacing
synthetic-only training and adding progress, boundary, error, and uncertainty heads.

## The questions the system must answer

1. Is the prescribed exercise occurring, or is this background/repositioning?
2. What phase is active: idle, outbound, endpoint/hold, return, paused, or aborted?
3. How far through that phase is the person, and did a valid phase boundary occur?
4. Has a complete rep occurred, or only a partial, oscillatory, or interrupted motion?
5. Which visible deviations occurred, during which phase, and with what confidence?
6. Which signals are unobservable because of view, occlusion, or pose failure?
7. Did the person pause within a set, finish a set, or explicitly restart it?
8. Which response is allowed: count, cue, ask to reposition, or abstain?
9. Does a learned temporal model add value beyond geometry, hysteresis, and a small
   classical classifier at the same false-warning rate?

The key product decision is that a set identity should not be inferred silently from
body motion alone. Long inactivity can propose a set boundary, but `start next set`,
`restart set`, and `end exercise` should be explicit UI/voice events. The model can
recognize motion; it cannot reliably infer the person's intent.

## How continuous pose becomes model input

At timestamp \(t\), a pose provider returns \(J\) joints:

\[
P_t = \{(x_{tj}, y_{tj}, z_{tj}, v_{tj})\}_{j=1}^{J}
\]

where \(v\) is visibility/confidence. Construct a side-canonical, root-relative,
scale-normalized vector with velocity and acceleration:

\[
X_t = [\tilde P_t,\; \tilde P_t-\tilde P_{t-1},\;
       \Delta^2\tilde P_t,\; A_t,\; D_t,\; V_t,\; M_t]
\]

Here \(A_t\) contains selected joint angles, \(D_t\) body-segment relationships,
\(V_t\) visibility statistics, and \(M_t\) an explicit missing-data mask. Timestamps,
not frame indices, determine derivatives. Resample or encode \(\Delta t\) so dropped
camera frames do not look like faster movement. Do not zero-fill a missing joint
without its mask.

Use two representations in the phone research model:

- an interpretable feature branch (angles, relative vectors, confidence, derivatives);
- a learned skeleton branch over all joints, using a small graph/MLP encoder.

Fuse their embeddings, rather than forcing every judgment through hand-engineered
angles. Graph models encode anatomical adjacency; PoseConv3D instead rasterizes
joint heatmaps into a spatiotemporal volume and has shown stronger noise robustness
and cross-dataset generalization in general skeleton recognition
([Duan et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html)).
For Relay's few known exercises, a small graph/MLP plus TCN is easier to stream and
deploy than PoseConv3D.

## Recommended model: a causal multi-task temporal encoder

Start with a residual causal TCN, not a language transformer:

\[
H_t^{(0)} = \phi(X_t), \qquad
H_t^{(l+1)} = H_t^{(l)} +
  \sigma\!\left(W_l *_{{d_l}} H_{\le t}^{(l)}\right)
\]

with dilation \(d_l \in \{1,2,4,8,16\}\). This produces a receptive field long
enough for slow reps while remaining causal. A GRU or a causal state-space model is
a valid alternative:

\[
h_t = \bar A_t h_{t-1} + \bar B_t X_t, \qquad z_t = C_t h_t
\]

but TCNs have mature quantized operators and are the lowest-risk first build. Recent
skeleton SSM work is promising for efficient long contexts, but its benchmark wins
do not establish rehabilitation-quality assessment
([SkelMamba, 2024](https://arxiv.org/abs/2411.19544)).

Attach separate heads to \(z_t\):

\[
\begin{aligned}
p_t^{phase} &= \mathrm{softmax}(W_p z_t) \\
p_t^{exercise} &= \mathrm{softmax}(W_e z_t) \\
b_t^{start}, b_t^{end} &= \sigma(W_b z_t) \\
q_t^{progress} &= \sigma(W_q z_t) \\
p_t^{error} &= \sigma(W_r z_t) \\
\hat u_t &= W_u z_t
\end{aligned}
\]

The outputs are phase, exercise/background, boundary probabilities, continuous phase
progress, multi-label visible errors, and uncertainty. Progress is important because
online action-segmentation work finds that causal models lose accuracy when simply
ported from offline inference; progress and learned task graphs reduce uncertainty
and over-segmentation ([Shen and Elhamifar, 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html)).
For cyclic exercises, predict \((\cos\phi_t,\sin\phi_t)\) as an auxiliary head rather
than a scalar phase alone; this removes the artificial discontinuity between the end
of one cycle and the start of the next. Count-density \(\rho_t\ge0\), constrained so
each valid cycle integrates to approximately one, is another useful auxiliary signal,
but it should not bypass the legal-state decoder.

Train with a multi-task objective:

\[
\mathcal L = \lambda_p\mathcal L_{phase}
 + \lambda_b\mathcal L_{boundary}
 + \lambda_q\mathcal L_{progress}
 + \lambda_r\mathcal L_{error}
 + \lambda_c\mathcal L_{cycle}
 + \lambda_s\mathcal L_{smooth}
 + \lambda_u\mathcal L_{calibration}
\]

Use focal or class-balanced loss for rare boundaries/errors; Huber loss for progress;
a truncated smoothness penalty on adjacent log-probabilities; and supervised
contrastive/cycle consistency so corresponding phases align across different tempos.
The 2026 PAMS-TCC result is especially relevant: self-supervised, multi-scale cycle
consistency on skeleton sequences achieved state-of-the-art RepCount/Countix results,
showing that large quantities of unlabeled repetitive motion can teach tempo-invariant
phase structure ([Gao et al., 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html)).

### The constrained state decoder

Neural frame labels should not count reps directly. Decode them through an allowed
state graph:

```text
IDLE -> READY -> OUTBOUND -> ENDPOINT/HOLD -> RETURN -> READY
                   \-> ABORTED <-/           \-> PAUSED
```

Maintain an online state score

\[
S_t(s)=\log p_t^{phase}(s)+
       \max_{s'\in pred(s)}[S_{t-1}(s')+\log A_{s's}]
       +D(s,\tau_s),
\]

where \(A\) forbids impossible transitions and \(D\) is a learned or configured dwell
time prior. This can be an online Viterbi/semi-Markov decoder or a simple auditable
state machine over calibrated probabilities.

A rep completes only if all are true:

1. READY was observed with sufficient tracking quality;
2. OUTBOUND persisted for a minimum dwell time;
3. endpoint evidence or exercise-specific progress exceeded its learned/calibrated
   criterion;
4. RETURN occurred in the correct direction;
5. READY was reacquired for a minimum dwell time;
6. uncertainty and missingness remained within policy.

This rejects bouncing near a threshold, partial reps, a pose jump after occlusion, and
moving the wrong direction. Keep a provisional rep until step 5; show progress but do
not increment the authoritative count early.

## Where a 25M Nexus/Ember-style decoder belongs

There are three meanings of “decoder,” and conflating them causes the architecture
confusion:

1. **Temporal state decoder:** converts frame evidence into a legal state sequence.
   It can be non-neural and is essential.
2. **Discriminative neural decoder:** the small phase/boundary/error heads above.
   This runs continuously.
3. **Autoregressive language decoder:** emits tokens. This should run only when a rep,
   set, or exception event is finalized.

For rep \(i\), pool the temporal states inside its interval into a compact record:

```json
{
  "exercise": "heel_slide",
  "side": "right",
  "duration_ms": 4100,
  "peak_flexion_deg": 82,
  "phase_durations_ms": [800, 1400, 900, 1000],
  "visible_errors": [{"type": "hip_rotation", "confidence": 0.82}],
  "tracking_coverage": 0.96,
  "trend_vs_prior_reps": "improving"
}
```

Convert continuous embeddings to decoder-width tokens with an adapter
\(E_i=W_a z_i+b_a\). Then either:

- prepend 4-16 learned motion tokens to the text/schema tokens; or
- let decoder hidden states cross-attend to motion tokens:

\[
\mathrm{Attn}(Q_{text},K_{motion},V_{motion})=
\mathrm{softmax}(QK^T/\sqrt d)V.
\]

Train the decoder to emit a constrained assessment schema, not unconstrained medical
advice. A deterministic policy maps that schema to clinician-approved cue templates.
The generated surface text may vary only inside approved semantic slots. For a first
paper, compare this model with template retrieval; the decoder earns its place only if
it improves error selection, prioritization, or personalization at equal safety-event
recall and calibration.

Do not run a 25M model once per pose frame. At 10 Hz that repeatedly solves the same
problem, increases heat, and couples safety latency to token generation. Run the
continuous encoder on every accepted pose; invoke the language decoder after a rep or
at a rate-limited high-confidence exception.

## Dataset: what matters most

Public data is useful for pretraining and protocol design, but it does not validate
Relay's three exercises on phones:

| Source | What it contributes | Main gap |
|---|---|---|
| FineRehab | 4,215 samples, 16 actions, 50 healthy/patient participants, two Kinect views and 17 IMUs | Different capture/product domain; license/access must be checked | 
| Clinical low-back-pain dataset | Patients, RGB, 2D/3D skeletons, medical annotations with error body-part and time span | Different exercises and sensor setup |
| KIMORE/UI-PRMD | Rehabilitation quality scores and correct/incorrect performances | Small/lab-centered; label and camera mismatch |
| IntelliRehabDS | 2,589 per-rep skeleton files from 29 healthy/patient subjects, with two-annotator correctness labels | Skeleton-only, imbalanced, small |
| UCOPhyRehab++ | 2,160 five-view sequences, 16 exercises, per-rep boundaries and two-physiotherapist scores | 27 subjects; controlled capture; RGB access restrictions |
| Fit3D/AIFit | Accurate multiview 3D motion, at least five annotated reps per sequence, exercise segmentation and localized feedback | Lab/trainer domain; access and use terms require audit |
| MM-Fit | More than 800 minutes of continuous, synchronized video/pose/wearable exercise data | Fitness rather than clinical quality |
| M3GYM | 47M frames, 8 cameras, 50+ people, 500+ expert-assessed actions, occlusion and lighting variation | Gym rather than home rehabilitation |
| Rehab-Pile | Unified 2025 benchmark across rehabilitation datasets | Benchmark transfer is not product validation |
| RepCount/Countix | Repetition/cycle pretraining | Counts do not encode clinical form or Relay phases |

Sources: [FineRehab](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Li_FineRehab_A_Multi-modality_and_Multi-task_Dataset_for_Rehabilitation_Analysis_CVPRW_2024_paper.html),
[clinical low-back-pain dataset](https://arxiv.org/abs/2407.00521),
[IntelliRehabDS evaluation](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1878438/full),
[UCOPhyRehab++](https://www.nature.com/articles/s41597-026-07362-5),
[Fit3D](https://fit3d.imar.ro/), [MM-Fit](https://mmfit.github.io/),
[M3GYM](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_M3GYM_A_Large-Scale_Multimodal_Multi-view_Multi-person_Pose_Dataset_for_Fitness_CVPR_2025_paper.html),
and [Rehab-Pile](https://arxiv.org/abs/2507.21018).

### Annotation hierarchy

Annotate at four levels, using a fixed ontology and explicit `unobservable` labels:

- **frame:** pose quality, phase, progress, visible compensation/error;
- **rep:** start/endpoint/end, complete/partial/aborted, ROM, tempo, error presence and
  severity, reference quality;
- **set:** set boundaries, pauses, restarts, prescribed versus completed dose;
- **session/person:** device, view, environment, body/clinical metadata permitted by
  consent, assistance/brace, and clinician global assessment.

Use two independent annotators for boundaries/errors and a clinician adjudicator on a
stratified subset. Store label provenance and disagreement instead of collapsing all
uncertainty to one “ground truth.” For numeric angles, synchronize a gold subset to an
electrogoniometer, inertial sensor, or optical motion-capture reference.

### Collection stages

**Stage 0 — freeze and test the existing model.** Evaluate the current artifact before
retraining. This measures the true synthetic-to-human gap.

**Stage 1 — instrument/pilot (30-50 people, approximately 3,000-8,000 reps).** Capture
the three Relay exercises, correct performances and one controlled deviation at a
time, plus natural mixed deviations. Use representative iOS/Android phones, supported
and deliberately adverse views, varied tempo/holds, clothing, lighting, beds/chairs,
occlusion, partial reps, pauses, and explicit restarts. Include a gold measurement
subset. This is enough to debug the ontology and compare small models, not enough for
broad clinical claims.

**Stage 2 — credible narrow MVP (approximately 120 people and 13,000-29,000
reps).** For a six-to-eight-exercise clinical pathway, recruit approximately 60 target
patients spanning condition stage/severity and 60 matched healthy or at-risk controls;
capture three visits, six-to-ten reps, natural restarts/aborts, and three representative
phone placements. Collect synchronized multiview/3D plus IMU or other reference on
approximately 30 participants. Fully label a stratified 20-30%, weak-label the rest,
and reserve an untouched 20-30-person external-device/site test set. These are planning
numbers, not a statistical power claim; final sample size should be driven by the
frequency and required sensitivity of each error class.

**Stage 3 — product scale (200-500 people initially; eventually multi-site).** Add
more target patients, repeated sessions, device/site diversity, assistive devices,
pain-limited and atypical strategies, and naturally occurring errors. Split by
participant first; reserve untouched device/site challenge sets. Before trusting a
learned product-critical error detector, seek hundreds of distinct contributing
participants and roughly 500-1,000 positive examples for that specific error; retain
rules/abstention for rarer safety events.

**Scale annotation cheaply.** Pretrain cycle/pose representations on large unlabeled,
consented sessions. Use active learning to send uncertain/disagreeing intervals—not
random frames—to annotators. A much larger unlabeled pool is more valuable than
auto-generated prose labels.

### What simulation can and cannot do

Simulation should create articulated 3D motion, project it through randomized camera
intrinsics/extrinsics, and then pass rendered or real-background-composited frames
through the exact production pose estimator. Randomize anatomy within responsible
ranges, tempo, ROM, compensations, occlusion, clothing proxies, lighting, camera
motion, image artifacts, and dropped timestamps. Most importantly, learn a
pose-corruption model from paired human video/reference data: joint-specific bias,
jitter, confidence collapse, swaps, hallucinated recovery, and correlated errors.

Simulation is excellent for state-transition coverage, rare failure injection,
software regression, and pretraining. It cannot validate clinical correctness,
real-patient strategies, pose accuracy, subgroup fairness, cue comprehension, or
safety. Synthetic and human metrics must remain separate.

## Phone versus ESP32: realistic scale

Parameter count alone is not the limit; weights, activations, operator support,
memory bandwidth, and per-inference multiply-accumulates all matter.

| Model | INT8 weights only | INT4 weights only | Appropriate target |
|---:|---:|---:|---|
| 0.1M | 0.1 MB | 0.05 MB | ESP32 temporal model |
| 1M | 1 MB | 0.5 MB | ESP32-P4 or phone |
| 25M | 25 MB | 12.5 MB | phone; not a sensible ESP32 continuous decoder |
| 60M | 60 MB | 30 MB | modern phone with careful profiling |
| 100M | 100 MB | 50 MB | phone, but justify energy/latency over smaller models |

Those numbers exclude quantization metadata, runtime arena, intermediate activations,
camera buffers, pose model, and autoregressive KV cache. A decoder token costs on the
order of \(2P\) operations just to touch each weight once; a 25M dense decoder is at
least roughly 50M multiply/add operations per generated token before overhead. Its
memory traffic also rereads approximately 12.5-25 MB of weights per token unless a
cache hierarchy retains them.

### Phone recommendation

- Run MediaPipe Pose Landmarker or another profiled pose model asynchronously; the
  official live-stream API may deliberately drop frames to reduce latency, so the
  temporal path must use timestamps rather than assuming one result per camera frame
  ([MediaPipe API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker)).
- Run a 0.1-2M INT8 temporal model at 10-15 Hz in LiteRT/ExecuTorch/Core ML.
- Run a 25M INT4/INT8 structured decoder only at events. A 60-100M model is technically
  plausible on current phones but should be admitted only by measured accuracy,
  latency, thermal, and battery gains.
- Use Core ML on iOS and LiteRT/ExecuTorch on Android, with CPU fallbacks and real-device
  profiling. Core ML can partition across CPU/GPU/Neural Engine
  ([Apple Core ML](https://developer.apple.com/documentation/coreml/)); ExecuTorch has
  mobile backends for Core ML, XNNPACK, Vulkan, Qualcomm, MediaTek, and others
  ([ExecuTorch edge platforms](https://docs.pytorch.org/executorch/stable/edge-platforms-section.html)).

### ESP32 recommendation

Use ESP32-P4, not a classic ESP32, if the board must own a camera. ESP32-P4 has dual
400 MHz RISC-V cores, 768 KB on-chip high-performance memory, camera/ISP/JPEG/H.264
hardware, and external PSRAM support; it is not a phone-class NPU
([ESP32-P4 datasheet](https://documentation.espressif.com/esp32-p4_datasheet_en.html)).
ESP-DL supports quantized Conv/GEMM and a YOLO11n pose path, and Espressif explicitly
notes that classic ESP32 is significantly slower than S3/P4
([ESP-DL](https://github.com/espressif/esp-dl),
[getting started](https://docs.espressif.com/projects/esp-dl/en/latest/getting_started/readme.html)).
The current official ESP-DL COCO-pose table is a useful reality check: INT8
YOLO11n-pose at 640×640 takes about 27.8 seconds per image on ESP32-S3 and 2.73
seconds on ESP32-P4—approximately 0.036 and 0.37 fps
([Espressif COCO Pose](https://github.com/espressif/esp-dl/tree/master/models/coco_pose)).
By contrast, Espressif reports its domain-oriented 0.36M-parameter, 0.17-GFLOP
ESPDet-Pico at 224×224 taking 126.2 ms on S3 and 51.4 ms on P4
([ESP-Detection](https://github.com/espressif/esp-detection)). This is why resolution,
operations, and specialization matter more than a parameter count headline.

Two viable ESP32 products are:

1. **Pose sensor companion:** ESP32-P4 camera runs a very small pose estimator or
   motion gate and sends landmarks/features to the phone. The phone owns the temporal
   and language models. This is the recommended split if a phone is present.
2. **Standalone constrained coach:** ESP32-P4 runs a sub-1M pose detector/estimator,
   a sub-100K temporal model, the explicit state machine, and prerecorded/template
   cues. Do not include a 25M language model.

External 32-64 MB PSRAM may make 25M INT8 weights *fit* in some ESP32-P4 designs, but
fitting is not equivalent to usable inference: it leaves poor activation/camera
headroom and is dominated by external-memory bandwidth and general-purpose compute.
ESP-DL's documented deployment precisions are INT8/INT16, so the optimistic INT4
weight column is not a ready ESP-DL path.
If Nexus/Ember currently reaches 8-15 tokens/s, document the exact processor,
accelerator, precision, memory, context, power, and whether that number excludes pose
inference; it should not be generalized to ESP32 without a measured port.

## Experimental program that can prove the idea

Freeze data splits and compare all models at matched input and latency:

| System | Purpose |
|---|---|
| geometry + hysteresis | transparent product baseline |
| logistic/XGBoost on rep summaries | tests whether “reasoning” is just tabular classification |
| current 2,351-param TCN | frozen synthetic baseline |
| 0.1-2M causal multi-task TCN | proposed continuous model |
| temporal model + constrained state decoder | tests sequence constraints |
| previous + template retrieval | bounded feedback baseline |
| previous + 25M Nexus/Ember decoder | tests the claimed tiny reasoner |
| larger 60-100M decoder | scaling ceiling, not default product |

Primary metrics:

- phase macro-F1 and segmental F1@10/25/50, edit score, and boundary F1 with temporal
  tolerance;
- rep count MAE and off-by-one rate, plus complete/partial/aborted classification;
- per-error precision/recall and rep-level quality MAE/correlation;
- calibration error, risk-coverage curve, and performance at a fixed abstention rate;
- false-negative rate and warning latency for predeclared constraints;
- participant/device/view/site subgroup results and worst-group confidence intervals;
- end-to-end camera-to-cue latency, peak RAM, package size, battery drain, and thermal
  throttling over a full session.

The decisive claim is not “25M is intelligent.” It is:

> Given the same causal pose stream, the compact temporal-plus-structured-decoder
> system produces more correct, calibrated, context-sensitive movement judgments than
> rules and small classifiers, at a measured on-device resource budget and without
> degrading safety-event recall.

## Concrete build order

1. Version the current pose-feature/event contract and freeze Synthetic v1.
2. Extend capture annotations with progress, boundaries, visible-error intervals,
   explicit unobservable labels, set/restart events, and independent reference data.
3. Collect Stage 1 human data and evaluate the frozen model before any tuning.
4. Train a causal multi-task TCN at several widths; add the constrained state decoder.
5. Quantize and profile the full phone loop, including camera, dropped poses, UI,
   speech, heat, and battery—not only the temporal kernel.
6. Build a deterministic rep-summary and approved-cue baseline.
7. Adapt Nexus/Ember to consume motion prefix tokens and emit a strict assessment
   schema; invoke it only at rep/set events.
8. Run the full ablation and risk-coverage analysis on an untouched participant split.
9. Only then decide whether the 25M decoder is a product component, a research result,
   or unnecessary complexity.

For every streaming implementation, cache temporal state. A causal network processes
only the newly accepted pose token. Re-running a 64-frame window through a 25M dense
model on every update costs roughly 64 times more than incremental decoding; expose
TCN ring buffers, recurrent/SSM state, or a bounded transformer KV cache in the runtime
contract.

## Research boundary

None of these architectures turns monocular pose into a clinical measurement. A
single camera cannot observe pain, load, muscle activation, tissue state, or every
out-of-plane compensation. The appropriate initial claim is a camera-guided movement
assessment aid for a small, clinician-defined exercise set, with explicit
unobservability and abstention—not autonomous physical therapy.
