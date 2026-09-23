# Capacity-data autoresearch amendment

Date: 2026-08-07  
Status: living research contract; does not modify or retroactively relax a frozen preregistration

All linked primary sources in this document were accessed on 2026-08-07.

## Decision

The architecture search envelope is expanded to **350M total active parameters**.
This is a research ceiling, not a preferred model size, a product requirement, or
evidence that scale will help. The deployable model remains the smallest model on
the real accuracy-calibration-transfer-latency-energy Pareto frontier.

The central research hypothesis remains **sub-150M pose-native causal physical
event decoding**. Models below 1M are controls for testing whether attention,
capacity, or pretraining is necessary; they are not the intended embodiment of
the hypothesis. Models from 150M to 350M are teacher, transfer and capacity-ceiling
tests unless measurement moves them onto the deployable Pareto frontier.

The experiment is not a one-dimensional march through parameter counts. It is a
capacity-data-deployment surface:

\[
M_k=M_k(P,N_o,N_e,N_s,N_d,N_c,Q,C),
\]

where \(P\) is active model capacity, \(N_o\) optimization tokens, \(N_e\)
independent physical episodes, \(N_s\) people/sessions/sites, \(N_d\) deployment
devices/views, \(N_c\) capability-specific labels, \(Q\) the source/mixture and
\(C\) measured compute. Repeated windows and epochs increase \(N_o\), not the
independent-support variables.

The current REHAB24-6 materialization is **Stage-0 falsification data**, not the
research dataset and not the intended training ceiling. Its measured 64 recordings,
10 people, 1.7101 accepted hours, 61,564 10 Hz observations and 1,057 repetitions
can test implementation, causality and a narrow boundary hypothesis. They cannot
support a 350M capability claim, clinical semantics or cross-domain generality.

## Adversarial result

The ceiling is worth testing for four reasons:

1. it admits motion-pretrained candidates excluded by the former 150M envelope,
   especially the reported 238M MotionGPT3 backbone;
2. it can provide a privileged offline teacher for distillation;
3. it lets the program distinguish a capacity plateau from a data, representation
   or optimization plateau;
4. contemporary 350M sequence models give a meaningful upper-bound systems test.

The ceiling is not justified by a monotonic scaling assumption:

- [MG-MotionLLM](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_MG-MotionLLM_A_Unified_Framework_for_Motion_Comprehension_and_Generation_across_CVPR_2025_paper.pdf)
  reports that its 220M configuration was generally stronger than 60M, while 770M
  did not consistently improve and degraded some tasks on 14,616 HumanML3D
  motions.
- [USDRL](https://arxiv.org/abs/2508.12586) reports 67.7M, 94.9M and 113.9M
  skeleton models; reported recognition peaked at the middle size.
- [MotionLM](https://arxiv.org/abs/2309.16534) reports overfitting in its larger
  trajectory-decoder configuration.
- [LLaMo](https://arxiv.org/abs/2602.12370) motivates continuous causal motion
  latents and modality-specific routes, but its smallest model is far beyond this
  ceiling and its streaming result is A100 motion generation rather than live
  phone-pose interpretation.

These are `SOURCED` warnings against assuming that more parameters produce more
physical understanding.

## Novelty target: empirical system, not assumed new mathematics

The broad claim "skeleton or body pose plus a transformer/LLM" is already
falsified as novel by motion-language models and by
[SkeletonLLM](https://arxiv.org/abs/2603.18003), which renders skeleton sequences
into an MLLM's visual modality and uses reasoning distillation. Similarly, causal
transformers, event likelihoods, distillation and abstention are established
methods. No current result in this repository establishes new mathematics.

The surviving claim is narrower and empirical:

> Can a sub-150M model directly ingest timestamped body/pose/confidence tokens,
> maintain causal state over an untrimmed physical session, and emit calibrated,
> exactly-once structured events and `WAIT/UNKNOWN` decisions on a phone—while
> outperforming equally supported modular, graph-temporal, TCN, GRU and SSM
> systems and transferring beyond the initial PT task?

As of 2026-08-07, no located primary source proves that full conjunction. This is
an `UNKNOWN` novelty hypothesis, not a first-claim. It can become a publishable
systems/task contribution in the same general sense that
[YOLO-World](https://arxiv.org/abs/2401.17270) was significant through a new
problem formulation, fusion/training method, large-scale pretraining and measured
accuracy-speed result rather than through inventing every primitive. Merely
projecting pose into an existing small LLM is insufficient. The contribution must
survive the full matched tournament, data/label audit, causal interventions,
external evaluation and sustained edge measurement.

## Parameter and system accounting

Every model report must separate:

\[
P_{system}=P_{perception}+P_{motion\ tokenizer}+P_{temporal}
+P_{output},
\]

and record total, non-embedding, active-per-step, trainable and adapter parameters.
A large text vocabulary can dominate a headline count without providing temporal
capacity. A model is not called a 350M on-device system while omitting its pose
estimator, motion tokenizer or output modules.

For 350M dense weights, storage alone is `DERIVED` as approximately 700 MB BF16,
350 MB INT8 or 175 MB INT4, before quantization metadata, activations, KV/state,
camera buffers and the pose model. For a cached dense body with \(P_b\) active
non-embedding parameters at observation rate \(f\), a first-order work estimate is

\[
C_{stream}\approx fP_b\ \text{MAC/s},
\]

before attention-context work and perception. This calculation is a planning
bound, not a phone measurement. All deployment claims require the complete
camera-to-event pipeline on the intended hardware under sustained thermal load.

## Two experiments, not one confounded leaderboard

### Controlled capacity scaling

Use a nested same-family configuration with the same representation, causal
contract, target, data exposure and optimizer search policy. This estimates a
capacity effect. Report both:

- equal independent episodes and optimization-token exposure;
- equal measured training and inference work.

Parameter-matched and compute-matched comparisons answer different questions.

### Pretrained system tournament

Compare available checkpoints as systems. This estimates transfer value but
confounds architecture, pretraining corpus and scale. At each useful tier include
random initialization, text initialization and motion initialization when they are
available. Text fluency must not be reported as motion knowledge.

Nominal tiers, whose exact counts are frozen per campaign, are:

| Tier | Active-parameter region | Purpose |
|---|---:|---|
| P0 | 0.1-0.3M | matched null/MLP/DeepSets/TCN/GRU/SSM/decoder floor |
| P1 | 1-4M | compact specialist families |
| P2 | 5-10M | Ember-scale and motion-specialist controls |
| P3 | 20-45M | Nexus-scale and MotionBERT-scale representation tests |
| P4 | 60-110M | motion-language and skeleton-foundation candidates |
| P5 | 120-250M | text controls and strongest sub-250M motion teachers |
| P6 | 300-350M | research ceiling and potential privileged teacher |

P0-P3 retain strong TCN/GRU/SSM/graph-temporal alternatives. P4-P6 retain the
best transformer and best non-transformer lineage that survived lower tiers. A
decoder-only assumption must earn its place.

Current high-value candidates, pending immutable checkpoint and license audits:

| Candidate | Reported scale | Admissible initial role |
|---|---:|---|
| [MotionBERT Lite/full](https://github.com/Walter0807/MotionBERT) | about 16M/42M | pose/motion representation teacher, not native causal decoder |
| [MG-MotionLLM small/base](https://github.com/BizhuWu/MG-MotionLLM) | 60M/220M | discrete motion-language transfer and offline teacher |
| [USDRL](https://github.com/wengwanjiang/FoundSkelModel) | 94.9M preferred reported configuration | skeleton/action representation teacher and causal early-action baseline |
| [MobileLLM](https://arxiv.org/abs/2402.14905) | 125M | deep-thin/GQA mobile decoder architecture and text-init control; not motion evidence |
| [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 135M | permissive text-initialization control only |
| [MobileLLM-R1 base/final and R1.5](https://huggingface.co/collections/facebook/mobilellm-r1) | 140M | base-, reasoning-SFT- and on-policy-KD initialization controls; text reasoning is not motion reasoning; FAIR-NC |
| [MotionGPT3](https://arxiv.org/abs/2506.24086) | 238M reported backbone | highest-priority continuous motion-language teacher/init ablation |
| [MobileLLM-Flash](https://arxiv.org/abs/2603.15954) and [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-350M) | about 350M | edge-oriented architecture/text-init ceiling, not motion-pretrained evidence |

Checkpoint availability, exact system parameters, input schema, license and
commercial rights are `UNKNOWN` until archived in a data/model admission record.
[SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) is a 3B model, not a
sub-150M candidate; quantizing its weights does not change its parameter count.

The 125-140M initialization experiment is factorial rather than a popularity
contest: identical pose adapter, causal body and objective under random init,
MobileLLM/SmolLM text init, MobileLLM-R1 reasoning init, motion init, and combined
motion-plus-text continued pretraining where legal. This distinguishes generic
sequence optimization, text/schema knowledge, explicit reasoning post-training
and motion knowledge. A result on math, code or API benchmarks cannot substitute
for a held-out physical-motion result.

## Persistent data-intelligence and admission agent

The program needs a continuously operating data agent, but not an indiscriminate
downloader. Its state machine is:

```text
discover
  -> rights and provenance
  -> immutable source/hash/archive
  -> raw numeric and stratified visual audit
  -> label/construct/observability audit
  -> capability mapping
  -> canonical conversion with ancestor lineage
  -> shortcut and marginal-utility probe
  -> ALLOWED | QUARANTINED | PROHIBITED AS SOLE EVIDENCE | REJECTED
  -> curriculum router
```

Each admitted physical episode becomes a lineage-tracked `MotionEpisode` with:

- immutable source, license snapshot, checksum and archive member;
- ancestor motion and every synthetic/augmentation/teacher transform;
- participant, session, site, device, camera and physical-episode group;
- timestamps, real sample rate, missingness and coordinate system;
- RGB/depth/IMU/force/2D pose/3D pose/rotation/contact fields with confidence;
- joint schema and explicit masks for unavailable fields;
- label source, temporal resolution, rater, disagreement and observability;
- allowed and prohibited objectives, claims and redistribution paths.

A 33-to-17/22/25-joint mapping does not fabricate rotations, root trajectory,
contacts or force. Missing variables remain masked or separately estimated and
evaluated.

The agent tracks separately:

\[
N_{optimization\ tokens},\quad N_{frames},\quad N_{episodes},\quad
N_{people},\quad N_{sites/devices},\quad N_{event/error\ families}.
\]

Its utility test is capability-specific. For source \(j\), a bounded pilot
estimates the held-out real-target change

\[
\Delta_j=M(\text{base}+j)-M(\text{base})
\]

with participant/site-clustered uncertainty and compute cost. This is evaluated
on a development panel; the external test remains sealed. Gradient alignment and
embedding diagnostics may explain a result but cannot admit a source alone.

## Data routes by learning stage

Data fields do not automatically flow into every stage.

| Route | Required signal | What it may establish |
|---|---|---|
| representation pretraining | admitted unlabeled pose/mocap/video, timestamps, masks | kinematic prediction, invariance/equivariance and robustness transfer |
| motion-language alignment | genuine paired motion/captions or audited temporal descriptions | semantic alignment, not clinical correctness |
| event SFT | real untrimmed timestamped boundaries/state labels | phase, attempt, partial, rest, restart, set and unknown semantics |
| error/quality SFT | localized expert labels, disagreement and observability | only the named, observable assessment targets |
| distillation | paired privileged teacher and deployable student observations | teacher geometry on observable targets, not missing sensor information |
| preference/reward | independently rated matched alternatives | ordering or reward model for the rated decision, not raw perception truth |
| offline RL | logged belief/action/propensity/outcome/cost trajectories | constrained intervention policy only where action support exists |

The initial discovery queue is deliberately broad:

| Discovered source | Sourced published scale | Candidate route, not an admission decision |
|---|---:|---|
| [HumanML3D](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) | 14,616 motions, 28.59 hours | motion tokenizer and motion-language pilot |
| [Motion-X](https://papers.neurips.cc/paper_files/paper/2023/hash/4f8e27f6036c1d8b4a66b5b3a947dd7b-Abstract-Datasets_and_Benchmarks.html) | 81.1K sequences, 15.6M frames | generic whole-body representation pretraining |
| [Motion-X++](https://arxiv.org/abs/2501.05098) | 120.5K sequences, 19.5M poses with multimodal subsets | multimodal alignment and observation-corruption studies |
| [MotionMillion](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Go_to_Zero_Towards_Zero-shot_Motion_Generation_with_Million-scale_Data_ICCV_2025_paper.html) | over 2,000 hours and 2M automatically extracted sequences | large generic pretraining only after provenance, rights and signal audit |
| [FineMotion](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_FineMotion_A_Dataset_and_Benchmark_with_both_Spatial_and_Temporal_ICCV_2025_paper.html) | about 442K snippets and 95K detailed descriptions | fine temporal/body-part semantic alignment |
| [MotionGB](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.html) | 400 verified hours expanded to 10,000 augmented hours | augmentation/physics hypothesis; ancestor support remains 400 hours |
| MM-Fit, Fit3D/AIFit and rehabilitation corpora | heterogeneous | continuous exercise, multimodal bridge and clinical pilots after separate audits |
| first-party continuous capture | not yet measured | target event/error/observability/device evidence |
| industrial task capture or datasets with tool/object/quality signals | not yet audited | cross-domain transfer; pose alone is insufficient for process quality |

These sources remain `DISCOVERED`, not admitted. Generic motion may teach
kinematics and semantics; it cannot by itself establish PT errors, industrial
efficiency, safety or monocular observability. Automatically extracted sequences,
captions and synthetic expansions inherit their source and teacher limitations.

## Coupled data stages

| Data stage | Required content | Allowed model work |
|---|---|---|
| D0 | current audited REHAB24-6 boundary corpus | all tiers may run bounded development probes; only narrow D0 boundary/optimization/mechanistic conclusions are admissible, never D3-D6 capability claims |
| D1 | broad admitted motion/mocap/skeleton/video plus calibrated synthetic trajectories | self-supervised causal representation scaling through P6 |
| D2 | synchronized deployment phone/RGB/pose plus multiview/mocap/IMU reference | observation-loss, pose-corruption and privileged-teacher experiments |
| D3 | real untrimmed PT sessions with partial/abort/pause/restart/set/transition/unknown | claimed persistent event decoder and capacity-data surface |
| D4 | localized errors, severity, protocol, observability and rater disagreement | assessment, abstention and biomechanics targets that pass observability |
| D5 | logged interventions, propensities, outcomes and safety/nuisance costs | preference learning and conservative offline policy evaluation |
| D6 | sealed external site/device/patient and separately audited industrial domain | external replication and cross-domain transfer claim |

The model and data agents choose deeper allocations by expected decision value,
but the first-pass observation grid is deliberately broad. Inadequate data limits
the claim attached to a cell; it does not forbid measuring optimization,
overfitting, gradient behavior, representation geometry, throughput or failure at
that scale. Such cells are labeled `MEASURED-DEVELOPMENT-PROBE` and cannot be
promoted as real-session capability.

## Promotion and stop logic

1. **Observability:** if synchronized reference shows that deployed input cannot
   distinguish the target, add RGB/view/sensor, weaken the target or abstain.
2. **Temporal signal:** the frozen P0 campaign estimates a D0 comparison; it is not
   a universal permission gate. Sparse predeclared pilots at larger tiers and on
   other data/objectives may run concurrently. If ordered models do not beat
   null/nonordered controls on D0, reject the D0 attribution and investigate the
   representation/data/task interaction; do not infer that D3 semantics or every
   larger architecture is impossible.
3. **Pretraining:** promote a source/objective only when it improves held-out real
   D3 transfer, calibration or low-shot efficiency—not source reconstruction.
4. **Capacity:** increase tiers only when the larger same-family model improves a
   real clustered metric beyond evaluator/rater noise without violating false
   events, rare-event recall, calibration or mechanism interventions.
5. **Transfer:** scale must improve unseen-exercise/domain low-shot area under the
   learning curve, not merely familiar-source accuracy.
6. **Structured generation:** compare direct fields with constrained AR fields on
   the identical state. If direct heads tie, use deterministic serialization.
7. **Distillation:** a P5/P6 teacher matters only if it wins on real D3/D4 and
   improves the same deployable student over a non-distilled control.
8. **Edge:** choose from complete sustained camera-pose-model-UI measurements;
   kernel-only A100 or text-generation numbers cannot pass.
9. **RL:** do not introduce it automatically after SFT. It requires D5 support,
   reward-hacking checks, conservative off-policy evaluation and a non-RL
   baseline.

The 350M tier is retired if it neither improves external real transfer over the
best smaller model nor improves a distilled student.

## Required mechanistic and efficiency suite

Every promoted neural family must pass future-suffix invariance, cached/full
equivalence, chunk invariance, pose removal and record derangement, history-reset
and history-length curves, reversal/shuffle, active-joint versus matched-control
deletion, boundary-proximal versus remote deletion, timestamp/confidence removal,
exercise-ID removal and counterfactual protocol changes.

Record token rates and physical seconds/context, total/active/trainable
parameters, optimizer/activation/KV memory, measured FLOPs/utilization, per-loss
gradient norms/cosines, representation effective rank, attention entropy/distance,
MLP activation selectivity, logit margins and calibration. Attention plots alone
are not causal evidence.

## Automated execution contract

GPU work runs from immutable isolated job specifications with an explicit wall
and GPU-hour ceiling. Every job writes a heartbeat and resumable state containing
model, optimizer, scheduler, scaler, RNG and data-order counters. Atomic artifacts
include raw predictions, metrics, logs, environment, peak memory and terminal
`PASS`, `REJECT`, `INCONCLUSIVE`, `FAILED` or `TIMEOUT` state.

Three A100s may run independent cells concurrently; they never share output
directories. `last-recovery` and `best-validation` checkpoints are distinct.
Checkpoint reload is verified before the original session terminates. Rejected
weight files need not be retained indefinitely, but their hashes, manifests,
metrics and failure logs remain in the append-only ledger. One-shot Colab runs
must release automatically and must never rely on an untouched interactive
notebook.

## Measured execution state before the next campaign

- The present local model tests pass 39/39 on the audited environment.
- The complete Stage-0 matrix has not run. Its intended neural matrix is six
  models by ten outer participants by three seeds, or 180 cells, plus nulls.
- The default runner expects a missing `stage0-execution-v1.lock.json`; the
  available older campaign lock and stored smoke identify a different
  `native_decoder_models.py` hash than the current source.
- The project root has no Git metadata. Source archives and tree hashes are
  therefore required for immutable runs.
- The current runner lacks mid-cell resumable model/optimizer/data-order state;
  a terminated Colab session can lose an entire cell.
- The current evaluator does not yet persist every preregistered calibration,
  intervention, profiler, prediction and manifest artifact.
- Ember training and Nexus training are not implemented in this repository.
  Nexus has only a random-input A100 runtime probe, the clean Nexus base is
  absent, and Ember checkpoint provenance/rights remain unresolved.

These are measured engineering blockers, not evidence against the research
hypothesis. They must be repaired before GPU hours are interpreted scientifically.

## Immediate order

These are parallel workstreams, not sequential permissions:

1. repair/re-freeze Stage-0 and execute its complete participant/seed matrix;
2. add orchestration, resume, profiling, calibration and intervention artifacts;
3. build the persistent data-admission ledger and begin D1 source audits;
4. run a sparse, bounded D0 implementation/optimization probe across the credible
   6M, 25M, 60M, 95M, 125--150M and teacher-scale families, with D0-only claims;
5. materialize canonical broad-motion pretraining data and run the same sparse
   capacity/family grid on causal self-supervision and transfer probes;
6. design and collect the synchronized D2 bridge and real D3 continuous-event
   corpus;
7. deepen allocations where the capacity-by-data surface has positive information
   value, while retaining an equal-budget finalist comparison;
8. test P5/P6 teachers and distillation when their target is observable, then
   measure the complete deployment Pareto frontier.

The resulting research question is:

> Is continuous physical-event understanding capacity-limited, data-limited,
> representation-limited or observability-limited, and can admitted physical data
> plus causal pretraining move a 0.1M-350M model family onto a better real
> accuracy-calibration-transfer-energy frontier?
