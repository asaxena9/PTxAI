# Body-native intelligence North Star

Status: **mandatory, durable research invariant**  
Last updated: 2026-08-12  
Applies to: every PTxAI research, data, model, evaluation, and demo task

> Build transferable body-native intelligence: continuous, time-aligned body
> signals become a causal retained state from which a model can predict, reason,
> assess, act, and communicate across tasks and domains. Physical therapy and
> every named movement are proving grounds, not the ontology.

This sentence must survive context compaction, task handoff, architecture changes,
and demo pressure. At the start of a research cycle and after compaction, read
this document, `AGENTS.md`, and
`experiments/autoresearch/RESEARCH_OPERATING_PHILOSOPHY.md` before choosing the
next experiment. A dated handoff or prior plan cannot silently narrow this object.

## 1. The research object

Let the admitted observations through time `t` be

\[
M_{\le t}=\{P,C,T,H,F,R,I,O,A,\ldots\}_{\le t},
\]

where the symbols may include body pose or pose distributions `P`, confidence and
missingness `C`, timestamps `T`, hand and face motion `H,F`, RGB/depth `R`, IMU
`I`, object/contact/force context `O`, and audio or other explicitly admitted
signals `A`. The available modalities vary by task and deployment.

With optional instruction, protocol, reference, or goal context `Q`, the central
object is a causal retained state

\[
Z_t=f_\theta(M_{\le t},Q),
\]

that supports held-out output families such as motion dynamics, temporal events,
movement semantics, grounded comparisons, uncertainty, counterfactual correction,
language, low-shot transfer, or action. No one output family is assumed to prove
the others.

The **body-native decoder** is the central architectural hypothesis: body evidence
must remain directly accessible to the model and its decoder, rather than being
replaced by a handcrafted session report. It is still falsifiable. Strong modular
encoders, graph/temporal networks, state-space models, analytic mechanics, and
task-specific systems are real competing hypotheses.

## 2. PT, sign language, and robotics are proving environments

Physical therapy is useful because it supplies repeated motion, protocols,
observable errors, temporal decisions, and consequential feedback. It is not the
representation boundary. A PT-only result proves only the measured PT task.

Other proving environments expose different parts of the hypothesis:

- sign language tests whether fine body, hand, face, and temporal signals can
  support compositional language;
- sport and fitness test dynamics, phase, comparison, and correction;
- ergonomics and industrial work test object-conditioned movement and risk;
- daily activity tests open transitions and long untrimmed context;
- human-robot demonstration tests whether body state transfers into executable
  task descriptions or policies.

Standard coarse pose is not sufficient for all of them. Signed language, for
example, may require hand shape, facial grammar, gaze, and RGB context. Force,
pain, load, intent, contact, and occluded geometry are likewise not generally
recoverable from monocular pose. Add the necessary sensor or context, weaken the
target, or abstain.

Google announced SignGemma in May 2025 as an upcoming sign-language-to-spoken-text
Gemma model, strongest for ASL-to-English. The announcement is **SOURCED** evidence
that this body-to-language problem matters, not evidence for our architecture or
for a released checkpoint. In January 2026 Google staff still gave no release
timeline; a current public model card and technical report remain **UNKNOWN**.

Primary sources:

- <https://developers.googleblog.com/en/google-io-2025-developer-keynote-recap/>
- <https://discuss.ai.google.dev/t/signgemma-release-date/113794>

## 3. Give the model the most useful admitted signals

The design rule is **raw plus aids**, not raw versus features.

1. Preserve native, timestamped body observations: coordinates or distributions,
   confidence, missingness, topology, identity, and history.
2. Add complementary measured modalities when they contribute information the
   body stream cannot contain: hands/face, RGB/depth, IMU, contact/force, object
   state, audio, or a supplied reference.
3. Add tagged derived evidence that can reduce finite-model sample and compute
   cost: angles, velocities, accelerations, symmetry, phase candidates, analytic
   facts, calibrated classifier posteriors, and retrieval hypotheses.
4. Add task context explicitly: instruction, protocol, reference trajectory,
   safety envelope, device/camera calibration, and uncertainty.
5. Give every channel a source, clock, units, mask, calibration status, and
   uncertainty. The model must be able to distinguish measurement, derived fact,
   learned hypothesis, and requested target.

If an auxiliary feature `A=g(M)` or classifier posterior `K=h(M)` is a deterministic
function of the same complete raw stream, then ideally

\[
I(Y;A,K\mid M,Q)=0.
\]

It adds no new physical information, but it can make that information cheaper for
a finite model to access. A complementary sensor `S` can add real information when
`I(Y;S | M,Q)>0`. Both are useful, but the evidence claim is different.

Do not turn aids into a report-card bottleneck. The decoder must retain a route to
raw evidence. Run raw-only, aids-only, full-signal, pose-zero/query-only, and
corrupted-aid ablations. If the full model merely repeats a classifier, credit the
classifier rather than claiming body-native reasoning.

Context is managed by aligned modality encoders, causal state, learned resampling,
addressable side memory, and a bounded number of query-relevant registers—not by
serializing every redundant scalar into the language context. What is retained or
discarded must be measured.

## 4. Exploratory 20-family curriculum

The number 20 and the membership below are **EXPLORATORY**, not a universal gate.
They create diversity early enough to resist a one-exercise classifier while data
admission determines the final surface.

1. stand, idle, and transitions;
2. walk;
3. run;
4. squat;
5. sit-to-stand and stand-to-sit;
6. hip hinge and bend;
7. forward and lateral lunge;
8. step-up, step-down, and stairs;
9. jump and land;
10. reach;
11. push;
12. pull or row;
13. lift and lower an object;
14. carry;
15. turn and whole-body rotate;
16. throw;
17. catch;
18. kneel, crawl, floor transfer, and get-up;
19. manual tool or assembly demonstration;
20. continuous signed utterance.

These are movement/task families, not twenty output labels. Each family should
contain phases, partial attempts, interruptions, speed and morphology variation,
occlusion, multiple instructions, and where meaningful both valid and invalid
counterfactuals. PT protocols become contexts over relevant families rather than
new ontological classes. The signed-utterance family requires a distinct
hand/face-capable observation adapter and culturally/linguistically valid data.

Broad representation learning can use the admitted subset of all families while
the first correction post-training surface uses only families with trustworthy
observable targets. Transfer to a held family or domain is more informative than
memorizing all twenty names.

## 5. What simulation can and cannot teach

Robot and game-control systems demonstrate a real pattern: when dynamics,
observations, actions, and reward are faithful enough, large-scale parallel
simulation can create useful policies through trial and error. Boston Dynamics,
for example, describes Atlas reinforcement learning from human reference motion
in thousands of parallel simulations and emphasizes both retargeting and the
remaining sim-to-real problem.

That does not make a human-motion simulator an automatic source of semantic or
clinical truth. We therefore separate three roles:

- **trajectory generation:** physics, inverse kinematics, search, or RL generates
  feasible movement and perturbation families;
- **analytic supervision:** known geometry and simulator state supplies exact
  observable facts, evidence spans, and counterfactual deltas;
- **semantic grounding:** real protocols, expert/reference pairs, or linguistic
  data determine what a movement means and whether a correction is appropriate.

RL is useful when a declared action/reward problem exists—for example, producing
a stable corrected trajectory under physical constraints. It is not a substitute
for supervised perception or for missing definitions of “good,” “safe,” “painful,”
or “clinically appropriate.” Synthetic data establishes simulator competence and
mechanics first; real held people and the deployed sensor path are required for a
human capability claim.

Primary precedents:

- <https://bostondynamics.com/blog/can-football-teach-a-robot-to-move/>
- <https://bostondynamics.com/blog/starting-on-the-right-foot-with-reinforcement-learning/>
- <https://arxiv.org/abs/1804.02717>

## 6. First executable proof ladder

The immediate program is intentionally broader than “recognize a squat and say
lower,” which an analytic threshold can solve.

### A. Data engine and physical oracle

- Admit real movement sources before promotion; retain subject/session ancestry.
- Generate simulator- or optimizer-based variants only for observable geometry,
  timing, contact, and gross alignment whose labels are exactly replayable.
- Produce paired raw and corrected trajectories, evidence joints/time, supplied
  targets, uncertainty, and invalid/unobservable cases.
- Audit the sim-to-real and pose-estimator gap rather than hiding it with volume.

### B. Broad representation, bounded correction surface

- Pretrain or adapt the body representation on multiple admitted curriculum
  families using causal prediction, masked reconstruction, contrastive alignment,
  or another explicitly tested objective.
- Post-train a genuine body-connected decoder first on a small set of movements
  with sound correction truth, while keeping held people, trajectories,
  instructions, and at least one movement family disjoint.
- Preserve direct body access while adding kinematics, calibrated classifiers,
  reference/protocol context, and any admitted complementary sensor.

### C. Typed output with grounded rendering

The decoder emits semantics and binding; deterministic tools handle exact
arithmetic and constrained geometry:

```text
exercise/activity + confidence/UNSURE
phase/attempt state
observable facts + uncertainty
supplied protocol/reference and protocol-relative assessment
evidence joints and time interval
correction target and delta
counterfactual pose/trajectory request
short language grounded in those fields
```

A numeric executor computes angles, distances, timing, and percentages with
defined denominators. A constrained kinematic or physics executor produces the
ghost/corrected movement. The language output may not invent values absent from
the structured record.

### D. Competing systems and falsifiers

Compare under matched data and compute where the comparison requires it:

- analytic geometry/task graph plus a small language verbalizer;
- a strong graph-temporal, SSM, or other modular motion model;
- the native decoder with raw only, aids only, and all admitted signals;
- query/task prior and pose-zero controls;
- relevant and matched control edits to joints, time, history, context, and
  auxiliary features.

The first claim is not “general body intelligence.” It is that the post-trained
body-connected decoder can use body evidence selectively to identify, assess, and
produce a grounded counterfactual on held movement instances beyond the strongest
shortcut and modular explanations. Cross-family and cross-domain evidence earns
the broader claim later.

## 7. Drift guards

Do not let the program silently become any of the following:

- a PT classifier;
- a rep counter or protocol event logger;
- a language model verbalizing an upstream report card;
- a synthetic-only clinical coach;
- a fixed list of twenty labels;
- an RL policy optimizing an invented reward;
- a demo whose fluent text is mistaken for physical grounding.

Conversely, do not reject useful analytic features, classifiers, references, or
complementary sensors merely to preserve a “pure pose” aesthetic. The goal is the
best defensible body-native system, with evidence that tells us what each part did.

