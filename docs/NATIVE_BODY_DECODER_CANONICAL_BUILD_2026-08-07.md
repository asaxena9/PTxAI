# Canonical build thesis: a native continual body-motion decoder

Date: 2026-08-07  
Status: living research thesis and stage contract  
Evidence status: architecture and capability claims are **HYPOTHESIS** unless
explicitly labeled otherwise below

This document fixes the target after reconciling the original project idea, the
pose-native Nexus/streaming-decoder designs, the data audit, the top-down A100
experiments, the Fable/Claude line of reasoning, and the compact prior-art audit.
It supersedes earlier documents only where they accidentally changed the first
capability into a post-hoc summary, a fixed rep logger, coaching, multi-person
tracking, or final-device deployment. Those remain baselines, internal machinery,
or later branches.

## The system in one sentence

> Build a causal model that receives one person's pose as a native, ongoing input
> modality and uses a decoder to perform grounded physical analysis over that
> stream: understand what the person is doing, compare motion across time, apply a
> supplied task or protocol, explain its conclusion with body/time evidence, and
> emit structured answers or actions. Event registration and a movement ledger are
> state primitives that support this intelligence, not the research endpoint.

Physical therapy supplies a bounded, falsifiable first domain. It does not define
the eventual capability. The first proof is not a clinical claim and does not
require coaching, multiple people, robot control, or every possible body metric.

## What “native body plus decoder” means

At accepted time `t`, the observed packet is

\[
O_t=(P_t,C_t,M_t,\Delta t,R_t),
\]

where `P_t` is the joint geometry, `C_t` is landmark confidence, `M_t` is an
observed/missing mask, `Delta t` is physical elapsed time, and `R_t` contains
separate root/camera channels. A learned spatial adapter produces one or a small
adaptive number of body embeddings:

\[
Z_t=E_\phi(O_t)\in\mathbb{R}^{n_t\times d}.
\]

Those embeddings enter the causal decoder residual stream directly, alongside
task/protocol/query and prior-event tokens:

\[
(H_t,S_t)=D_\theta(Z_t,Q_{\le t},E_{<t};S_{t-1}),
\qquad
p_\theta(Y_t\mid\mathcal H_t,Q_t)
=\prod_i p_\theta(y_{t,i}\mid y_{t,<i},\mathcal H_t,Q_t).
\]

`S_t` is cached attention state, recurrent/SSM state, or a measured hybrid; it is
not a handcrafted rep summary. `Y_t` may be a registered event, a multi-field
analysis, an evidence span, an answer to a movement question, or no response. A
pose SDK before `E_phi` is allowed in the first build, just as a VLM may use a
visual encoder. A learned graph/spatial adapter is also allowed. Feeding only
handcrafted phase, rep count, movement summary, or an FSM state into the decoder
is not the native path and belongs in a control pipeline.

The word *decoder* does not mean that the model must produce prose after every
frame. Most input steps can produce `WAIT` or no externally rendered output.
When evidence is sufficient, the model may first register an event:

```text
<ATTEMPT_START t=12.4>
<REP_COMPLETE id=7 t=15.8 duration=3.4>
```

The committed event updates a monotonic external ledger. That ledger provides
transactional exactly-once counting across retries and finite-context eviction;
it does not supply the analysis. The same decoder can then answer, for example:

```json
{
  "protocol": "8 reps, 3 s descent, 1 s pause",
  "observed": {"attempts": 8, "completed": 7},
  "analysis": "rep 4 reversed before the target range; reps 6-8 descended faster than requested",
  "evidence": [{"rep": 4, "time": [12.4, 14.1], "joints": ["hip", "knee"]}]
}
```

This is an illustrative output contract, not a claim that the current data can
supervise every field. A deterministic serializer may render accepted output as
JSON. The decoder's scientific role is the conditional, temporally grounded
analysis—not JSON syntax and not the counter alone.

## Why a decoder at all?

If the task is a fixed function such as

\[
f(O_{\le t})\rightarrow\{\text{rep complete or not}\},
\]

then a decoder is unnecessary. A TCN, graph model, SSM, RNN, FSM or phase detector
can solve it more cheaply, and compact online pose-event systems already exist.
The decoder is justified only if one shared state can condition on a variable
instruction/query `Q`, compose several body/temporal operations, and emit a
variable structured result without training a new fixed head for every requested
analysis.

Examples of those operations include `COUNT`, `COMPARE`, `ARGMAX`, `BEFORE`,
`DURATION`, `REVERSAL`, `PROTOCOL_CHECK`, `EVIDENCE_LOCALIZE` and, later,
`FORECAST`. “Efficiency” must be defined relative to a task and observable metric:
elapsed time, idle time, path length, extra motion relative to a reference,
sequence violations, or protocol deviation. Pose alone does not identify energy
expenditure, force, pain, object state, or work quality.

No computation is uniquely possible only in an autoregressive decoder. The
research hypothesis is that decoder-style conditional generation provides a
useful *unified and transferable* body-analysis interface at compact scale. If it
does not beat or generalize beyond specialized alternatives, the decoder is not
justified.

## The first theory to prove

**HYPOTHESIS H1:** A sufficiently capable causal decoder, trained on direct pose
streams, can jointly maintain movement state and answer *held-out compositions*
of body, temporal and task/protocol questions on unseen people. Its answers will
remain grounded in the causally relevant joints and time intervals rather than
only the current pose, elapsed time, exercise identity, output-language priors,
or a handcrafted cycle statistic.

The initial estimand is

\[
p(Y_{1:T},E_{1:T}\mid O_{1:T},Q),
\]

evaluated in prefix-causal operation. `E` is the registered movement state; `Y`
is a query/instruction-conditioned structured analysis. Event accuracy is a
necessary substrate measurement, not sufficient evidence of decoder intelligence.

**Null explanations:** the same answers can be obtained from endpoint geometry,
cadence, duration, recording order, exercise ID, a pose multiset, an analytic
summary, or a simple query-conditioned temporal model; the decoder may also learn
valid output syntax while ignoring motion. These are competing hypotheses, not
reasons to avoid the run.

Let `G` be a set of primitive operations such as comparison, ordering, duration,
protocol checking and evidence localization. Training exposes some compositions
`C_train subset G*`; the decisive split contains `C_test` whose primitives were
seen but whose combinations, protocol values, or task instances were not. This
distinguishes memorizing a field mapping from composing an analysis.

The first proof needs to show all of the following on subject-held-out streams:

1. a minimal movement state—attempts, repetitions and relevant intervals—is
   registered accurately enough to support the requested analysis;
2. multi-field answers are correct on held-out query/protocol compositions, not
   merely held-out paraphrases;
3. changing the supplied protocol while holding motion fixed changes only the
   protocol-relative judgment, while motion facts remain invariant;
4. temporal shuffling, history reset, or deletion/counterfactual editing of the
   causally relevant joint/time span changes the relevant answer more than a
   matched irrelevant intervention; and
5. evidence intervals/body parts agree with the answer and the output remains
   schema-valid and internally consistent.

Primary measurements include whole-analysis exact match and field F1, held-out
composition accuracy, evidence interval/body-part accuracy, counterfactual
consistency, and calibration. State-substrate measurements include boundary-time
error, false increments per observed hour, count error and decision latency. A
boundary tolerance must be estimated from label resolution and rater disagreement
rather than chosen as an arbitrary round gate.

Event registration remains an auxiliary/state objective. Two formulations are
live competitors:

- discrete-time next-event-token likelihood, including a carefully weighted or
  sampled `WAIT` state; and
- a marked-event/hazard likelihood read from the same decoder state, followed by
  constrained event serialization.

For token likelihood,

\[
\mathcal L_{event}=-\sum_t w(e_t)\log p_\theta(e_t\mid\mathcal H_t).
\]

The decoder analysis loss is

\[
\mathcal L_{analysis}=-\sum_i\log p_\theta(y_i\mid y_{<i},\mathcal H_t,Q).
\]

Its gradient rewards the complete requested answer conditional on prior answer
fields, but may reward language/template shortcuts. Composition holds, symbolic
protocol perturbations and motion interventions are therefore required. The
event loss raises the correct event logit at the causally available time but can
reward cadence shortcuts when `WAIT` dominates. The objective weights and their
gradient conflicts must be measured; neither objective is accepted by name.

## Capability stages

Each stage proves one additional property. Later properties are not prerequisites
for beginning the first model, and a result may change the subsequent branch.

### Stage 1 — native body analysis, with movement registration as state

This is the first model we are building. Rep/event registration is included
because the decoder needs temporal state, but a rep-count result alone does not
pass this stage.

- One tracked person.
- Direct timestamped pose/confidence/missingness tokens.
- A strong causal decoder, tested from the top of the admitted 270--350M research
  range before descending.
- Prefix-by-prefix movement state and an exactly-once event ledger.
- Query- and protocol-conditioned structured analysis over the raw pose history,
  current state and prior registered events.
- Body/time evidence fields and held-out operation compositions.
- A small, bounded PT exercise set as the first physical distribution, with
  derived analytic questions and only valid human/clinical labels.

Current REHAB24-6 can support a **development version**: start/end/count state plus
mathematically derived questions about duration, ordering, amplitude, comparison,
reversal and synthetic protocol adherence for six exercises. These targets must
be regenerated from audited pose and verified; many query rows from one movement
increase optimization coverage, not independent physical support. The corpus
cannot support natural partial reps, pauses, restarts, sets, clinical form errors,
safety, coaching or treatment effectiveness.

### Stage 2 — continual and retained movement intelligence

Extend the Stage-1 analysis across longer untrimmed sessions and arbitrary query
times. The state and ledger support raw pose; they never replace it. Examples are:

- “Was the third rep slower or deeper than the second?”
- “At what point did the movement reverse?”
- “How long was the break between sets?”
- “Which repetitions followed the requested tempo or count protocol?”
- “What evidence caused the last rep to be committed?”

Add phase/progress, rep-to-rep comparison, causal future-motion prediction, and
memory beyond the raw context only as separately measured capabilities. At one
pose token per 100 ms, 1,024 raw positions span 102.4 seconds before instruction,
memory and output reservations; this is a derived capacity fact, not a fixed
tokenization choice. Rolling KV, compressed KV, recurrent/SSM state and
event-memory hybrids remain competing designs.

### Stage 3 — incomplete, incorrect and partially observable movement

Extend the event ontology to partial, aborted, paused, resumed and restarted
attempts, then to localized movement deviations and quality. This requires real
untrimmed examples and label fields that identify those events. Monocular pose
cannot be assumed to reveal pain, load, force, intent, joint torque or an occluded
joint. The model must learn `UNKNOWN`/abstention for targets outside the measured
observation envelope, or receive additional RGB, depth, IMU or force information.

This is where patient-specific adaptation, richer biomechanics and clinician-rated
form may become valid tasks. Basic protocol-conditioned analysis already belongs
in Stage 1; clinical correctness and safety do not. None of this is synonymous
with coaching.

### Stage 4 — capability descent and on-device proof

Use the strongest demonstrated model as an achievable reference, then map what
survives near 150M, 100M, 60M, 25M and below through motion pretraining,
architecture changes, distillation, pruning and quantization. These are response
surface points, not permission gates. Report active/non-embedding parameters,
FLOPs, state/KV bytes, data exposure and achieved hardware utilization.

Only this stage claims edge operation. Measure the complete sustained
camera--pose--decoder--ledger path on the target phone: p50/p95 latency, dropped
timestamps, peak memory, energy and thermal behavior. A100 throughput and model
size alone are not edge evidence. ESP32 is a later compressed/split deployment
question, not the primary host for a 25--350M decoder.

### Stage 5 — action, personalization and cross-domain transfer

Longitudinal personalization can be built on the retained movement record.
Coaching or cue selection is an optional intervention-policy problem, not the
definition of the body decoder. Supervised perception, state and analysis
semantics come first; preference learning or offline RL is used only when logged
actions, propensities, later outcomes and safety/nuisance costs exist.

Industrial transfer requires its own body+tool+object observations and quality or
process truth. Robotics may use the learned human-motion state, predictions and
counterfactual trajectories, but neither transfer follows automatically from a PT
result.

## Training ladder inside the capability stages

The capability stages above must not be confused with an automatic recipe of
“SFT, then RL.” The live training hypotheses are:

1. **Body/motion representation pretraining:** causal future motion, masked or
   corrupted pose recovery, cross-view consistency, spatial/temporal objectives,
   or a learned motion tokenizer on admitted public and synthetic trajectories.
2. **Pose-to-decoder alignment:** train the spatial adapter with the core frozen,
   partially adapted, LoRA-adapted and fully adapted as explicit alternatives.
3. **Causal state/event SFT:** train on full streams with event tokens/likelihood,
   not isolated rep clips with only a final label.
4. **Grounded multi-task/instruction SFT:** comparison, temporal QA, protocol
   application, evidence fields and JSON/language are part of the first decoder
   proof; forecast and clinical counterfactuals wait for valid targets.
5. **Distillation and descent:** compare a motion/privileged teacher, the same
   student without distillation, and gold-only training on held-out real data.
6. **Policy learning:** preference, bandit or offline RL only for a later action
   policy with measurable outcomes. RL is not a repair for missing event labels.

For any joint objective `L=sum_j lambda_j L_j`, record per-head gradient norms
and cosine similarities. A decreasing scalar sum does not establish that event,
motion, uncertainty and language tasks are helping rather than interfering.

## Model search: top-down without a predetermined winner

The first capability attempt should not begin with a 0.1M model. Small TCN/GRU,
graph, SSM, analytic and summary systems are controls that identify shortcuts and
the need for capacity. The constructive search begins at the strongest admitted
models and works downward:

- 270--350M causal/hybrid and motion-language candidates as capability references;
- 125--150M mobile/text and causal-SSM candidates;
- 60--100M motion-language and skeleton-pretrained candidates;
- 25--45M Nexus/MotionBERT-scale candidates; and
- Ember-scale and smaller students only when the question is compression or a
  control, not as a gate on the whole thesis.

Text pretraining, reasoning post-training and motion pretraining are distinct
initialization factors. A fluent compact LLM is not assumed to contain a motion
prior. **MEASURED-PILOT:** the first 354.48M LFM2.5 frozen-core future-pose probe
trained only a 593,346-parameter adapter/head for 20 updates on D0. On its single
held-out participant, macro component MAE was 0.2237 for the pretrained core and
0.1709 for the architecture-random core, while analytic persistence was 0.0696.
This shows that this particular frozen-text-core/short-adaptation probe did not
beat a trivial motion baseline. It did not test conditional physical analysis and
does not decide LoRA, full adaptation, motion pretraining, another architecture,
or the body-decoder hypothesis.

The next expensive experiments should attack Stage-1 body analysis directly. Up to
three A100 jobs can run isolated alternatives in parallel—for example a strong
270--350M direct-pose decoder, a motion-native/spatial-temporal alternative, and a
query-conditioned causal baseline—under the same observations, questions,
composition split and state semantics. The three cells are competing explanations,
not a checklist whose smallest model can block the others.

## Data program tied to the stages

**MEASURED:** the admitted current event corpus has 64 usable recordings, 10
people, 1.710 hours, 61,564 accepted 10 Hz observations and 1,057 annotated
start/end events. It has enough signal for development and optimizer/capability
pilots, not a defensible general body-intelligence or clinical claim. The numbers
10,000 or 100,000 are not yet empirically justified sample-complexity thresholds.
Frames and augmented descendants do not equal independent movements.

The data engine therefore runs in parallel with model research:

- **D1 — broad motion pretraining:** audited pose/mocap/video motion corpora and
  physically constrained synthetic trajectories. These teach geometry and
  dynamics, not PT event semantics by themselves.
- **D2 — deployed observation bridge:** synchronized phone RGB/estimated pose and
  privileged 3D/multiview/IMU truth, used to measure what the phone representation
  preserves or loses.
- **D3 — real continual event data:** full sessions containing attempts,
  completed/partial/aborted repetitions, rests, pauses, resumes, restarts and
  exercise transitions. This supports continual state and Stage-3 semantics.
- **D3Q — grounded physical-analysis supervision:** auditable analytic/symbolic
  questions over real motions, real task/protocol descriptions, evidence spans,
  held-out operator compositions and, where required, independent expert labels.
  This is decisive for Stage 1; a high event score cannot substitute for it.
- **D4 — assessment data:** localized error/quality targets, observability,
  severity and rater disagreement.
- **D5 — action/outcome logs:** only for personalization or policy/RL.

Each source retains participant/session/device/ancestor grouping and an admission
contract. Data scale is selected from held-out learning curves over independent
people, sessions and event types. Synthetic and teacher-generated examples may
increase coverage and optimization signal, but they do not increase independent
real-world support.

## The broader research object

The intended object is a **compact continual physical-analysis model**, not a PT
application with an LLM attached. Its reusable contract is:

```text
timestamped body observations
+ task / protocol / question
+ optional object, tool, environment or sensor tokens
              |
              v
learned physical token interface -> causal decoder state
              |
              +-> registered state/events/memory
              +-> structured analysis + body/time evidence
              +-> prediction/counterfactual
              +-> optional action token
```

Pose is the first high-information, privacy-friendlier observation interface. It
is not sufficient for every physical question. A coffee-making system needs cup,
machine and object-state observations to judge process completion; an industrial
system needs tool/load/process truth to judge work quality; a robot handover model
needs object and robot state. The body stream remains native while other physical
tokens are added through the same conditional interface.

Potential applications, with their actual observable targets, include:

| Domain | Body-decoder analysis | Additional truth required |
|---|---|---|
| PT/rehabilitation | protocol adherence, rep comparison, movement sequence, kinematic deviation, retained session history | clinician labels for form/safety; pain and force are not pose facts |
| Sports/fitness | technique phase, consistency, asymmetry, tempo, reference comparison and change over a session | task rules; force/load/physiology sensors for claims about exertion |
| Industrial/ergonomics | posture exposure, reach/lift sequence, excess kinematic path, idle time, procedure deviation | object/tool/load/process sensors and a defined quality metric |
| Assembly/cooking/daily activity | which physical step occurred, order, omissions, timing and body-object coordination | object identities/states and scene context; pose alone is insufficient |
| Human-robot collaboration | human-motion intent, handover readiness, near-future body trajectory and explanation | robot/object/world state plus safety validation |
| Robot learning from people | segment demonstrations, align language to trajectory intervals, compare executions and generate candidate corrected motion | robot embodiment/dynamics and success signals |
| Accessibility/home monitoring | trends in sit-to-stand, gait or repeated-task execution and failed attempts | prospective population validation and conservative uncertainty |

“Efficiency” is therefore a task-relative estimand, not a universal score. For
task `q`, reference/constraint set `R_q`, and motion `O`, an analysis may estimate

\[
g_q(O,R_q)=
(\text{time},\text{idle},\text{path},\text{sequence violations},
\text{reference deviation}),
\]

then explain which intervals generated each component. The decoder's advantage is
conditioning `g_q` on a new task/protocol and composing requested fields. It must
not relabel kinematic economy as metabolic, clinical or production efficiency.

## What we are trying to bring to the table

[SkeletonLLM](https://arxiv.org/abs/2603.18003) changed the *interface*: it learned
a differentiable skeleton renderer so an 8B vision-language model could reuse its
visual prior across recognition, captioning, QA and topology shifts. UPS changed
the *output formulation* by mapping pose tasks into a unified autoregressive token
space. MotionGPT changed the *motion-language vocabulary*. FiGMo broadened the
*offline temporal QA/data curriculum*. Compact online skeleton models changed the
*state-update efficiency* but retained fixed heads.

Our candidate change is different:

> Treat observed human motion as a persistent native physical modality inside a
> compact decoder, and train/evaluate it for instruction-conditioned, evidence-
> grounded physical analysis while the activity is still unfolding.

That candidate contains five separable contributions. They are not all assumed
to work, and publication does not require inventing all five:

1. **Continual physical token interface.** Direct numerical pose, confidence,
   missingness, topology and physical time update the model incrementally instead
   of being rendered/summarized only after a completed clip.
2. **Precision/semantic representation hypothesis.** Direct kinematic tokens may
   preserve timing and geometry; a learned part/body-glyph or visual route may
   reuse semantic priors; a gated dual route may outperform either. This is an
   experiment inspired by SkeletonLLM's interface insight, not a selected answer.
3. **Physical reasoning output contract.** The decoder must produce analysis plus
   body/time evidence, not just a caption, action class or fluent rationale. It is
   tested on held-out task/query compositions and protocol counterfactuals.
4. **Causal continual benchmark/data engine.** Real streams, analytic questions,
   hard matched negatives, partial/restarted actions, topology/device shifts and
   intervention tests expose whether the model uses physical evidence.
5. **Capability-efficiency descent.** Establish the capability with the strongest
   admitted model, then find the smallest model that retains it and measure the
   complete edge pipeline.

The strongest architecture possibility is a **dual-route physical token bus**:

\[
Z_t^{num}=E_{graph}(P_t,C_t,M_t,\Delta t,G),\qquad
Z_t^{sem}=E_{glyph/visual}(P_t,C_t,G),
\]

\[
Z_t=\operatorname{Fuse}_\phi(Z_t^{num},Z_t^{sem},Q_t).
\]

The numerical route should dominate exact geometry/timing; the semantic route
should help low-shot concept and topology transfer. The hybrid is valuable only
if matched ablations show complementary information, gradient flow and held-out
benefit after accounting for token/compute cost. It is one high-value hypothesis,
not the definition of the project.

The first real scientific benchmark should therefore include four analysis
families rather than a rep counter alone:

1. **state/facts:** activity, attempt, count, phase/order, duration and breaks;
2. **relations:** compare reps, body parts, time intervals and a reference motion;
3. **instruction/protocol application:** change the prompt while holding the
   motion fixed and evaluate the corresponding adherence judgment;
4. **grounded explanation/prediction:** identify responsible joints/intervals and
   answer a bounded causal forecast or counterfactual whose target is observable.

Hold out compositions, protocols, people and at least one physical task family.
The final item is important: if the same learned interface transfers from PT to a
separately audited physical procedure with new object/context tokens, the result
supports a general physical-analysis claim rather than a rehabilitation-specific
classifier.

**MEASURED-DEVELOPMENT-PROBE:** the existing 110,480-parameter GRU/query decoder
showed only a small +3.0875-point exact-accuracy lift over the current-pose model
on held-composition historical queries. History deletion reduced the relevant
score, but the model failed the stronger identical-current-pose/opposite-history
collision test and showed weak joint/time localization. This does not demonstrate
body intelligence. It usefully specifies what the larger native decoder must
beat: robust state-indexed discrimination, composition and evidence grounding—not
another easy query score.

Today, none of these is a demonstrated contribution. What exists is a sharper
hypothesis, audited inputs, prior-art boundaries, and initial negative/weak pilot
evidence. We bring something to the field only when one or more of the five claims
survives matched baselines, causal interventions, held-out physical support and
compute accounting.

## What is and is not novel today

**SOURCED:** Direct pose-to-decoder systems already exist: UPS uses pose to
autoregressive unified outputs; AutoSign reports a sub-150M direct pose-to-text
system, although its exact total is unreconciled; MotionGPT evaluates 60M/220M/
770M motion-language models; FiGMo shows broad offline body QA at 7B. Therefore
“first pose plus decoder” is not defensible.

**UNKNOWN / open empirical conjunction:** no primary system located in the
current audit establishes the combination of compact direct pose input,
continual causal state, held-out compositional physical analysis with grounded
body/time evidence, task/protocol conditioning, and sustained end-to-end phone
operation. Absence from a bounded audit is not proof that no paper or product
exists.

There is no established new mathematics yet. A publishable contribution can
still be an empirical/system contribution: a new continual body-decoder task and
dataset, a validated objective or memory mechanism, a capability-efficiency
frontier, or a compact edge deployment whose interaction ablations show that the
joined design contributes beyond its known parts. Mathematical novelty would
require an actual new result, such as a consistent event objective, identifiable
state/observability result, or state-compression guarantee—not equations attached
to an otherwise known architecture.

## Non-goals for the first proof

- multi-person tracking;
- clinical diagnosis, safety or treatment efficacy;
- coaching or RL;
- full raw-RGB end-to-end training;
- infinite-context memory;
- industrial efficiency or robot-control claims;
- claiming that a decoder is needed for fixed rep counting.

The Stage-1 objective is the actual decoder question: prove that a serious compact
model can read native body pose and perform conditional, temporally grounded
physical analysis that a fixed logger or summary pipeline does not already solve.
The event ledger is one internal instrument for that analysis. Evidence—not the
wording of this document—decides which memory, data, training and deployment
questions come next.
