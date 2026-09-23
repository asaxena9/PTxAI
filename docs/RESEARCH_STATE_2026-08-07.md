# PTxAI research reset: problem formulation and belief state

Date: 2026-08-07  
Status: living research document; not a product claim or final architecture

> **Canonical scope update (2026-08-09):** The governing build thesis is now
> [`NATIVE_BODY_DECODER_CANONICAL_BUILD_2026-08-07.md`](NATIVE_BODY_DECODER_CANONICAL_BUILD_2026-08-07.md).
> This older document remains evidence and problem reconstruction, but its
> narrower event-semantics directional conclusion no longer defines the North
> Star. The active reconciliation is recorded in
> [`../experiments/autoresearch/CANONICAL_RECONCILIATION_AND_ACTIVE_BUILD_2026-08-09.md`](../experiments/autoresearch/CANONICAL_RECONCILIATION_AND_ACTIVE_BUILD_2026-08-09.md).
>
> **STALE (2026-08-13 status banner):** superseded again by the 2026-08-12
> north-star revision and the 2026-08-13 path directive. Do not resume work
> from this document. Live state: `docs/CURRENT_STATE.md` glance block; live
> rules: `docs/FOUNDER_DECISIONS.md` ACTIVE RULES block. Retained as the dated
> 08-07 belief record only.

## 1. Directional conclusion

The broad thesis “connect continuous pose to an autoregressive decoder for live
coaching” is not novel. Stream-VLM already learns asynchronous `<next>` versus
`<feedback>` decisions from live fitness video; BioCoach already conditions a
language decoder on 3D skeletal kinematics and biomechanical context; AIFit
already performs pose-based repetition segmentation and localized feedback;
LLaMo and Superman already unify motion/vision with decoder-style generative
models.

Primary sources:

- [Stream-VLM / QEVD](https://arxiv.org/abs/2407.08101)
- [BioCoach](https://arxiv.org/abs/2603.26938)
- [AIFit](https://fit3d.imar.ro/)
- [LLaMo](https://arxiv.org/abs/2602.12370)
- [Superman](https://arxiv.org/abs/2602.02401)

The strongest open problem currently visible is narrower:

> Can a compact causal model learn calibrated, observability-constrained event
> semantics for genuinely continuous clinical movement—especially partial,
> aborted, restarted, and erroneous repetitions—from scalable generic or
> synthetic motion plus limited real-patient supervision, and retain that
> capability under monocular pose noise and actual edge constraints?

This is a hypothesis about an open gap, not yet a novelty claim.

## 2. What exists versus what is proposed

The deployed Relay product is:

```text
MediaPipe Full
  → selected-side hand-engineered geometry and quality gates
  → exercise-specific deterministic state machine
  → deterministic constraint evaluator
  → template cues and summaries
```

A 2,351-parameter synthetic TCN may corroborate two state transitions. A real
145,282-parameter pose-native causal transformer implementation and a two-epoch
implementation smoke now exist, but there are no sealed outer-test predictions;
this does not establish held-out capability or decoder advantage. The stored
smoke hashes also identify an older source version, so the present campaign must
be re-frozen before execution. The 25.9M Nexus experiment measures a
random-weight exact-shape CUDA graph on an A100; it does not measure learning.

The supplied external/Claude track argued that perception should compress the
physical world and a small decoder should perform contextual judgment. That is
a useful factorization hypothesis. Its claims about 25M capability and 8–15
tokens/s on embedded hardware were not supported by an experiment in the
provided attachment. `CLAUDE.md` further fixed “prove sub-25M on MCU” as a
founder constraint. The current research mandate supersedes that restriction:
model size, decoder use, and MCU deployment are now variables to test.

## 3. Mathematical problem reconstruction

### 3.1 The physical system is partially observed

Let the latent physical state be

\[
S_t=(q_t,\dot q_t,\ddot q_t,c_t,f_t,m_t,i_t,e_t),
\]

where (q_t) is 3D body configuration, (c_t) contact state, (f_t) external
and internal forces, (m_t) physiological state such as fatigue or pain,
(i_t) intent, and (e_t) environment/equipment state. A cue or intervention
is (A_t). Dynamics are

\[
S_{t+1}\sim P_*(S_{t+1}\mid S_t,A_t,U_t),
\]

with unobserved human control (U_t).

The phone does not observe (S_t). It observes

\[
O_t=(I_t,\hat P_t,C_t,\Delta t_t,V_t),
\]

where (I_t) is RGB, (hat P_t) estimated pose, (C_t) landmark confidence or
heatmap uncertainty, (Delta t_t) actual time since the prior accepted sample,
and (V_t) camera/device context. The observation process is

\[
O_t\sim P_{obs}(O_t\mid S_t,V_t).
\]

The prescription, exercise, side, contraindications, and prior session context
form (K). A causal observer should approximate the belief state

\[
b_t(S)=P(S_t=S\mid O_{\le t},A_{<t},K).
\]

A TCN, GRU, state-space model, or transformer cache is useful only insofar as
its hidden state (H_t) is a sufficient or decision-sufficient approximation
to (b_t).

### 3.2 Assessment and coaching are different estimands

The desired outputs are not one homogeneous label:

- (Y_t^{kin}): visible kinematic quantities and their uncertainty;
- (Y_t^{event}): exercise, phase, boundary, partial/aborted/restart state;
- (Y_t^{error}): observable deviation type, location, onset, and severity;
- (Y_t^{quality}): expert ordinal or continuous judgment;
- (A_t): whether, when, and how to intervene;
- (R_{t:t+h}): downstream response after the intervention.

Perception estimates the first three. Coaching is a policy. For asymmetric
cost (C(a,S,K)), the Bayes-optimal decision is

\[
a_t^*=\arg\min_a\mathbb E_{S\sim b_t}[C(a,S,K)].
\]

That expectation explains why calibrated uncertainty and abstention matter:
two observations with identical predicted class but different posterior spread
can imply different actions.

### 3.3 Some requested targets are not identifiable from monocular pose

For observation modality (X), the irreducible classification risk is

\[
R_X^*=\mathbb E_X\left[1-\max_y P(Y=y\mid X)\right].
\]

If different latent states produce the same monocular pose history but require
different labels—e.g. different joint loading, pain, or intent—then (R_X^*>0)
regardless of model size. OpenCap demonstrates that joint moments and muscle
forces can be estimated under stronger multiview and musculoskeletal-model
assumptions; it does not make those variables identifiable from one ordinary
phone stream ([OpenCap](https://pubmed.ncbi.nlm.nih.gov/37856442/)).

The research task must therefore include an observability taxonomy:

1. directly visible kinematics;
2. inferable only with temporal/context assumptions;
3. inferable only with additional sensors or views;
4. clinically meaningful but unobservable and requiring abstention or user input.

## 4. Representation is an information decision

### 4.1 Pose can only discard information from RGB

Let (R=g(I)) be the pose representation produced from RGB. By the data
processing inequality,

\[
I(Y;R)\le I(Y;I).
\]

Pose is sufficient for target (Y) only when

\[
I(Y;I\mid R,K)=0.
\]

This gives a decisive experiment: compare a pose model with an RGB-residual
model that receives both pose and video under matched subject/device splits. If
the residual branch improves held-out error detection after controlling for
capacity and data exposure, final landmarks are not sufficient.

Useful information that pose may discard includes object contact, equipment,
surface, self-occlusion evidence, clothing deformation, and pose-estimator
failure cues. Conversely, pose removes texture/background shortcuts and can
reduce sample complexity.

### 4.2 Invariance must not erase the label

Root translation, scale, view, time warping, and left/right reflection are not
universally nuisance variables. Let (G) be a transformation group. Enforcing
invariance (f(gX)=f(X)) is valid only if

\[
P(Y\mid gX,K)=P(Y\mid X,K),\qquad \forall g\in G.
\]

Left/right reflection can invalidate asymmetric impairment labels; removing
gravity orientation can erase balance information; time-warp invariance can
erase tempo errors. The safer hypothesis is a factored representation:

\[
R_t=[R_t^{intrinsic},R_t^{world},R_t^{sensor}],
\]

where intrinsic articulation is body-centered, world/root motion remains
gravity-aware, and sensor/view/visibility remain explicit.

### 4.3 Frequency is task dependent

A representation optimized for joint-position reconstruction can preserve slow
posture while destroying velocity and landing transients. MotionVLA reports a
concrete version of this problem for motion generation: a small number of DCT
coefficients preserve much more position energy than velocity energy
([MotionVLA](https://arxiv.org/abs/2606.15142)).

A serious candidate is a multi-rate representation:

\[
R_t=R_t^{low}(q_{t-w:t})\oplus R_t^{high}(\Delta q,\Delta^2q,\epsilon_t),
\]

where the slow stream carries phase/posture and the high-frequency stream
carries residual dynamics, timestamp irregularity, and estimator innovations.
Whether this helps assessment is unknown and must be tested against an equal
bandwidth single-stream model.

## 5. What different objectives actually optimize

### 5.1 Supervised cross-entropy / SFT

For label or token sequence (Y), SFT minimizes

\[
\mathcal L_{SFT}(\theta)=-\mathbb E_{P_{tr}}\log p_\theta(Y\mid O_{\le t},K).
\]

For a class logit (z_j),

\[
\frac{\partial \ell}{\partial z_j}=p_\theta(j\mid X)-\mathbf 1[j=Y].
\]

The gradient increases probability of the observed annotation and decreases all
others. This is sufficient when labels are consistent, coverage is adequate,
and the deployment conditional matches training. It does not distinguish
expert disagreement from error, learn counterfactual cue value, or guarantee
compositional generalization. An autoregressive decoder trained this way is not
automatically open-set.

### 5.2 Hard-negative contrastive learning

For embedding (h_i), positive (h_i^+), and negatives (h_j^-), InfoNCE is

\[
\mathcal L_{NCE}=-\log
\frac{\exp(s(h_i,h_i^+)/\tau)}
{\exp(s(h_i,h_i^+)/\tau)+\sum_j\exp(s(h_i,h_j^-)/\tau)}.
\]

Its gradient pulls positives together and pushes sampled negatives apart, with
hard/high-similarity negatives receiving more force. This may help subtle form
errors, but false negatives can destroy meaningful within-class variation. A
positive/negative relation must therefore be defined by error semantics and
observability, not merely exercise ID.

### 5.3 Pairwise ranking

For expert preference (i\succ j) and scalar score (s_\theta), a logistic
ranking loss is

\[
\mathcal L_{rank}=\log\left(1+\exp[-(s_\theta(X_i)-s_\theta(X_j))]\right).
\]

The gradient enlarges the score margin in the annotated direction. Ranking can
be more statistically stable than absolute quality scores when raters agree on
ordering but not scale. It cannot localize an error unless paired with spatial
or temporal supervision.

### 5.4 Structured sequence likelihood

Framewise CE ignores legal phase order and duration. A constrained semi-Markov
model assigns a segment sequence (Z) score

\[
\log p(Z\mid X)=
\sum_n\left[
\log p(z_n\mid z_{n-1})+\log p(d_n\mid z_n)+
\sum_{t\in n}\phi_\theta(z_n,X_t)
\right]-\log\mathcal Z(X).
\]

The gradient increases the total score of the annotated legal path relative to
all competing paths. This can reduce fragmentation, but incorrect transition
constraints will force systematic errors. It is a baseline, not an automatic
truth layer.

### 5.5 Self-supervised future prediction

Next-motion likelihood can pretrain on unlabeled trajectories:

\[
\mathcal L_{pred}=-\mathbb E\log p_\theta(R_{t+1:t+h}\mid R_{\le t},K).
\]

It rewards features that predict common dynamics. It may ignore rare clinical
errors or spend capacity on nuisance motion. Its value is empirical: add it to
the same supervised model and measure real subject/device-held-out sample
efficiency and rare-error recall.

### 5.6 Distillation

A privileged teacher may observe full future video, multiview pose, IMU, or
force while the causal student sees monocular past input:

\[
\mathcal L_{KD}=T^2\operatorname{KL}(p_T^T\Vert p_S^T).
\]

This transfers the teacher's soft class geometry. It cannot give the student
information that is absent from its input at decision time; overconfident
teacher probabilities can instead teach hallucination. Compare with an
equal-sized non-distilled student and score calibration under sensor shift.
Privileged offline-to-online action distillation is established prior art, not
itself novel.

### 5.7 Reinforcement learning

RL is relevant only when the target is a sequential intervention policy:

\[
J(\pi)=\mathbb E_{\tau\sim\pi}\sum_t\gamma^t r(S_t,A_t,S_{t+1}).
\]

SFT learns actions demonstrated by therapists; RL can prefer actions with
better delayed outcomes. But logged clinical data create support and
confounding problems: off-policy value is unreliable where the proposed action
was rarely taken. RL does not improve pose observability and is not justified
until the perception/event interface, reward, behavior-policy coverage, and
safety constraints are measured. Online exploration on patients is out of
scope.

## 6. Data scale is a multiaxis learning curve

Frames are not independent examples. For an autocorrelated sequence of length
(T), a rough effective sample size is

\[
n_{eff}\approx \frac{T}{1+2\sum_{k\ge1}\rho_k},
\]

but the deployment generalization unit is usually the subject, session, site,
device, and error family. One million overlapping frames from ten people may
provide less external-validity evidence than ten thousand events from hundreds
of people.

The data-scale estimand should be a hierarchical learning surface

\[
M=M(n_{subject},n_{session},n_{device/site},n_{error},n_{synthetic}),
\]

estimated by repeated cluster-level subset draws and training seeds. A useful
parametric summary is

\[
M(n)=M_\infty-A n^{-\alpha},
\]

but only if residuals and the sampled range support it. The slope (alpha),
asymptote, and uncertainty—not an arbitrary “100,000 examples”—determine
whether the system is data limited.

For synthetic source (Q_\phi(X,Y)) and real target (P_*(X,Y)), the relevant
question is not synthetic accuracy. It is real-data savings:

\[
\Delta n_{real}(m)=
n_{real}^{real-only}(m)-n_{real}^{synthetic\rightarrow real}(m),
\]

the reduction in real independent units needed to reach target metric (m).
Measure this separately by error family and device/view. Synthetic descendants
of one base motion remain one cluster.

## 7. Current measured belief updates

| Claim | Status after reset | Evidence |
|---|---|---|
| Pose/vision plus decoder for live fitness feedback is novel | Rejected | Stream-VLM, BioCoach, AIFit, LLaMo, Superman |
| A 25.9M decoder-shaped graph can stream on an A100 | Strengthened narrowly | 9.01 ms p50, 9.24 ms p95 at 1,024-token starting context |
| That graph learns movement judgment | Unknown | no fusion/training experiment |
| Current synthetic TCN uses ordered history | Strengthened within generator | 0.887 versus 0.853 parameter-matched order-invariant model |
| Current synthetic result shows unseen-subject generalization | Rejected | anatomy is redrawn per sequence; overlapping frame windows |
| 10 Hz preserves slow midpoint crossings | Strengthened narrowly | 57/57 crossings versus 30 Hz proxy, noncausal pilot |
| 10 Hz is sufficient for jumps/landings | Unknown and physically doubtful | no high-rate reference experiment |
| 100k real examples are required | Unsupported | no cluster-level learning curve |
| SFT must plateau | Unsupported | no real-data objective comparison |
| RL is required | Unsupported | no learned intervention policy or reward dataset |
| REHAB24-6 correctness requires temporal order | Not strengthened | unordered pose F1 0.683; ordered DCT 0.651; matched TCN 0.700 versus DeepSets 0.685 with interval crossing zero |
| REHAB24-6 is shortcut-safe | Rejected | recording/block-position metadata reaches 0.759 participant-macro F1 |
| A small causal model has proven value over an invariant model | Unknown | matched TCN point gain 0.0153; participant interval [-0.0264, 0.0571] |

## 8. Candidate publication contributions, ranked today

### A. Continuous clinical session benchmark and evaluator — highest current value

A dataset/evaluation protocol jointly labeling exercise state, phase, valid,
partial, aborted, paused, resumed and restarted repetitions, error onset/offset,
observability, rater disagreement, cue timing, and decision-availability time.
No located benchmark combines these in untrimmed patient sessions.

The important technical contribution would be causal evaluation and
observability, not simply collecting videos.

### B. Clinical synthetic-to-real transfer frontier

Biomechanics-constrained counterfactual errors plus camera and pose-estimator
corruption, evaluated by how much real patient data they save per error family.
This survives as a credible gap if the generator is independently validated and
synthetic ancestry is split correctly.

### C. Frequency- and uncertainty-aware compact motion representation

A slow articulation stream plus high-frequency dynamics/innovation stream,
with explicit pose uncertainty and RGB residuals. The contribution exists only
if it beats single-stream landmark and RGB baselines at matched bandwidth on
real errors.

### D. Compact unified pose-language-action model — conditional contribution

Potentially valuable if a **sub-150M** causal pose-native model beats strong TCN/SSM,
structured-state, and boundary-detector-plus-decoder baselines on long-context
or compositional decisions, calibration, transfer and edge Pareto metrics. The
0.1M region is a scientific control, not the north-star architecture. The 350M
endpoint is an offline research/teacher ceiling, not a deployment target. Small
size, large size or native fusion alone is not novel. The potential contribution
is the full measured conjunction of native pose tokens, causal untrimmed state,
structured physical-event outputs, calibrated abstention, and sustained
on-device performance. Broad "pose plus decoder" novelty is already falsified by
prior motion-language and skeleton-MLLM work.

## 9. Model search is not limited to Ember and Nexus

The local model names are anchors, not a prior over the winning architecture.
The verified local Ember architecture has 6.49M active parameters and the
verified Nexus architecture has 25.83M; both are ordinary Llama-family causal
decoders without pose pretraining or a pose adapter. A clean Ember base weight
is present. The intended clean Nexus base weight is absent; the only complete
local Nexus weight is a refrigerator-QA fine-tune and is not a fair
initialization for pose research.

The architecture tournament therefore includes any credible open or
reproducible model up to a nominal 350M total active-parameter research ceiling,
including pose/skeleton networks, causal video models, state-space models,
motion foundation models,
compact VLMs, spatial/geometry models, and robotics/VLA components. Candidates
are admitted by verified checkpoint, license, input/output contract, causal
adaptability, and a matched benchmark—not by model name or parameter count.
The ceiling does not alter the frozen Stage-0 gate and does not authorize large
models on inadequate supervision. Capacity scaling and checkpoint transfer are
separate experiments, and model size is co-scaled with independent physical
support rather than repeated windows. The detailed amendment and persistent
data-agent contract are in `docs/SCALE_AND_DATA_AUTORESEARCH_2026.md`.

Every candidate is separated into four effects:

1. architecture from random initialization;
2. transferable pretraining at a frozen or adapter-only budget;
3. capacity under data-matched training;
4. efficiency under compute- and deployment-matched training.

Language generation earns its cost only if it improves a conditional structured
decision that a state estimator plus renderer cannot match. Fluency or JSON
validity is not movement understanding.

Mechanistic promotion also requires token/embedding accounting, per-loss
gradient analysis, attention/MLP/logit diagnostics, and causal interventions.
Attention heatmaps alone are not evidence; a proposed movement circuit must
predict the behavioral effect of joint/frame/token deletion or activation
patching and survive a matched irrelevant intervention.

## 10. Adaptive experiment portfolio

The next experiment is selected by expected decision value, not a fixed ladder.
The current highest-value branches are:

1. **Real trajectory learnability — first gate complete:** REHAB24-6 LOSO
   establishes pose signal, a stronger recording-order shortcut, and no reliable
   matched temporal advantage on binary correctness. The next version requires
   intrinsically temporal labels rather than larger models on the same target.
2. **Observability:** find or collect synchronized monocular RGB, pose,
   multiview/mocap/IMU reference and expert error labels; quantify residual
   information in RGB and stronger sensors conditional on pose.
3. **Annotation reliability pilot:** two clinicians label event time,
   availability time, error onset, severity, and observability on untrimmed
   sessions; disagreement determines the target and evaluation ceiling.
4. **Continuous semantics:** benchmark FSM/HSMM, causal TCN/GRU/SSM, compact
   decoder, and two-stage boundary-plus-decoder on partial/restart/aborted
   sequences under identical inputs and exposure.
5. **Scaling and synthesis:** build an ongoing discovery/admission/conversion
   agent now, but promote sources and larger tiers only through capability-specific
   held-out utility. After a real baseline exists, fit the joint capacity ×
   subject/error/device surface and measure synthetic real-data savings.

Training a large native decoder before branches 1–3 would produce an answer to
an easier surrogate rather than evidence about the research problem.
