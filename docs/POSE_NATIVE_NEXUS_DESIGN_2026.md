# Pose-native Nexus: a continuous vision-motion-language decoder

**Research/design cutoff:** 2026-08-07  
**Target:** a genuinely unified body-pose + decoder model, not a post-hoc summary
model.

> **Feasibility correction after sampling audit:** The output heads described below
> are architectural research hypotheses, not jointly established capabilities. A
> linear head is cheap to attach, but its target may be undersampled, unobservable from
> monocular pose, uncalibrated, or unsupported by intervention data. In particular,
> 10 Hz can support coarse slow-exercise phase tracking but cannot resolve fast jump
> subphases or landing kinetics. See
> [the sampling and capability audit](POSE_NATIVE_CAPABILITY_AUDIT_2026.md).

## The direct answer

Body pose can be made a native decoder modality exactly as image embeddings are made
a native modality in a VLM. The decoder does not need to receive English descriptions
of the pose. A pose tokenizer emits vectors in the same 512-dimensional space as Nexus
text embeddings; causal self-attention processes the mixed sequence of prescription,
pose, motion, action, and language tokens.

```text
text instruction ────────────────┐
                                 │
camera -> pose heatmaps/features -> pose tokenizer -> 512-D soft tokens
                                 │
calibration/profile ─────────────┤
                                 v
                    one interleaved causal sequence
                                 v
                    Nexus-Flame decoder backbone
                    /           |             \
          state/action heads  future-motion   token/JSON head
```

The shared model can be trained to attempt the following outputs, subject to the
sampling and identifiability limits audited separately:

- recognize exercise, phase, repetition and set state;
- localize which body part is deviating and when;
- forecast the next 0.25-2 seconds of motion and detect an approaching violation;
- choose a cue, camera instruction, overlay target, pause, count, or abstention action;
- answer questions grounded in the current and recent movement;
- emit a constrained JSON/tool call or natural-language explanation;
- compare the present motion with prior reps, a reference demonstration, or a
  personalized baseline;
- predict a corrected target pose/trajectory, not merely describe the error.

A summary is only one optional output task.

## Evidence that this is a real model family

This design is not speculative in its basic mechanics:

- PaLM-E inserts continuous sensor and visual encodings directly into the language
  embedding sequence and trains the resulting embodied model end-to-end
  ([PaLM-E](https://arxiv.org/abs/2303.03378)).
- RT-2 represents robot actions as tokens and trains a vision-language model to emit
  them in the same autoregressive format as language
  ([RT-2](https://arxiv.org/abs/2307.15818)).
- MotionGPT quantizes human motion into a motion vocabulary and trains a common
  motion-language model on both kinds of tokens
  ([MotionGPT](https://arxiv.org/abs/2306.14795)).
- LLaMo uses continuous causal motion latents, a decoder-only backbone and a
  motion-specific output head, supporting unified motion understanding and streaming
  generation above 30 fps in its tested setting
  ([LLaMo, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_LLaMo_Scaling_Pretrained_Language_Models_for_Unified_Motion_Understanding_and_CVPR_2026_paper.html)).
- Superman learns a vision-guided motion vocabulary and uses one MLLM for video-to-3D
  pose, motion prediction and motion in-betweening
  ([Superman, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Superman_Unifying_Skeleton_and_Vision_for_Human_Motion_Perception_and_CVPR_2026_paper.html)).

The open research contribution would be doing this at roughly 25M parameters, causally,
on a phone, for fine-grained movement assessment rather than generic motion captioning
or robot control.

## Nexus is already a suitable exact backbone

The Nexus checkpoint in the adjacent repositories is not an abstract 25M model. Its
released configuration is:

| Component | Nexus-Flame value |
|---|---:|
| Layers | 8 |
| Hidden width | 512 |
| Query heads | 8 × 64 |
| KV heads | 2 × 64, grouped-query attention |
| SwiGLU intermediate width | 1,408 |
| Vocabulary | 6,400, tied input/output embedding |
| Parameters | 25,829,888 |
| Configured positions | 32,768 |

The parameter count decomposes almost exactly as:

\[
\begin{aligned}
P_{attn/layer} &= 512^2 + 2(512\cdot128)+512^2=655{,}360 \\
P_{mlp/layer} &= 3(512\cdot1408)=2{,}162{,}688 \\
P_{blocks} &\approx 8(2{,}819{,}072)=22{,}552{,}576 \\
P_{embedding} &= 6400\cdot512=3{,}276{,}800.
\end{aligned}
\]

The local audio-preset work deliberately kept prompt plus output within 1,024 tokens,
despite the larger configured position limit. Therefore `32,768 configured positions`
must not be treated as `32,768 positions behaviorally trained and validated`. Begin
with a 1,024-token streaming contract, then measure length generalization or continue
pretraining with the actual cache-eviction/context policy.

## What a pose token should contain

### Do not throw away the pose estimator's uncertainty

A conventional SDK returns hard landmarks. A deeper integrated model should tap the
pose estimator before the final argmax. Let:

- \(F_t\in\mathbb R^{H\times W\times C}\) be its intermediate visual feature map;
- \(L_{tj}(u,v)\) be the heatmap/logits for joint \(j\);
- \(\pi_{tj}=\operatorname{softmax}(L_{tj})\) be the joint location distribution.

Then compute differentiable location and uncertainty:

\[
\mu_{tj}=\sum_{u,v}\pi_{tj}(u,v)[u,v],\qquad
\Sigma_{tj}=\sum_{u,v}\pi_{tj}(u,v)([u,v]-\mu)([u,v]-\mu)^T.
\]

Sample local appearance around the joint:

\[
a_{tj}=\sum_{u,v}\pi_{tj}(u,v)F_t(u,v).
\]

The initial joint token is:

\[
g_{tj}=W_g[	ilde\mu_{tj},\Sigma_{tj},c_{tj},a_{tj},
\Delta\tilde\mu_{tj},\Delta^2\tilde\mu_{tj},b_{tj},m_{tj}]
+e_j+e_{part(j)}.
\]

Here \(b\) contains bone/angle relationships, \(m\) is a missingness mask, and the
coordinates are root-relative and scale-normalized while preserving camera/global
motion in separate channels. This token tells the decoder not only *where the knee
was estimated*, but whether the estimate was broad, occluded, visually ambiguous, or
moving inconsistently.

If MediaPipe remains a black-box provider in the first prototype, construct the same
interface from its image/world landmarks, confidence, derivatives and feature quality.
Later, replace it with an exportable pose network whose heatmaps and feature maps are
available for end-to-end training.

### Joint graph and learned resampling

Two small graph-attention blocks exchange information along anatomical edges:

\[
G_t^{l+1}=G_t^l+operatorname{GAttn}(G_t^l,A_{body})
+\operatorname{MLP}(G_t^l).
\]

A learned query resampler converts 33 joint tokens into a small number of motion
tokens:

\[
M_t=Q+\operatorname{softmax}\left(\frac{Q(G_tW_K)^T}{\sqrt d}\right)G_tW_V.
\]

For a 25M system, start with one global token per 100 ms plus sparse part tokens. One
512-D vector can comfortably carry the information in 33 structured landmarks; the
decoder then supplies temporal context. Add five part tokens (torso, left/right arm,
left/right leg) only every 500 ms or when their feature change exceeds a learned
threshold. This preserves joint-specific detail without consuming six tokens for
every frame.

An initial pose adapter can remain around 0.2-0.8M parameters:

- 64-D joint embeddings;
- two 64-D graph/attention blocks;
- one global and five part queries;
- a 64→512 projection;
- modality/time/view embeddings.

That produces a roughly 26-27M complete system while keeping Nexus intact.

## How the pose enters the decoder

### Recommended: PaLM-E/LLaVA-style interleaved soft tokens

Nexus already uses `LlamaForCausalLM`, whose transformer accepts `inputs_embeds`.
Replace positions reserved for pose observations with continuous pose embeddings:

\[
E_i=\begin{cases}
W_{word}[token_i]+e_{text}, & token_i\text{ is text/control}\\
W_{pose}M_t+e_{pose}+e_{time}(t), & token_i\text{ is motion}\\
W_{profile}r+e_{profile}, & token_i\text{ is calibration/profile}.
\end{cases}
\]

The mixed sequence might be:

```text
<BOS>
<PLAN> right heel slide; target 80; maximum 90; 3x10 </PLAN>
<CAL> ...learned baseline tokens... </CAL>
<POSE t=0.0> <POSE t=0.1> ... <POSE t=1.4>
<ACT:REP_START>
<POSE t=1.5> ...
<ACT:CUE_HIP_STABLE> <POSE ...>
<ACT:REP_COMPLETE> <REP_INDEX:3>
<USER> Am I lifting my hip? </USER>
<ASSISTANT> ...
```

Pose tokens are not strings and do not need BPE vocabulary IDs. They are 512-D soft
embeddings occupying causal positions. Action/state tokens can be ordinary learned
vocabulary entries, or predictions from parallel heads.

All tokens pass through the same self-attention:

\[
H^{l+1}=H^l+operatorname{CausalGQA}
(H^lW_Q,H^lW_K,H^lW_V)+\operatorname{SwiGLU}(H^l).
\]

Therefore an action or word generated now can attend directly to the complete retained
motion history, plan, profile, previous actions and dialogue. That is the actual
vision-motion-decoder fusion.

### Alternatives and why they are not the first experiment

| Fusion | Advantage | Cost/problem at 25M |
|---|---|---|
| Direct soft-token projection | Simplest, end-to-end, reuses every Nexus block | Pose consumes main context |
| Q-Former/resampler prefix | Compresses variable joints/windows into fixed tokens | Additional transformer; prefix alone is awkward for an infinite stream |
| Flamingo-style gated cross-attention | Keeps motion memory separate; preserves frozen LM | Cross-attention adapters can add several million parameters and new runtime operators |
| Discrete VQ motion vocabulary | Motion can be both input and autoregressive output using ordinary CE | Codebook error/jitter; tokenizer must be trained first |
| Continuous latent + flow head | Avoids quantization and enables high-fidelity future motion | More complex training/runtime; use after the recognition system works |
| Modality-specific Mixture-of-Transformers | Protects language capacity, as in LLaMo | Duplicates substantial block parameters; better at 60-100M |

Direct interleaving is the right Nexus experiment. Add cross-attention only if context
pressure or interference is measured, not assumed.

## One decoder, several native output spaces

A VLM is not less unified because it uses an image encoder and a language head. In the
same way, one Nexus hidden state can drive several modality-appropriate heads:

\[
h_t=\operatorname{Nexus}(E_{\le t};KV_{t-1}).
\]

### 1. Instantaneous structured state

These are parallel, one-forward-pass predictions:

\[
\begin{aligned}
p(phase_t)&=\operatorname{softmax}(W_ph_t)\\
p(error_{t,k})&=\sigma(W_eh_t)\\
p(action_t)&=\operatorname{softmax}(W_ah_t)\\
(\cos\phi_t,\sin\phi_t)&=W_\phi h_t\\
(\hat\mu_t,\hat\sigma_t)&=W_u h_t.
\end{aligned}
\]

This can immediately produce phase, rep transition, error type, body part, severity,
uncertainty and recommended action. A deterministic serializer can turn those neural
fields into guaranteed-valid JSON without another model pass. That is still a decoded
model decision, just as bounding-box heads are native outputs of vision models.

### 2. Autoregressive structured actions

Add a compact action vocabulary:

```text
<PHASE_READY> <PHASE_OUT> <PHASE_RETURN>
<REP_START> <REP_COMPLETE> <REP_PARTIAL>
<ERR_HIP_ROTATION> <ERR_KNEE_FLEXION> <ERR_UNOBSERVABLE>
<SEV_0> ... <SEV_3>
<CUE_HOLD> <CUE_SLOW> <CUE_REPOSITION> <CUE_STOP>
```

RT-2's core idea applies directly: train actions as tokens in the same next-token loss
as language. A decision can be 3-8 semantic tokens, then serialized as JSON:

```json
{"phase":"return","rep_delta":0,"error":"hip_rotation","action":"cue_stable"}
```

Use a JSON grammar or finite-state logit mask during generation. Do not ask a
6,400-token language vocabulary to rediscover the schema character by character when
low latency matters.

### 3. Future motion and counterfactual correction

The decoder can predict a future latent distribution, not only classify the present:

\[
p(M_{t+1:t+K}\mid M_{\le t},plan,profile).
\]

Three implementations are worth comparing:

- regression or mixture-density head for normalized future joints;
- VQ motion codes and next-code cross-entropy, following PoseGPT/MotionGPT;
- continuous autoregressive latents with a small flow-matching head, following LLaMo.

From the predicted trajectory, estimate `probability of exceeding the prescribed
limit within 500 ms`, `expected endpoint`, and `likely incomplete return`. For visual
coaching, condition on a correction token and decode the desired future skeleton:

\[
\hat P^{correct}_{t+1:t+K}
=D_{motion}(h_t,\langle CORRECT\_HIP\_STABLE\rangle).
\]

That skeleton can drive an overlay or avatar. This is a materially richer output than
a summary.

### 4. Grounded language and dialogue

Because motion tokens and dialogue coexist in the same KV cache, the user can ask:

- “Was the last rep deeper than the second?”
- “Why did you not count that one?”
- “Tell me when I begin rotating my hip.”
- “Only speak if I am likely to cross 90 degrees.”

The model can answer or install an action condition grounded in the retained stream.
Training must include current-perception, backward-tracing and future-trigger tasks,
not only captions.

## Streaming execution: how it makes an instant decision

Most VLM demonstrations are not literally reconsidering an infinite camera stream.
They encode sampled frames, prefill visual tokens, then autoregressively decode. A
streaming body model instead performs one cached decoder step for each accepted pose
token:

```python
pose_embedding = pose_tokenizer(new_pose, timestamp)
hidden, kv_cache = nexus.step(pose_embedding, kv_cache)
state = state_heads(hidden)        # immediate, no text generation

if should_emit_action(state):
    action = action_head(hidden)   # one pass or a few semantic tokens
if user_asked_question:
    response = generate_from(kv_cache.fork(), constrained_schema_or_text)
```

The previous 1,000 positions are not recomputed; their keys and values are cached.
Current streaming VLM work follows the same principle, retaining attention sinks and
selected recent visual/text KV states rather than repeatedly prefilling overlapping
windows ([StreamingVLM](https://arxiv.org/abs/2510.09608)).

### Do not block perception while generating language

A single autoregressive sequence has a concurrency problem: generating 20 words can
take longer than the next pose interval. Use two cache branches with shared weights:

- the **live cache** continues accepting pose tokens and producing state/actions;
- a **response snapshot** forks at the question/event and generates JSON or language
  asynchronously.

This is still one trained model. It is two inference streams through the same weights,
analogous to serving two continuations of one prompt. Critical action tokens come from
the live branch; cancel or revise a stale language branch if new motion invalidates it.

## What 1,024 context positions actually mean

If one pose token represents 100 ms:

| Stream format | Effective token rate | Raw temporal horizon in 1,024 positions |
|---|---:|---:|
| one pose token at 10 Hz | 10/s | 102.4 s |
| pose + action token every frame | 20/s | 51.2 s |
| four pose tokens at 10 Hz | 40/s | 25.6 s |
| one pose token at 30 Hz | 30/s | 34.1 s |

The plan, profile and generated language also consume positions. A workable 1,024
budget is:

| Allocation | Positions |
|---|---:|
| attention sink + prescription/profile | 96 |
| persistent event/memory tokens | 128 |
| recent fine-grained pose/action stream | 672 |
| dialogue/response reserve | 128 |

At approximately 10-15 motion tokens/s, the fine-grained window covers about 45-67
seconds—several slow repetitions. Older history is not simply erased. Compress an
evicted block into persistent recurrent memory tokens:

\[
R_{k+1}=\operatorname{Resampler}(Q_R,[R_k;H_{evicted}]).
\]

Keep exact sparse event tokens such as rep boundaries, error episodes and calibration
updates; compress dense old poses. This is internal hierarchical memory, not a
post-hoc output summary. For session-long exact counts, also retain an explicit
monotonic counter value as a control token or external state—transformer memory should
not be trusted to preserve an exact integer through arbitrary cache eviction.

Streaming video research reports that most adjacent visual tokens are redundant; one
2025 method removed 82.8% while retaining 98% of its baseline result on its benchmark
([TimeChat-Online](https://arxiv.org/abs/2504.17343)). Pose streams are even more
structured, so train a differential token gate:

\[
keep_t=\mathbf1[\|M_t-M_{last}\|>\tau\;\lor\;u_t>\tau_u\;\lor\;boundary_t>\tau_b].
\]

Never drop boundary, uncertainty-spike, direction-change, or constraint-relevant
tokens. Train with the same dropping/eviction pattern used at inference.

## Exact Nexus KV and compute implications

Nexus has 8 layers, 2 KV heads and head dimension 64. KV memory per retained position
is:

\[
2(K,V)\cdot8\cdot2\cdot64=2048\text{ scalar values}.
\]

Therefore a 1,024-position cache is approximately:

- 2.0 MiB at FP16;
- 1.0 MiB at INT8;
- plus cache metadata/alignment.

For one cached token at full 1,024 context, approximate dense work is:

- 22.55M MAC for Nexus block matrices;
- up to about 8.39M additional attention MAC from querying/reading 1,024 cached
  positions across eight layers;
- 3.28M MAC for the full vocabulary projection, only when language/action logits over
  the full vocabulary are requested.

At 10 pose tokens/s, continuous Nexus block work is therefore on the rough order of
225-310M MAC/s depending on active context; 15 Hz is roughly 338-465M MAC/s. This is
phone-feasible territory but must be measured because one-token recurrent execution
can underutilize NPUs and incur delegate-launch overhead. The continuous state path
should skip the 6,400-way LM projection and use small heads. Run the full LM head only
for action/language generation.

INT8 Nexus weights are about 25.8 MB, FP16 about 51.7 MB; add the pose adapter,
activations and runtime. The principal sustained cost is repeatedly reading decoder
weights, not the 1-2 MB KV cache. Benchmark CPU, GPU and NPU paths separately on target
phones for p50/p95 step latency, 20-minute thermal behavior and battery energy.

## Training the unified model

### Stage A — motion tokenizer pretraining

Use millions of unlabeled real/synthetic pose windows. Train the pose network/tokenizer
with:

1. masked joint and masked time-span reconstruction;
2. denoising from production-like pose corruptions;
3. next latent/pose prediction;
4. tempo-invariant phase/cycle contrastive learning;
5. view/side-consistent embedding with view-sensitive reconstruction;
6. motion-text contrastive alignment when captions/instructions exist.

\[
\mathcal L_A=\lambda_{mask}L_{mask}+\lambda_{denoise}L_{denoise}
+\lambda_{next}L_{next}+\lambda_{cycle}L_{cycle}
+\lambda_{align}L_{motion-text}.
\]

Pretrain on general motion, then emphasize the target rehabilitation exercises. A
well-pretrained language decoder is not automatically a well-pretrained motion model;
the modality adapter needs its own broad pretraining pressure.

### Stage B — frozen-Nexus alignment

Freeze Nexus and train only the pose tokenizer/projector and new heads. Given mixed
pose-text examples, require Nexus to recover:

- exercise and phase tokens;
- motion captions and temporal questions;
- next-motion latents;
- structured action/error tokens;
- whether a requested fact is currently observable.

This forces pose embeddings into a representation the existing decoder can use without
immediately overwriting Nexus's JSON/language circuits.

### Stage C — multimodal causal pretraining

Unfreeze progressively, then full-train at a lower learning rate with text/JSON replay.
Each training row is an actual causal stream, not a trimmed clip plus a final label:

```text
[plan/profile tokens]
[pose 0 ... pose t]
-> state/action at t
-> future pose t+1:t+k
-> optional question/JSON/language
```

Use random causal cut points so the model learns what can and cannot be known at every
moment. Include negative future-trigger samples: before the event, the correct action
is `WAIT`, not a premature cue.

The joint objective is:

\[
\begin{aligned}
\mathcal L={}&\lambda_{LM}CE(y^{text})
+\lambda_{act}CE(y^{action})
+\lambda_{phase}CE(y^{phase})
+\lambda_{boundary}Focal(y^{boundary})\\
&+\lambda_{error}BCE(y^{error})
+\lambda_{progress}(1-\hat v_\phi\cdot v_\phi)
+\lambda_{future}L_{motion}
+\lambda_{cal}L_{calibration}
+\lambda_{retain}L_{text/JSON\ replay}.
\end{aligned}
\]

Important example families:

- current perception: “what phase/error is visible now?”;
- backward trace: “when did hip rotation begin?”;
- future response: “cue only if the trajectory is likely to cross the limit”;
- counterfactual: corrupted versus corrected motion pair;
- reference comparison: user rep versus clinician demonstration;
- personalization: same motion may be acceptable or unacceptable under different
  confirmed plans/baselines;
- missingness: visually identical landmark streams with different confidence/camera
  evidence should yield different observability;
- action-language equivalence: the same hidden situation maps to both a compact action
  token and an approved verbal realization;
- continuous background, pauses, aborted attempts and set restarts—not only trimmed
  successful repetitions.

### Stage D — teacher distillation and hard negatives

Use a larger temporal/vision teacher offline to label embeddings, uncertainty and
future distributions, but keep expert/human annotations authoritative for semantic
errors. Distill hidden states or distributions, not just teacher prose:

\[
L_{KD}=T^2 KL(p_{teacher}^{(T)}\|p_{Nexus}^{(T)})
+\|W_dh_{Nexus}-h_{teacher}\|_2^2.
\]

Hard negatives are essential: same exercise/different phase, same knee angle/different
hip compensation, same coordinates/different visibility, full versus partial return,
and camera motion versus body motion.

### Stage E — streaming-aware quantization

Fine-tune with the exact 1,024-position eviction policy, differential token dropping,
timestamp jitter and cache layout. Apply QAT or calibrated INT8 and re-measure each
head, boundary latency and action calibration. Offline full-attention accuracy does not
validate the deployed causal cache.

## Distillation: how a 25M model acquires capabilities it cannot discover cheaply

Distillation should occur at multiple interfaces, not only by asking a large VLM to
write answers that Nexus imitates.

### 1. Perception distillation

A privileged teacher sees synchronized multiview video, motion capture/IMUs and the
deployment camera. The student sees only the deployment phone frame. Distill:

- joint heatmaps and covariance, not only point coordinates;
- 3D pose and camera/view factors;
- occlusion and observability;
- intermediate joint/part features.

This teaches a monocular model which uncertainty to expose to the decoder. It cannot
make genuinely unobservable geometry observable; the target for those cases is a broad
distribution or abstention.

### 2. Temporal representation distillation

Use a larger offline RGB+pose teacher with future context during training. Align the
causal Nexus representation to a teacher representation computed at the same cutoff:

\[
L_{repr}=\|W_d h^{student}_t-operatorname{sg}(h^{teacher}_{t,causal})\|_2^2
+L_{contrastive}(h_s,h_t).
\]

An offline/bidirectional teacher may also supply retrospective labels, but do not
distill its future-leaking confidence as if the student could know it online. Create a
separate `event later confirmed` target from the teacher and measure detection delay.

### 3. Output and policy distillation

For every causal prefix, distill calibrated distributions rather than only the top
answer:

\[
L_{logit}=T^2 KL(\pi_{teacher}^{(T)}(\cdot|o_{\le t})\|\pi_{Nexus}^{(T)}(\cdot|o_{\le t})).
\]

Apply it independently to phase, error, action, observability, future trajectory and
language/action tokens. Retain supervised ground-truth loss because a teacher can be
confidently wrong. Weight or reject teacher examples using agreement with reference
sensors and clinician labels.

### 4. World-model/trajectory distillation

The large teacher predicts a distribution over future poses and possible responses to
each approved cue:

\[
p_T(P_{t+1:t+K}\mid P_{\le t},a_t,profile).
\]

Distill this into Nexus's future-motion head. This is what enables anticipation and
lets a later policy estimate `wait versus cue now` without executing every candidate
on a person. The counterfactual predictions are only as causal as the intervention
data: observational sessions alone confound natural improvement with cue effect.

### 5. Sequence and language distillation

Use a capable teacher to propose rationales, structured errors and language variants,
but train Nexus on the approved decision/action schema first. Distill hidden
distributions or short semantic action sequences; treat prose chain-of-thought as
synthetic augmentation, not clinical ground truth. Nexus's existing exact-JSON and
copy behavior is an asset worth retaining through text/JSON replay.

The full student objective becomes:

\[
L_{student}=L_{human\ labels}+\alpha L_{perceptionKD}+\beta L_{reprKD}
+\gamma L_{logitKD}+\delta L_{trajectoryKD}+\eta L_{retention}.
\]

## Reinforcement learning: learning when and how to intervene

Supervised learning teaches `what state is visible`. RL is valuable for the sequential
policy question: `given uncertain motion and previous interactions, which action now
maximizes useful coaching while minimizing harm and nuisance?`

### Formulate the live system as a constrained POMDP

The person and unobserved physical state make the problem partially observable:

\[
o_t=(pose_t,confidence_t,plan,profile,dialogue),\quad
h_t=Nexus(o_{\le t}),\quad a_t\sim\pi_\theta(a|h_t).
\]

Approved actions are deliberately small and typed:

```text
WAIT, COUNT_REP, MARK_PARTIAL, CUE_<type>, REPOSITION_CAMERA,
ASK_USER, PAUSE_ASSESSMENT, STOP_AND_ESCALATE, SPEAK_<template>
```

The reward must reflect a trajectory, not an isolated classifier score:

\[
\begin{aligned}
r_t={}&w_1\,\Delta quality_{after\ cue}
+w_2\,correct\_rep\_event
+w_3\,timely\_useful\_cue\\
&-w_4\,false\_reassurance
-w_5\,incorrect\_count
-w_6\,nuisance\_cue
-w_7\,late\_cue
-w_8\,unjustified\_confidence.
\end{aligned}
\]

Safety penalties should dominate convenience rewards, and hard contraindications or
unobservable states constrain the action set before sampling:

\[
\pi'(a|h)\propto\pi(a|h)\mathbf1[a\in A_{allowed}(h)].
\]

RL must not be allowed to “learn” that silence scores well by never assessing, or that
constant warnings avoid false reassurance. Optimize and publish the complete
coverage-risk-nuisance frontier.

### The practical RL ladder

1. **Behavior cloning / action SFT.** Train action tokens from therapist-authored and
   deterministic policy trajectories. This initializes a safe support.
2. **Preference learning.** Therapists compare two proposed actions/cue timings for
   the same causal prefix. Train a reward/preference model. DPO-like training is useful
   for short action/language continuations; it does not replace temporal outcome data
   ([DPO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)).
3. **Offline policy optimization.** Optimize only within logged action support using
   conservative offline RL, advantage-weighted regression or a return-conditioned
   Decision-Transformer-style sequence. Penalize actions far from the behavior policy
   ([Decision Transformer](https://papers.nips.cc/paper_files/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html),
   [CQL](https://papers.nips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)).
4. **Simulator rollouts.** Use the distilled future-motion/world head and physical
   simulation to stress pause/count/cue timing, rare state transitions and camera loss.
   Randomize human response to cues. Simulation reward trains policy timing, not
   clinical truth.
5. **Policy distillation.** Distill an ensemble/planner or larger RL policy into the
   Nexus action-token distribution for phone inference.
6. **Prospective shadow evaluation.** Run the learned policy without delivering its
   actions; compare with clinicians and the existing safe policy.
7. **Constrained live study.** Only approved low-risk action classes are exposed in a
   reviewed protocol. Start with contextual-bandit choices among equivalent cue
   wording/timing—not unconstrained backbone updates.
8. **Personal adaptation.** Update a tiny per-user adapter, calibration state or bandit
   policy under hard limits. Never silently online-train the global perception/backbone
   on one patient's outcomes.

Safe offline RL remains an active research problem because out-of-distribution actions
can have spuriously high estimated value. Constraint-conditioned IQL work in 2025 is
directly relevant as an algorithmic reference, not evidence that this clinical policy
is safe ([C2IQL](https://proceedings.mlr.press/v267/liu25ai.html)).

### Where RL does not belong

- discovering pose geometry that dense supervision can teach more efficiently;
- defining whether a motion is medically safe without authoritative labels;
- exploring unsafe errors on patients;
- compensating for an uncalibrated perception model;
- optimizing only token-level “helpfulness” while ignoring future movement and false
  reassurance.

The native token sequence is naturally compatible with policy learning:

```text
PLAN, POSE_1 ... POSE_t, ACTION_t, OUTCOME_t+1:t+k, REWARD/QUALITY_CHANGE
```

The same Nexus weights can learn next action, next outcome and language. That is a
small embodied motion-language-action model, not a captioner.

## Why the exact system has been rare

There are examples of nearly every component, but few examples at their intersection.
It is inaccurate to say body pose has never been connected to a decoder: PoseGPT,
MotionGPT, LLaMo, Superman, PaLM-E, RT-2, streamable gesture models and multiple
vision-language-motion systems do so in different ways. What remains unusual is:

> noisy monocular phone pose → native causal decoder tokens → continuous fine-grained
> assessment, anticipation and intervention → calibrated on-device policy, at 25M.

The gap exists for structural reasons:

1. **Benchmark fragmentation.** Pose estimation rewards joint error; skeleton action
   recognition rewards clip labels; action-quality assessment rewards one score;
   motion-language work rewards caption or generation quality; rehabilitation needs
   causal boundaries, observable localized errors, intervention timing and abstention.
2. **The datasets do not contain decisions.** Motion-capture/text datasets rarely have
   continuous camera failures, set restarts, therapist cue timestamps and the person's
   response after each cue. RL additionally needs action-outcome variation, not only
   demonstrations.
3. **Offline leakage is easier.** Published motion models commonly see the completed
   clip. A live model must decide before future evidence, survive cache eviction and
   distinguish `not yet knowable` from `negative`.
4. **Hard landmarks are an awkward interface.** SDK coordinates discard heatmap
   uncertainty and appearance; black-box SDKs cannot be trained end-to-end. Researchers
   either stay with specialized GCN/TCN models or return to expensive RGB encoders.
5. **Small decoders lack motion pretraining.** VLMs inherit internet-scale image-text
   alignment. There is no equally large, diverse and commercially clear corpus of
   pose-language-action-outcome streams. A 25M model must obtain this through a
   purpose-built tokenizer curriculum and distillation.
6. **Runtime systems run at different clocks.** Camera, detector, tracker, pose,
   recurrent decoder, speech and UI have different rates and hardware delegates.
   Research code often benchmarks one module, not a thermally sustained phone loop.
7. **Real-time output evaluation is missing.** “Correct eventually” is insufficient;
   the product cares about early/late cues, false alerts per minute, action effects and
   user interruption cost.
8. **Clinical exploration is constrained.** Robotics can collect repeated policy
   rollouts in simulation or on machines. A rehabilitation model cannot freely explore
   actions or ask patients to perform unsafe errors, slowing RL and causal validation.
9. **The specialized baseline is strong.** For three known exercises, a TCN plus state
   machine is cheap and auditable. A native decoder must prove added anticipation,
   transfer, dialogue or personalization—not merely match rep counting.
10. **2026 is when the architecture is visibly converging.** LLaMo and Superman now
    demonstrate unified temporal motion-language or vision-skeleton modeling, while
    streaming VLMs address cache and real-time interaction. The missing work is to
    combine those advances under tiny-model, phone and rehabilitation constraints.

This scarcity is the opportunity, but also the reason the research program needs
multiple stages rather than one supervised fine-tune.

## The data object should be a causal episode, not a labeled clip

Each stored episode needs aligned channels:

```json
{
  "plan": {...},
  "profile": {...},
  "frames": [
    {
      "timestamp_ms": 0,
      "pose_distribution": "...",
      "provider_features": "...",
      "phase": "ready",
      "progress": 0.0,
      "visible_errors": [],
      "observable": true
    }
  ],
  "events": [...],
  "questions": [...],
  "approved_actions": [...],
  "reference_motion": "..."
}
```

Training samples are random prefixes of this episode. If the model only sees completed
reps during training, it will learn offline narration rather than online perception.

Dataset scale should be counted in independent participants, error events and causal
episodes—not frames. Large unlabeled motion pools can pretrain the tokenizer, but each
product-critical error needs enough distinct real people and environments to separate
the error from identity, camera and clinician shortcuts. Synthetic data supplies
coverage and counterfactual pairs; final calibration and every safety claim come from
held-out real participants/devices/sites.

## “One model” has three legitimate boundaries

1. **One semantic decoder:** frozen external pose SDK emits structured observations;
   Nexus consumes native pose tokens and emits all decisions. Fastest research path.
2. **One end-to-end trainable multimodal network:** exportable pose encoder heatmaps and
   Nexus are jointly optimized with intermediate pose and downstream losses. Strongest
   scientific fusion claim.
3. **One compiled deployment graph:** pose encoder and decoder are trained in stages but
   exported together. This may be worse operationally because pose runs at a different
   rate/delegate from the recurrent decoder.

VLMs themselves usually contain multiple trained modules—vision encoder, projector,
decoder—and are still one multimodal model. “One model” should mean shared learned
representations and end-to-end objectives, not that every operation must be fused into
one runtime file.

## The experiment that answers whether this idea actually works

Compare under the same causal input and phone budget:

| Arm | What it tests |
|---|---|
| rules/current Relay TCN | current baseline |
| TCN + rep summary + Nexus | the previous, weaker post-hoc architecture |
| Nexus with text-serialized landmarks | whether structured text alone suffices |
| Nexus + one continuous global pose token/frame | direct VLM-style fusion |
| Nexus + global/dynamic part tokens | fine-grained adaptive fusion |
| previous + action/state heads | instant decisions without AR latency |
| previous + future-motion head | anticipation and counterfactual correction |
| discrete VQ motion-token Nexus | unified input/output motion vocabulary |
| continuous latent/flow Nexus | quantization-free motion prediction |
| 60-100M modality-specific model | capacity ceiling |

Evaluate current, backward-tracing and future-trigger queries; phase/boundary/rep/error
metrics; future trajectory error; JSON/action exactness; calibration/abstention;
decision latency; KV/weight memory; sustained energy and heat. The central ablation is
whether native motion tokens improve fine-grained and anticipatory decisions over a
post-hoc rep summary at equal causal latency.

## Recommended first implementation

Build **Nexus-Body-27M v0**:

- existing 25.829888M Nexus weights;
- black-box MediaPipe landmarks initially, preserving image/world coordinates,
  confidence, derivatives and timestamps;
- approximately 0.3M graph/resampler pose adapter producing one 512-D global token at
  10 Hz and dynamic part tokens;
- modality and elapsed-time embeddings;
- phase/progress/boundary/error/uncertainty/action heads;
- 64-128 added semantic action tokens for constrained generation;
- 1,024-position GQA cache with plan sink, recent motion, persistent event tokens and
  trained eviction;
- a forkable response cache for simultaneous continuous perception and JSON/dialogue;
- INT8 phone deployment after FP16 proof.

Train it first on dense causal state/action prediction and next-motion pretraining.
Then add language/JSON instruction tuning. In parallel, train a VQ motion vocabulary as
the second input/output representation. Only after those results should the pose
estimator be unfrozen and replaced with an end-to-end heatmap/visual-feature interface.

That is the direct analogue of turning a text decoder into a VLM—except the new native
modality is articulated human motion, and the outputs include actions, trajectories
and structured decisions in addition to language.
