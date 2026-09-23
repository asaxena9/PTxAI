# A pose-native streaming decoder: the VLM analogy carried all the way through

**Evidence cutoff:** 2026-08-07  
**Question:** Can continuous 2D/3D body pose be a native modality inside a
25M-parameter decoder, with online decisions, structured outputs, forecasting,
and actions—not merely a post-hoc movement summary?

## The direct answer

Yes. A summary-only architecture is not the limit of the idea. It is one
application choice, and for this project it is too narrow.

The technically faithful analogue of a vision-language model is:

```text
RGB frame -> pose estimator -> pose/joint encoder -> projected pose token(s)
                                                    |
instruction/profile tokens ------------------------+
                                                    v
                          one causal multimodal decoder
                                                    |
             +----------------+---------------------+------------------+
             |                |                     |                  |
       online state      event/action          next motion       JSON/language
       every frame       when needed           prediction        when requested
```

The decoder does **not** see one isolated moment. At time `t`, the new pose
token attends to the cached keys and values of all retained pose, instruction,
event, and output tokens. Its decision is therefore conditioned on the movement
history. This is how an autoregressive decoder processes a continuous stream.

There is already strong evidence for every important piece of this design:

- [MotionGPT (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html)
  turns continuous 3D motion into VQ tokens and adds them to a text vocabulary,
  allowing one encoder-decoder language model to read and generate either motion
  or language.
- [MotionLLM (2024)](https://arxiv.org/abs/2405.20340) uses a motion encoder and
  a learned linear motion-to-language projection, directly paralleling the
  image-encoder/projector/LLM construction used in VLMs.
- [LLaMo (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Human_Motion_Instruction_Tuning_CVPR_2025_paper.html)
  argues that motion should remain a native continuous modality because discrete
  or textual conversion can erase fine spatiotemporal details. Its Cross Talker
  performs text-conditioned selection and fusion of native motion frames.
- [Superman (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Superman_Unifying_Skeleton_and_Vision_for_Human_Motion_Perception_and_CVPR_2026_paper.html)
  is the clearest existence proof as of the cutoff: one decoder-only MLLM accepts
  video, skeleton sequences, and text, and autoregressively emits 3D pose tokens
  for pose estimation, motion prediction, and motion in-betweening. It uses a
  vision-guided VQ motion vocabulary. It is a 7B offline research system, not an
  edge implementation, but it proves that perception and temporal skeleton
  generation can inhabit one conditional sequence model.
- [VideoLLM-online (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_VideoLLM-online_Online_Video_Large_Language_Model_for_Streaming_Video_CVPR_2024_paper.html)
  shows how a VLM becomes genuinely online: one projected token per frame,
  continuous KV cache, a learned `respond/silent` decision on frame tokens, and
  frame buffering while language is generated.
- [RT-2 (2023)](https://arxiv.org/abs/2307.15818) shows that an autoregressive
  vision-language model can emit actions, not only prose, by expressing actions
  as tokens in the same output vocabulary. [FAST (2025)](https://arxiv.org/abs/2501.09747)
  further shows that continuous time-series actions can be efficiently compressed
  into autoregressive tokens instead of binning every scalar at every time step.

What does **not** yet exist in the literature is the exact combination proposed
here: a roughly 25M, pose-native, streaming decoder trained on untrimmed physical
therapy sessions and validated at edge latency. That gap is the research project.

## What a visual token really is—and why pose can be one

A VLM usually does not turn an image into English before the language decoder.
The image encoder emits continuous vectors and a projector maps them to the
decoder residual width. The decoder treats those vectors as input positions.
[LLaVA](https://arxiv.org/abs/2304.08485) demonstrates the simple projection
approach; [BLIP-2](https://proceedings.mlr.press/v202/li23q) uses a learned
Q-Former to extract a fixed number of visual vectors; and
[Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)
uses a Perceiver Resampler and gated cross-attention.

Pose can be injected by exactly the same mechanisms. It does not have to be
converted to words, JSON, angles, or a summary first.

Assume a pose estimator supplies `J` joints at timestamp `t`:

\[
P_t=\{(x_{tj},y_{tj},z_{tj},c_{tj},m_{tj})\}_{j=1}^{J},
\]

where `c` is confidence and `m` is a missing/occluded flag. For each joint,
construct a body-native feature rather than a prose description:

\[
\xi_{tj}=\left[
\frac{p_{tj}-r_t}{s_t},
\frac{\Delta p_{tj}}{s_t},
\frac{\Delta^2p_{tj}}{s_t},
b_{tj},c_{tj},m_{tj},\Delta t
\right].
\]

`r_t` is a root such as the pelvis, `s_t` is body scale, and `b_tj` contains
bone-relative geometry. Retain a separate global position/view feature instead
of destroying it during root normalization.

A small skeleton stem produces a frame representation:

\[
u^0_{tj}=W_x\xi_{tj}+e_j,
\]

\[
U_t^{\ell+1}=U_t^\ell+
\operatorname{GraphAttn}_{A}(\operatorname{LN}(U_t^\ell))+
\operatorname{FFN}(\cdot),
\]

\[
s_t=\operatorname{Attn}(q_{frame},U_t),\qquad
\bar s_t=\operatorname{CausalTemporalStem}(s_{t-2:t}),
\]

\[
e_t^{pose}=W_p\bar s_t+e_{modality}+e_{time}(t).
\]

`A` carries anatomical adjacency and left/right symmetry. The important result is
that `e_pose` has the same width `d_model` as a text-token embedding. It can be
interleaved directly with text and event tokens:

```text
<BOS> <SYSTEM> prescription/profile <LIVE>
<POSE t=0.0> <POSE t=0.1> ... <POSE t=3.4>
<EVENT rep_start> <POSE t=3.5> ...
```

The `<POSE>` entries above are continuous vectors supplied by the pose stem,
not necessarily integer vocabulary IDs.

## Why the decoder understands continuity

At decoder layer `l`, the new pose position computes:

\[
q_t^l=h_t^{l-1}W_Q^l,
\quad k_t^l=h_t^{l-1}W_K^l,
\quad v_t^l=h_t^{l-1}W_V^l.
\]

It attends to the retained past cache:

\[
h_t^l = h_t^{l-1}+
\operatorname{softmax}\!\left(
\frac{q_t^l[K_{\le t}^l]^\top}{\sqrt{d_h}}
\right)V_{\le t}^l.
\]

The cached keys and values are the temporal memory. A rep-completion decision is
therefore

\[
P(\text{rep complete at }t\mid P_{1:t},I,C)
=\operatorname{softmax}(W_{rep}h_t),
\]

not `P(rep | P_t)`. Here `I` is the instruction/prescription and `C` is patient,
session, or prior-event context. A decoder is only "one moment in, one moment
out" if it is deliberately invoked without history.

KV caching changes the implementation cost, not the mathematical conditioning.
The prior tokens are not recomputed; their per-layer keys and values remain in
memory, and only the new pose position is propagated through the blocks.

## One model, four output clocks

The same fused hidden state should support several output modes. Reducing the
model to a prose summary throws most of them away.

### 1. Continuous state, every accepted pose

At 5–15 Hz, small heads on `h_t` can emit:

\[
\begin{aligned}
p_t^{phase}&=\operatorname{softmax}(W_{phase}h_t),\\
p_t^{boundary}&=\sigma(W_{boundary}h_t),\\
\hat q_t^{progress}&=\sigma(W_{progress}h_t),\\
p_t^{deviation}&=\sigma(W_{deviation}h_t),\\
p_t^{action}&=\operatorname{softmax}(W_{action}h_t),\\
\hat u_t&=W_{uncertainty}h_t.
\end{aligned}
\]

Possible online outputs include exercise identity; inbound/outbound/hold phase;
cyclic phase; movement amplitude; velocity; rep start/complete/partial/aborted;
form deviation by body region; camera/pose observability; predicted next phase;
time to endpoint; cue-now/wait/reposition/stop/abstain; and uncertainty.

These heads are not a separate reasoning model. They are alternative decoders of
the same multimodal transformer state, just as a VLM has an LM head and a VLA can
have an action head.

### 2. Event tokens

At significant instants, the same LM head can emit a compact action/event string:

```text
<REP_END> <INVALID> <KNEE_VALGUS> <CUE_OUTWARD> <CONF_82>
```

This is autoregressive action decoding in the RT-2 sense. A grammar or vocabulary
mask makes invalid sequences impossible. Five compact tokens can be decoded much
faster than a 40-token human-readable JSON object.

### 3. JSON and natural language

The compact event can be serialized into JSON immediately:

```json
{
  "event": "rep_end",
  "valid": false,
  "deviation": "knee_valgus",
  "action": "cue_outward",
  "confidence": 0.82
}
```

The **decision** came from the unified decoder. Serialization does not reduce it
to a summary. If generated JSON is required, constrain autoregressive decoding to
a schema. But at 10 output tokens/s, 30 JSON tokens take about three seconds. No
VLM escapes this token-by-token latency. "Instant" VLM behavior means fast
time-to-first-token after visual prefill, not zero-time completion of an output
sequence. For real-time control, fixed structured heads or compact action tokens
are the technically correct output channel.

### 4. Motion generation and counterfactuals

With a learned motion vocabulary or continuous trajectory head, the same model can
also emit:

- future pose tokens: what the person is likely to do next;
- a corrected target trajectory: what a valid continuation should look like;
- motion completion across an occlusion;
- a counterfactual delta such as the joint-space adjustment associated with the
  selected cue;
- a temporally grounded answer pointing to the frames that caused the decision.

Superman demonstrates the first three capabilities at large scale. For therapy,
the model should train a predicted observed continuation and a reference/corrected
continuation as separate outputs so it does not confuse forecasting with advice.

## The best fusion design for 25M parameters

| Fusion | Pose representation | Strength | Failure mode at this scale |
|---|---|---|---|
| Direct projection / early fusion | One or a few continuous pose vectors inserted in the decoder stream | Simplest; native precision; continuous KV cache; fully analogous to LLaVA and VideoLLM-online | Pose tokens consume the 1,024-position budget |
| Discrete unified vocabulary | VQ/FSQ motion codes added to text/action vocabulary | Pure next-token training; motion generation/completion; demonstrated by MotionGPT, MG-MotionLLM, HSI-GPT, and Superman | Quantization can erase small clinical deviations; codebook training is consequential |
| Q-Former | Learned queries cross-attend to a pose window | Fixed token count; can adapt a mostly frozen decoder | Adds parameters and window latency; queries can smear exact boundaries |
| Perceiver resampler | A variable pose window becomes `K` latent tokens | Strong bounded compression for long clips | Recomputing it on every frame is wasteful unless made chunk-recurrent |
| Per-layer cross-attention | Decoder text/action states attend to a separate rolling pose memory | Pose memory does not consume text positions; good for retrofitting a frozen Nexus/Ember | Extra projections in many layers; more complex KV scheduling |
| Encoder-decoder | Causal/bidirectional pose encoder conditions an autoregressive decoder | Excellent for completed clips and batch sequence-to-sequence tasks | A standard bidirectional encoder is not naturally streaming; separate encoder memory adds RAM |

The recommended 25M research model is **hybrid early fusion**:

1. continuous pose embeddings are input tokens, retaining measurement detail;
2. a decoder-only causal transformer fuses pose, instruction, profile, and event
   history in one residual stream;
3. parallel heads provide control-rate state and actions;
4. the tied LM head emits compact event, JSON/language, and optionally discrete
   future/corrected-motion tokens.

That is more unified than a temporal classifier followed by an unrelated LLM. It
also avoids forcing millimeter/degree-level input through a lossy discrete
codebook. A discrete motion vocabulary can still be an **output** language and an
auxiliary training target.

## A concrete 25M model

Call it **Pose-Language-Action Transformer 25M (PLAT-25M)**.

### Pose stem

- 33 joints; coordinates, confidence, missingness, velocity, acceleration,
  bone-relative vector, and timestamp delta;
- width 128 shared joint embedding;
- two small anatomy-masked graph-attention blocks;
- learned full-body pooling query;
- short causal temporal convolution;
- linear projection from 128 to decoder width 384;
- approximately 0.45M parameters;
- one full-body token per 100 ms by default; optionally inject two part-detail
  tokens only when visibility or deviation gating requests them.

### Decoder

- decoder-only, 14 blocks;
- `d_model = 384`;
- six query heads, two grouped KV heads, head dimension 64;
- SwiGLU feed-forward width 1,024;
- RMSNorm and rotary time/position encoding;
- 1,024 total positions;
- 6,144-token tied input/output vocabulary, including language pieces, schema
  syntax, action/event codes, optional motion codes, and special modality tokens.

Approximate parameter accounting:

\[
N_{attn/layer}=384^2+2(384\cdot128)+384^2=393{,}216,
\]

\[
N_{ffn/layer}=3(384\cdot1024)=1{,}179{,}648,
\]

\[
14(N_{attn}+N_{ffn})=22{,}020{,}096,
\]

\[
N_{embedding}=6{,}144\cdot384=2{,}359{,}296.
\]

Adding the pose stem, norms, and online heads gives approximately 24.9M
parameters. INT8 weights occupy about 25 MB; INT4 weights about 12.5 MB, before
runtime metadata and activations.

Grouped-query attention matters for edge memory. With KV width 128, 14 layers,
1,024 positions, and FP16 cache:

\[
2\cdot14\cdot1024\cdot128\cdot2\text{ bytes}\approx7\text{ MiB}.
\]

An INT8 KV cache is about 3.5 MiB. A full six-head KV cache would be three times
larger.

### If 25M must include the RGB pose estimator

Then this is a different budget. A 25M decoder plus pose estimation is a roughly
30–45M total system depending on the estimator. For a strict 25M end-to-end graph,
allocate approximately 6–9M to a mobile RGB/heatmap backbone and 16–18M to a
narrower decoder. Train through soft-argmax or pose heatmap tokens with an
auxiliary keypoint loss:

\[
\mathcal L=\mathcal L_{action/language}
+\lambda_{pose}\mathcal L_{keypoint}
+\lambda_{motion}\mathcal L_{future}
+\lambda_{cal}\mathcal L_{calibration}.
\]

That makes the whole camera-to-output system differentiable. Keeping the pose
provider frozen is easier and enables reuse of real-device pose failures, but it is
a fused multimodal system rather than literally one end-to-end neural graph.

## What 1,024 context positions actually buy

One token per pose timestamp makes the budget surprisingly usable:

| Pose rate | Tokens/frame | Raw time in 1,024 positions | Time with 256 positions reserved |
|---:|---:|---:|---:|
| 5 Hz | 1 | 204.8 s | 153.6 s |
| 10 Hz | 1 | 102.4 s | 76.8 s |
| 15 Hz | 1 | 68.3 s | 51.2 s |
| 10 Hz | 4 | 25.6 s | 19.2 s |

This is why every joint must not become a separate decoder position. The skeleton
stem should perform spatial reasoning over all joints and contribute one full-body
token per timestamp. Part tokens are an adaptive exception.

`1,024` is not inherently the best context length. The right window is the longest
causal dependency the product must resolve under its memory budget. For most reps,
10–30 seconds suffices; for a complete set, 50–100 seconds may be useful.

For longer sessions, keep raw pose at high resolution only in a rolling window and
carry learned recurrent memory, not a prose summary:

\[
m_b=R_\phi(m_{b-1},e_{bB:(b+1)B}),
\]

where `B` might be 32 or 64 pose tokens. Retain instruction/profile tokens, recent
raw pose, boundary/error event tokens, and a small bank of learned set-memory
vectors; evict old raw pose keys/values. Train with this exact eviction pattern so
the model does not rely on information unavailable at deployment. This memory can
preserve rep count, prior error trend, set identity, and personalization without
turning the motion into text.

## Online inference scheduling

At each accepted pose timestamp:

1. compute one pose embedding;
2. append one position to the continuous KV cache;
3. read the fast state/action heads from that position;
4. evaluate `P(RESPOND | h_t)`;
5. if no response is needed, consume the next pose;
6. if a response is needed, decode a compact event/action sequence or a
   schema-constrained response.

This is the pose equivalent of VideoLLM-online's streaming loss:

\[
\mathcal L_{stream}=-\sum_t
\left[y_t\log P(RESPOND\mid h_t)+(1-y_t)\log P(SILENT\mid h_t)\right].
\]

Do not append thousands of `SILENT` output tokens to the context. Supervise the
probability on each input pose state, then append output tokens only when the model
responds.

Language generation and incoming pose compete for decoder steps. If the hardware
ingests 10 pose tokens/s and generates 10 output tokens/s, it needs roughly 20
decoder steps/s to do both without falling behind. Three implementation choices
exist:

- interleave high-priority pose positions with low-priority language positions;
- buffer pose embeddings during a short language burst and catch up in a batched
  prefill, as VideoLLM-online does with video frames;
- never put safety-critical timing behind language generation: use the fast action
  head continuously and let language be asynchronous.

The third is recommended. It does not split reasoning into separate models; it
uses two output clocks from one fused state.

## Training the unified model

Training examples must be **untrimmed causal streams**, not bags of completed
reps. A useful serialized training sample is:

```text
<SYSTEM> task and allowed actions
<PROFILE> exercise, side, prescribed range, prior constraints
<LIVE>
pose_1 pose_2 ... pose_37
<REP_START>
pose_38 ... pose_91
<REP_END> <VALID> <NO_CUE>
pose_92 ...
<REP_END> <PARTIAL> <HIP_COMPENSATION> <CUE_STABILIZE>
```

Use one shared objective over the decoder state:

\[
\begin{aligned}
\mathcal L={}&\lambda_{out}\mathcal L_{next\ output\ token}
+\lambda_{stream}\mathcal L_{respond/silent}
+\lambda_{phase}\mathcal L_{phase}
+\lambda_{boundary}\mathcal L_{boundary}\\
&+\lambda_{dev}\mathcal L_{deviation}
+\lambda_{future}\mathcal L_{future\ pose}
+\lambda_{mask}\mathcal L_{masked\ joint/time}
+\lambda_{cal}\mathcal L_{calibration}.
\end{aligned}
\]

A practical curriculum is:

1. **Motion pretraining:** real mocap, real pose tracks, and simulation; masked
   joint/time reconstruction, next-motion prediction, phase consistency, temporal
   order, and view augmentation.
2. **Motion-language alignment:** captions, exercise instructions, body-part and
   temporal QA, and paired correct/incorrect trajectories. MotionGPT, LLaMo, and
   MG-MotionLLM provide precedents. MG-MotionLLM's published 60M/220M/770M
   comparison also found that scaling did not improve consistently on HumanML3D,
   pointing to dataset scale—not only parameter count—as the limiting factor
   ([CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_MG-MotionLLM_A_Unified_Framework_for_Motion_Comprehension_and_Generation_across_CVPR_2025_paper.html)).
3. **Streaming instruction/action tuning:** exact timestamps for start, endpoint,
   completion, partial, abort, pause, resume, set boundary, tracking failure,
   deviation, cue choice, and abstention; include long no-action intervals.
4. **Teacher distillation:** train a larger offline motion-language teacher, then
   distill token distributions, fast-head outputs, selected hidden states, and
   uncertainty into PLAT-25M.
5. **Quantization-aware fine-tuning:** use the final context policy, KV precision,
   and missing-pose patterns during training.

Simulation is useful because it supplies exact joints, contacts, phase, intentional
errors, and counterfactual corrected trajectories. Domain randomization should
project that motion through real camera/view/occlusion distributions and, crucially,
through a learned pose-estimator error model. Real untrimmed human sessions remain
necessary for pauses, hesitation, compensations, restarts, fatigue, and device
failure patterns.

## Why this system is not already common in rehabilitation

There are precedents for the pieces, but almost no precedent for the complete
intersection.

| Existing line of work | What it already proves | What it does not yet prove |
|---|---|---|
| MotionGPT, MG-MotionLLM, HSI-GPT | Motion can become a decoder vocabulary; a language model can read and generate skeleton/motion tokens | True causal streaming, clinical deviations, compact edge runtime |
| MotionLLM and LLaMo | Native motion features can be projected/fused into an LLM and support detailed temporal reasoning | 5–15 Hz online operation, 25M scale, patient validation |
| Superman | A single decoder-only MLLM can unify video-to-skeleton perception, skeleton continuation, and in-betweening | It uses Qwen2.5-VL-7B, 16-frame tasks, and general pose benchmarks—not therapy or edge deployment |
| VideoLLM-online, StreamMind, later streaming VLM work | Continuous multimodal tokens, KV-cache state, learned silence/response timing, and online generation are possible | Skeleton-native inputs, precise kinematics, rehabilitation decisions |
| RT-2, OpenVLA, Octo, pi0/FAST | A multimodal decoder can emit discrete or continuous actions instead of summaries | Human coaching actions, clinical safety, small-device execution |
| Rehabilitation pose systems | Rep counting, phase/error classification, and movement-quality scoring from pose are useful | Unified motion-language-action decoder, broad instruction following, native generative outputs |

The missing combination is:

```text
phone/edge pose + native decoder tokens + causal KV streaming +
online state/action/JSON + untrimmed patient data + safety calibration +
sustained device measurements
```

Several forces explain the gap.

1. **The research communities optimize different targets.** Motion-language work
   is driven by avatar generation, mocap completion, captioning, and general QA.
   Rehabilitation work is driven by angle error, rep count, clinical scores, and
   small-cohort feasibility. Streaming-VLM work is driven by narration and event
   dialogue. VLA work is driven by robot success rate. The common architecture is
   visible only after combining literatures that rarely share datasets or metrics.

2. **The required data is much harder than motion captions.** HumanML3D-scale data
   says that somebody walked, sat, or raised an arm. It usually does not contain
   the exact frame where a therapeutic rep became invalid, which compensation was
   visible, whether a pause was intentional, which cue a clinician selected, and
   whether that cue improved the next rep. Patients, braces, pain-limited ranges,
   assistive devices, unusual tempos, and partial movements are also scarce in
   entertainment-oriented mocap. The language decoder cannot learn a reliable cue
   policy from captions that never encode cue decisions.

3. **Clinical labels are expensive and partially unobservable.** Fine error and
   cue labels require clinicians, timestamped adjudication, and often a reference
   sensor. Two competent therapists can disagree. A monocular camera cannot observe
   every plane or distinguish all anatomical causes. A serious dataset must preserve
   disagreement and `unobservable`; conventional classification datasets tend to
   force a single label.

4. **Pose tokenization has an uncomfortable precision tradeoff.** Discrete VQ
   tokens fit language modeling beautifully but may merge trajectories separated
   by a clinically important few degrees. Native continuous tokens preserve detail
   but require a custom pose stem, mixed-modality batching, and serving support.
   Large-model papers can absorb a Q-Former, cross-attention stack, or huge codebook;
   a 25M edge model must prove every projection earns its memory.

5. **A streaming decoder pays for input tokens even while silent.** A clip model
   runs once. A live model must ingest 5–15 tokens every second for an entire
   session, keep or compress its KV cache, and generate outputs without falling
   behind incoming frames. The pose estimator already consumes camera, memory,
   and accelerator budget. Sustained thermal and battery behavior—not one forward
   pass—is the actual product constraint.

6. **The deployment stack is fragmented.** Mobile runtimes are good at a static
   vision graph or a conventional text decoder. A runtime that accepts externally
   produced continuous pose embeddings between text tokens, incrementally updates
   a KV cache, switches between fast heads and grammar-constrained generation, and
   evicts learned memory needs more custom engineering. On ESP32-class hardware,
   external-memory bandwidth and the combined pose-plus-decoder rate make the gap
   larger.

7. **There is no accepted end-to-end benchmark.** Existing benchmarks separately
   measure offline action accuracy, motion FID/MPJPE, caption quality, rep-count
   error, or robot task success. This system needs event latency, boundary F1,
   false-cue rate per minute, patient-held-out calibration, safety-event recall,
   recovery after occlusion, joules per session, thermal throttling, and structured
   output validity. Without such a benchmark, a unified system can look worse than
   a specialist on each specialist's home metric even if it is more useful overall.

8. **Modularity is organizationally safer.** A pose model plus rules is easier to
   debug, certify, and assign to teams. End-to-end multitask gradients can improve
   semantic behavior while degrading keypoint geometry or calibration. The unified
   model therefore needs auxiliary pose losses, frozen-stage training, ablations,
   and explicit uncertainty—not merely an end-to-end slogan.

9. **Medical incentives punish uncontrolled novelty.** A false robot action can
   fail a benchmark trial; a false therapeutic cue can harm or erode trust. Free
   language raises hallucination, auditability, human-factors, privacy, consent,
   validation, and potentially medical-device regulatory burdens. Teams reasonably
   constrain output vocabularies and keep a policy shield, which can make the system
   appear less like a flashy VLM even when its internal representation is unified.

10. **Large generalist models attract clearer research rewards.** Scaling a known
    benchmark from 7B to a larger model is easier to publish and compare than
    collecting a longitudinal patient dataset and demonstrating a 25M model under
    a phone thermal envelope. MG-MotionLLM's 60M result is an important compact
    precedent, but it is still an offline motion-language model, not the system
    described here.

This gap is evidence of an unsolved systems-and-data problem, not evidence that
the architecture is impossible.

## Where reinforcement learning belongs

RL is not required to make continuous pose understandable. Causal supervised and
self-supervised sequence objectives should learn pose perception, phase, boundaries,
deviations, forecasting, uncertainty, and valid event tokens. If an exact correct
label exists for a frame or rep, RL is an unnecessarily indirect way to learn it.

RL becomes relevant one level later: **choosing an intervention whose value is
measured by what happens after the intervention**.

Let the unified decoder hidden state be the policy state:

\[
s_t=h_t=f_\theta(P_{\le t},I,C,A_{<t}).
\]

The action head chooses from a clinician-approved set:

\[
\pi_\theta(a_t\mid h_t),\quad
a_t\in\{\text{wait},\text{count},\text{cue}_1,\ldots,
\text{reposition},\text{pause},\text{escalate}\}.
\]

Reward arrives after one or more later reps:

\[
J(\pi)=\mathbb E_\pi\left[\sum_{k\ge0}\gamma^k(
r^{form}_{t+k}+\alpha r^{completion}_{t+k}
-\beta r^{nuisance}_{t+k}-\kappa r^{risk}_{t+k})\right].
\]

This can learn questions that supervised frame labels cannot answer directly:

- Is a cue needed now, or will it interrupt a self-correction?
- Which of several correct cues works best for this user?
- Should the system repeat, simplify, or suppress a cue after nonresponse?
- Does cueing every detectable deviation reduce adherence or overload the user?

The first realistic version is a contextual bandit over cue choice/timing, because
the horizon and attribution problem are smaller. More ambitious personalization is
offline constrained RL from logged trajectories. [Decision Transformer](https://papers.nips.cc/paper_files/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html)
shows that return-conditioned action selection can itself be expressed by a causal
sequence model, so RL does not require abandoning the unified decoder. A return or
goal token can condition the same pose-language-action stream.

However, ordinary offline RL can overvalue actions absent from the logged data.
[Conservative Q-Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
was designed to lower-bound policy value under this distribution-shift problem.
For rehabilitation, use behavior cloning first, then conservative/offline policy
improvement with an allowed-action mask, uncertainty rejection, and a runtime
safety shield. Freeze or strongly regularize the pose/perception trunk while
optimizing the action adapter so a noisy reward does not destroy calibrated
movement understanding.

Do **not** use unconstrained online RL to explore cues on patients. Simulation can
pretrain cue timing and generate adversarial trajectories, but a simulated human's
response is not a validated reward model. Real adaptation requires consent,
randomized or otherwise causally interpretable feedback, clinician oversight, and
predefined stopping rules.

RLHF/DPO-style preference tuning has a narrower role: making language concise,
empathetic, and schema-compliant, or ranking two otherwise safe cues. It cannot
repair missing pose supervision or establish that a biomechanical judgment is
correct. The core division is:

```text
supervised/self-supervised learning -> perceive and understand motion
imitation learning                 -> copy approved cue behavior
offline/constrained RL             -> optimize cue timing/personalization over time
hard policy shield                 -> retain authority over unsafe outputs
```

## What should be claimed—and what should not

The strong claim is not "a decoder writes a better summary." It is:

> A compact causal decoder treats human pose as a native streaming modality and
> jointly learns online state estimation, temporal event grounding, motion
> forecasting, structured action selection, and language generation from one
> shared multimodal representation.

As of August 2026, the research record supports the architecture's feasibility but
does not establish that a 25M model is clinically reliable. Existing motion-language
models are mostly 60M to multi-billion parameters, usually process completed clips,
and are trained on general motion rather than untrimmed patient exercise. The
paper-worthy experiment is precisely whether domain-specific pose pretraining,
streaming supervision, and action-token distillation let 25M retain the useful part
of that unified behavior on a phone or constrained edge target.
