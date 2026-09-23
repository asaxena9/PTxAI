# Streaming skeleton and body-decoder systems audit

Date: 2026-08-07  
Status: sourced prior-art and code audit; not a proof of novelty  
Scope: primary papers and author/official implementations located under online
action detection, early action prediction, continual inference, causal skeleton
recognition, repetition counting, streaming video-language modeling, and
infinite-video memory. Sources were accessed on 2026-08-07.

Authoritative project target: [Canonical build thesis: a native continual
body-motion decoder](./NATIVE_BODY_DECODER_CANONICAL_BUILD_2026-08-07.md). This
audit supplies prior-art evidence and controls; it does not redefine that target.

## Result in one paragraph

The literature contains nearly every ingredient separately. Direct skeleton
streams have been classified and temporally localized online since at least
2015. Compact recurrent, dilated-convolutional, graph-convolutional, Transformer,
and state-space body models range from roughly 0.29M to 16M parameters. A direct
pose-to-autoregressive-decoder model also predates this project in UPS. Large
streaming VLMs learn when to remain silent or emit language, retain incremental
KV state, and, in StreamingVLM, evict old visual context while preserving a
bounded cache. Within the searched corpus, however, no system at or below 400M
parameters combined direct pose observations, persistent incremental inference
over an untrimmed stream, query-conditioned semantic/structured autoregressive
outputs, learned wait/commit behavior, explicit survival of facts after raw
context eviction, and measured sustained edge deployment. This is an
**UNKNOWN/open conjunction in the searched literature**, not evidence that no
such implementation exists anywhere.

That full conjunction must not become the gate for the first build. The canonical
first build is a serious 270--350M compact continual physical-analysis decoder:
direct timestamped pose, confidence, and missingness tokens update persistent
causal state; task/protocol/query tokens condition that state; and the decoder
emits structured physical analysis with body/time evidence, comparisons,
protocol judgments, and bounded predictions/counterfactuals. Rep/event
registration and an exactly-once ledger are supporting state/audit machinery,
not the scientific reason to use a decoder. Efficient cache equivalence,
long-context rollover, edge deployment, multi-person tracking, coaching, and
broader domains are subsequent independently falsifiable claims, not
prerequisites for the first capability test.

## Terms that papers frequently conflate

Let an accepted pose observation be $X_t$, a query or task context be $Q_t$,
and retained state be $S_t$.

1. **Causal output** means
   $p(Y_t\mid X_{\leq t},Q_{\leq t})$: a prediction at $t$ cannot depend on
   a future suffix. A lower-triangular attention mask can establish this
   dependency property even if the implementation recomputes the entire prefix.
2. **Incremental computation** means an implementation exposes a recurrence
   $S_t=F_\theta(S_{t-1},X_t)$ and computes the next output without
   re-encoding $X_{<t}$. RNN state, an SSM state, cached convolution
   activations, and KV cache all qualify.
3. **Untrimmed operation** means background, multiple actions, pauses, and
   boundaries occur in the same stream. Predicting every prefix of a trimmed
   action clip is not this setting.
4. **Autoregressive semantic decoding** means, for a response sequence $W$,
   
   \[
   p_\theta(W\mid X_{\leq t},Q)=
   \prod_{k=1}^{|W|}p_\theta(w_k\mid w_{<k},X_{\leq t},Q),
   \]
   
   rather than a fixed class, boundary, count, or scalar-quality head.
5. **Proactive control** adds an action such as wait/silence/commit:
   $A_t\in\{\text{wait},\text{commit},\text{abstain}\}$.
6. **Rollover memory** requires a bounded raw context and a test that information
   remains available after the decisive observations have been evicted. An
   ever-growing LSTM state or KV cache is persistent state, not by itself an
   evaluated rollover policy.
7. **Continual learning** often means learning new action classes over time.
   That is a different problem from continual inference on a live stream.

The audit applies these definitions to paper claims and released code rather
than accepting “online,” “continual,” “streaming,” or “decoder” in a title.

## Compact direct-pose systems

| System | Size, input, and temporal state | Output and evidence | What the released implementation actually does | Exact nonmatch |
|---|---|---|---|---|
| [Bags of Gesturelets](https://arxiv.org/abs/1502.01228) (2015) | Non-neural descriptor, linear SVMs, and a per-class dynamic program over 3D skeleton. The descriptor uses a temporal window centred on the current frame with two frames on either side, so wall-clock causality includes two frames of explicit look-ahead. | Detects class plus start/end in unsegmented MSRC-12 streams. The paper reports 10.7 ms/frame, about 93 FPS; MSRC-12 was captured at 30 FPS. | The dynamic program retains the current maximum-scoring subarray and candidate boundary rather than rescanning all subintervals. | Fixed class detectors; no learned body state, query, semantic decoder, or memory rollover. It is an important boundary/event baseline and a warning to account for tolerated look-ahead explicitly. |
| [JCR-RNN](https://arxiv.org/abs/1604.05633) (ECCV 2016) | Direct $25\times3=75$-value skeleton frames; three stacked LSTMs provide recurrent hidden/cell state. SSNet's comparison reports about 290K parameters. | Per-frame action/blank class plus start/end confidence and limited start forecasting. OAD contains 59 long videos, 103,347 frames, 216 minutes, sampled at 8 FPS. The paper reports 1,230 skeleton frames/s, but does not identify the measurement hardware in the running-speed section. | The recurrence is naturally incremental and avoids a sliding window. No official implementation was located in this audit. | Genuine untrimmed causal pose state, but only a fixed ontology and regression head; no autoregressive response, arbitrary query, wait policy, or state-rollover evaluation. |
| [SSNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_SSNet_Scale_Selection_CVPR_2018_paper.html) (CVPR 2018; expanded TPAMI 2020) | Fourteen causal dilated-convolution layers expose receptive scales through 255 frames. It regresses distance to the current action start, uses that distance to select a scale, and shares cached activations across adjacent steps. The multi-level version is 310K parameters and 40 FPS; coordinate concatenation is 270K and 50 FPS. | Predicts the ongoing action at each step of untrimmed OAD, ChaLearn Gesture, PKU-MMD, and G3D streams. | The paper specifies incremental activation sharing. No author implementation was located, so cache equivalence could not be code-tested. | Strongest early compact untrimmed pose event/prefix control; fixed class and scale outputs, not a semantic decoder or persistent semantic memory. |
| [DDCN + SRM](https://doi.org/10.1016/j.patcog.2023.109595) ([official code](https://github.com/Powercoder64/DDCN_SRM), Pattern Recognition 2023) | Ten dynamic dilated-convolution layers, 32 channels, nominal receptive field 1,024; direct skeleton plus an additional RGB-derived object-semantic module. Parameter count was not reported. | Early action detection on untrimmed OAD and PKU-MMD; fixed action probabilities at observation ratios. | **MEASURED-CODE-AUDIT**, commit `547f415445d8694e2c2ab637d34930462aa77637`: queue classes and `queue_dilate` exist, but the released `forward` calls the full-sequence `DDCN_dilate` path. The test path loads whole arrays and invokes one full `model(x)`. | A causal formulation is not a demonstrated incremental deployment path. It also adds object semantics and has no generative decoder or rollover memory. |
| [Continual ST-GCN / CoST-GCN](https://arxiv.org/abs/2203.11009) ([official code](https://github.com/LukasHedegaard/continual-skeletons), Pattern Recognition 2023) | CoST-GCN 3.14M, CoAGCN 3.47M, CoS-TR 3.09M. Continual convolution/pooling caches intermediate features. | On NTU60, stride-1 CoST-GCN reports 0.16G FLOPs/prediction, 36.1 MB peak GPU memory, 46.1 CPU and 4,202 GPU predictions/s, with 86.3/93.8 X-Sub/X-View accuracy. Hardware: MacBook Pro 2.6-GHz six-core i7 and RTX 2080 Ti. | **MEASURED-CODE-AUDIT**, commit `a8fe2937a33f24cce65c1f8c2fc41081bceda721`: models expose `forward_step`, `forward_steps`, and `clean_state`; tests compare batch and stepping results down to `atol=5e-7` in key paths. | Best code-verified compact incremental skeleton backbone found, but evidence is trimmed clip classification on NTU/Kinetics Skeleton. It does not solve background/boundaries, query-conditioned composition, decoder control, or rollover. |
| [InfoGCN++](https://arxiv.org/abs/2310.10547) ([official code](https://github.com/stnoah1/infogcn2), 2023) | 2.09M-parameter SA-GC causal Transformer plus NeuralODE latent future prediction. Inputs are truncated to 64 frames for NTU60/120 and 52 for NW-UCLA. | Early action class prediction across observation ratios and three-step latent motion forecasting. The paper reports 18.0 ms/56 FPS on NTU and 15.6 ms/64 FPS on NW-UCLA on an unspecified RTX GPU. | **MEASURED-CODE-AUDIT**, commit `8c44085ac134f35dc85f683dcc775fdb43172ad1`: lower-triangular attention is present, but attention constructs full $T\times T$ scores. No KV cache or `forward_step` path was found; evaluation executes the full fixed tensor. The paper notes quadratic memory growth. | Causal outputs but not incremental computation. Trimmed action prefixes, task heads, and no semantic decoder or session memory. |
| [HumMUSS](https://openaccess.thecvf.com/content/CVPR2024/papers/Mondal_HumMUSS_Human_Motion_Understanding_using_State_Space_Models_CVPR_2024_paper.pdf) (CVPR 2024) | 16M-parameter gated diagonal state-space model. In causal mode, the current 2D pose and retained SSM state give $O(1)$ time and memory per new frame; continuous-time discretization supports new/variable frame rates. | Pretraining by 2D-to-3D reconstruction and task-specific finetuning for 3D pose, mesh, and skeleton action recognition. For sequential 243-frame inference at batch 32 on an 80-GB A100, the paper reports 3.8x memory efficiency and 11.1x speed versus a same-size causal MotionBERT. | No official code was linked from the paper/project page at access time, so the recurrence was not independently exercised. | Strong compact recurrent body-representation prior, but no untrimmed event task, semantic query decoder, proactive output, rollover, or edge-device measurement. |
| [Rehabilitation repetition segmentation](https://arxiv.org/abs/2304.09735) ([official code](https://github.com/abedidev/repetition-segmentation), 2023) | Direct skeleton into a unidirectional LSTM and per-time density/binary output. Parameter count not reported. | Segmentation and counting on KIMORE, UI-PRMD, and IntelliRehabDS. The best general KIMORE binary-sequence row reports count MAE 0.5313; the paper also evaluates IoU and boundary error. | **MEASURED-CODE-AUDIT**: the released LSTM model processes each complete sequence and reinitializes hidden state per sequence; no stepwise deployment API was found. The architecture is causal-compatible but the evaluation is not a live multi-activity stream. | Bounded one-exercise sessions, fixed density/binary outputs, and no query semantics or learned commit/memory policy. |
| [PoseRAC](https://arxiv.org/abs/2303.08450) ([official code](https://github.com/MiracleDance/PoseRAC), 2023) | Per-frame MediaPipe pose with 99 values. The released checkpoint contains 2,686,682 tensor parameters and 10,746,728 tensor bytes (**MEASURED-CODE-AUDIT**). | Classifies two action-salient poses and counts their alternation. The official report gives MAE 0.236, OBO 0.560, and 20 ms per frame on RepCount-pose. | **MEASURED-CODE-AUDIT**, commit `469590b611bde3595eaf163b517263da634e2096`: input is reshaped to sequence length one before a six-layer `TransformerEncoder`; therefore self-attention is not temporal. Persistent behavior comes from a hand-written hysteresis `Action_trigger` and counter. | A useful tiny live rep-count control, but no learned temporal body state, event memory, or semantic decoder. Its “Transformer” label must not be mistaken for temporal attention. |
| [Pūioio](https://arxiv.org/abs/2308.02420) (2023) | On-device BlazePose plus joint-angle thresholds, optical-flow fallback, and exercise-specific finite-state machines for squats, push-ups, and pull-ups. | Ninety curated YouTube clips and a two-person, 450-repetition, 17-day real-life evaluation; the paper reports 98.89% real-life absolute count accuracy and says no discernible lag. | The mobile implementation is live, but no phone model, FPS, p50/p95 latency, memory, energy, or thermal trace is reported. | Complete on-device pipeline evidence is limited, and learned semantics stop at the pose estimator. The temporal logic is handcrafted and exercise-specific. |
| [Efficient skeleton action recognition for embedded systems](https://openaccess.thecvf.com/content/CVPR2024W/MAI/papers/Noor_Efficient_Skeleton-Based_Action_Recognition_for_Real-Time_Embedded_Systems_CVPRW_2024_paper.pdf) (CVPRW 2024) | YOLOv8s-pose plus a 0.1M-parameter modified ResNet classifier over pseudo joint heatmaps from 32 or 64 uniformly sampled frames. | Five-class recognition. On a Rockchip RV1126 NPU, the complete quantized system reports 84.9 ms pose time, 66.4 ms classification time, 10.70 FPS, and 16.91 MB total action-model memory. | Fixed-window classification; no retained model state between windows. | Strongest full embedded pose+recognition measurement in this table, but not a continual decoder and not proof that the 0.1M classifier learns temporal composition. |
| [Foundation model for skeleton understanding / USDRL](https://arxiv.org/abs/2508.12586) ([official code](https://github.com/wengwanjiang/FoundSkelModel), 2025/TPAMI 2026) | Large two-layer DSTE with 1,024-dimensional representations. The paper does not give a defensible complete parameter count. | Dense transfer across 25 benchmarks and nine skeleton tasks. Action prediction uses causal masking; action detection/segmentation use fixed windows or whole long sequences and fixed heads. | **MEASURED-CODE-AUDIT**, commit `0b4d60c3430a6e777eda85790c29c1bc0cbfbe98`: no persistent cache or `forward_step` path was found; long detection uses sampled/sliding windows and postprocessing. | Broad pose representation pretraining, not an incrementally executed or autoregressive semantic stream model. |

### Terminology false positives

- [Else-Net](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Else-Net_Elastic_Semantic_Network_for_Continual_Action_Recognition_From_Skeleton_ICCV_2021_paper.pdf) uses *continual* for class-incremental learning when new action categories arrive, not retained inference state on each new frame.
- [Online Skeleton-based Action Recognition with Multi-feature Early Fusion](https://github.com/fabro66/Online-Skeleton-based-Action-Recognition) presents a camera demo, but **MEASURED-CODE-AUDIT**, commit `de00cbf17ceea98a7d07f68bbbd966bfd02d3b40`, shows complete padded sequences passed to convolution and adaptive pooling, with no causal cache or step API.
- [LaSA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07145.pdf) performs language-assisted action segmentation on long untrimmed skeleton sequences, but the language representation is supervision/prior information and inference is offline full-sequence frame classification.
- “Continuous” sign-language recognition commonly means an unsegmented signed sentence supplied as a complete sequence. It is not automatically continual inference.

## Pose plus autoregressive decoding already exists

[Unified Pose Sequence Modeling](https://openaccess.thecvf.com/content/CVPR2023/html/Foo_Unified_Pose_Sequence_Modeling_CVPR_2023_paper.html)
(UPS, CVPR 2023) is the decisive architecture prior. It sends direct coordinate
pose through a TCN-GCN tokenizer and a dynamically routed Transformer encoder,
then uses a three-layer autoregressive Transformer decoder over a unified
vocabulary of action tokens, quantized (x/y/z) pose coordinates, and an
end-of-task token. It trains the usual negative log likelihood

\[
\mathcal L_{\mathrm{UPS}}=-\frac{1}{K}\sum_{k=1}^K
\log p_\theta(y_k\mid y_{<k},X,\tau_{\mathrm{task}}).
\]

It handles 2D/3D action recognition, 3D pose estimation, and early action
prediction in one output interface. Therefore **“first body pose plus a
Transformer/autoregressive decoder” is not defensible novelty**.

UPS does not close the current stage-1 question. Its early-action evaluation is
on trimmed NTU prefixes at fixed observation ratios, its encoder sees the whole
supplied prefix, no incremental cache/state is reported, its outputs are a small
task vocabulary rather than arbitrary body/temporal compositional queries, its
total parameter count is unreported, and no edge result or official
implementation was located.

Stage 1 should use UPS as a positive architectural control and ask a stronger
empirical question: does a compact direct-pose decoder retain and conditionally
compose temporal/body facts that deliberately matched non-sequential or
handcrafted controls cannot recover? That claim can be tested before continual
inference exists.

## Large streaming decoder capability anchors

| System | Decoder and stream mechanism | Context/data/measurement | What transfers; exact gap |
|---|---|---|---|
| [Stream-VLM](https://arxiv.org/html/2407.08101) ([official code](https://github.com/Qualcomm-AI-research/FitCoach), NeurIPS 2024) | Causal 3D CNN fused by cross-attention into LLaMA-2-7B. `<next>` means consume more visual evidence and remain silent; `<feedback>` commits to autoregressive feedback. | Qevd has 474+ video hours. Qevd-Fit-300K has 281,660 train clips, 1,800+ train participants, 535,299 high-level and 377,678 fine-grained train questions. Qevd-Fit-Coach has 149/74 train/test videos, 23 exercises, and 5,403/2,484 timestamped feedbacks. Training is limited to individual 30-second exercise segments. The reported T-F-score is 0.56. | **MEASURED-CODE-AUDIT**, commit `a573ee0b98938b81c1b3e19f94c5fab5d1856f26`, confirms persistent `past_key_values`, explicit `feedback_mode`, and handling of frames received while text is decoded. This is the closest “when and what to say” control. It is RGB, roughly 7B plus vision, needs at least 48 GB VRAM per the repository, has no cache eviction/semantic rollover, and is coaching-specific rather than direct-pose compact intelligence. |
| [VideoLLM-online](https://arxiv.org/html/2406.11816) ([official code](https://github.com/showlab/videollm-online), CVPR 2024) | CLIP-ViT-L (307M), two-layer projector, and LLaMA-2-7B. One token per frame at 2 FPS in the efficient model. Streaming-EOS loss teaches silence; EOS is not appended to context. Continuous KV cache and a FIFO buffer let vision encoding proceed while language is generated. | A one-token/frame model can nominally hold about half an hour in 4,096 positions. On a five-minute Ego4D stream it reports 18.2 GB and 13.5 FPS on A100, with over 10 FPS in the abstract. LoRA rank 128. It synthesizes online queries about past/current/future events from timestamped COIN/Ego4D annotations. | The most reusable objective/data/inference pattern for query-conditioned live decoding. **MEASURED-CODE-AUDIT**, commit `3da632217258b830a9a594bdabd7a40ae99646dc`, confirms KV trimming/caching and greedy generation. It is RGB and 7B, A100-measured, and has no bounded long-session eviction policy in the reported configuration. |
| [BioCoach](https://arxiv.org/html/2603.26938) (CVPR 2026) | A 3D CNN and LLaMA-2-7B receive HSMR+SKEL 46-D Euler kinematics, morphometrics, and text serialized from cycle detection, curated reference alignment, and exercise-specific constraint violations. Only cross-attention and a three-layer DoF selector are tuned. | Twelve samples at 4 Hz cover a fixed three-second history; cycle durations are constrained to 0.8-5 s. QEVD-bio-fit-coach has 149/74 videos and 23 exercises. On its augmented feedback benchmark, T-F is 0.544 and LLM-Bio-Accuracy 3.26 versus 1.72 for its Stream-VLM baseline. No edge/latency result is reported. | Evidence that explicit skeletal/biomechanical channels can improve specificity. It is not a native learned pose decoder: RGB remains primary, the strongest body signal is manually/algorithmically converted into text and exercise-specific rules, the output timestamps are inherited from Qevd annotations, and the 7B decoder sees only a short fixed motion window. |
| [StreamingVLM](https://arxiv.org/html/2510.09608) ([official code](https://github.com/mit-han-lab/streaming-vlm), ICLR 2026) | Qwen2.5-VL-7B with a compact reusable KV cache: attention-sink text, recent text, and recent visual tokens. Old vision is evicted first; contiguous 3D RoPE shifts retained and incoming positions back into a bounded range. | Main setting retains 512 sink text tokens, 512 recent text tokens, and 16 seconds of vision. Training uses 525K Inf-Streams samples, 526K Live-WhisperX samples, and 14K annealing samples; reported cost is about 128 H100-days. Inf-Streams-Eval averages 2.12 hours; the model reports stable commentary over three hours and up to 8 FPS on one H100. | Best located rollover mechanism. **MEASURED-CODE-AUDIT**, commit `a940d8cbfa14eefe651b7f4adeb81825c28cc6c6`, confirms the 16-second visual window, 512/512 text policy, physical KV deletion, and contiguous cache tensors. However, “long-term memory” is largely retained/generated text plus sink state; discarded visual evidence is not registered as a verified structured body-event ledger. It is RGB, 7B, H100-measured, and produces commentary rather than pose-grounded arbitrary-time body queries. |

These large models are capability anchors and possible teachers. Their existence
does not show that a compact direct-pose decoder can learn the same decision
structure, and their GPU throughput does not establish phone feasibility.

## Smallest systems approaching each required property

| Property | Smallest or closest located system | What is established |
|---|---|---|
| Direct pose plus recurrent state on untrimmed input | JCR-RNN, about 0.29M | Causal framewise classification, background, start/end regression, and forecasting are feasible at very small scale. |
| Direct pose plus cached convolution over untrimmed input | SSNet, 0.27-0.31M | Dynamic receptive scale and cached activations can support online prefix prediction. |
| Code-verified incremental pose network | CoST-GCN, 3.14M | Batch and `forward_step` executions can be made numerically equivalent while reducing redundant work; the dataset is still trimmed. |
| Compact recurrent general body representation | HumMUSS, 16M | An SSM can carry pose state with constant sequential cost and adapt its discretization to frame rate. |
| Direct pose plus autoregressive output vocabulary | UPS, size unreported | A unified pose/action decoder is feasible offline; no continual state or free-form semantics are established. |
| Learned semantic silence/commit | Stream-VLM / VideoLLM-online, 7B language backbones | Action tokens or streaming-EOS supervision can jointly learn whether and what to decode. No compact pose-native example was located. |
| Bounded raw-context rollover | StreamingVLM, 7B | KV reuse, eviction, and position reindexing can remain stable for multi-hour RGB streams. It does not prove faithful body-event memory after eviction. |
| Full on-device pose pipeline | Pūioio (heuristic) and the 0.1M Rockchip classifier | Phone/NPU pose analysis is feasible; neither contains a learned semantic decoder. |

## Research consequences, staged rather than serially gated

### Stage 1: native continual physical analysis, now

The canonical first build is direct timestamped body observations -> persistent
causal decoder state -> task/protocol/query-conditioned physical analysis with
body/time evidence. A minimal event ledger supports temporal state, but logging
is not the endpoint. The immediate hypothesis is

> **H1:** a serious 270--350M direct-pose causal decoder can jointly maintain
> movement state and generalize on unseen people to held-out compositions of
> body, temporal, and task/protocol operations, with conclusions grounded in the
> causally relevant joints and time spans.

Let each observation packet be

\[
O_t=(P_t,C_t,M_t,\Delta t,R_t),
\]

where pose geometry, landmark confidence, missingness, physical elapsed time,
and root/camera channels remain distinct. With persistent state $S_t$, task or
query $Q$, auxiliary registered event state $E$, and structured analysis $Y$,
the first estimand is

\[
p_\theta(Y_{1:T},E_{1:T}\mid O_{1:T},Q),
\qquad
S_t=D_\theta(S_{t-1},O_t,Q,E_{<t}).
\]

The analysis decoder is trained with

\[
\mathcal L_{\mathrm{analysis}}
=-\sum_i\log p_\theta(y_i\mid y_{<i},O_{\leq t},Q,S_t,E_{<t}).
\]

This objective rewards the complete requested answer conditional on earlier
answer fields. It can still be minimized through language templates, exercise
identity, elapsed time, or analytic-summary shortcuts. Event registration may
use a separate token or marked-hazard loss on the same state; its gradients and
conflicts with the analysis objective must be measured. High count accuracy is
necessary substrate evidence, not proof that a decoder is justified.

The decisive tests are therefore:

- the same motion under different protocols, requiring invariant physical facts
  but changed protocol-relative judgments;
- identical endpoints or pose multisets with different temporal order;
- identical current pose with opposite histories;
- deletion or counterfactual editing of the causally relevant joint/time span
  versus matched irrelevant interventions;
- held-out compositions of previously seen primitives such as `COMPARE`,
  `BEFORE`, `DURATION`, `REVERSAL`, `PROTOCOL_CHECK`, and
  `EVIDENCE_LOCALIZE`; and
- matched fixed-head, query-conditioned temporal, and handcrafted-summary-to-LLM
  controls under the same data and compute accounting.

Primary evidence is whole-analysis exact match and field F1, held-out
composition accuracy, evidence interval/body-part accuracy, protocol
counterfactual consistency, calibration, and grounding-intervention effect.
Boundary error, false increments/hour, count error, duplicates, and event delay
measure the supporting state substrate. A fluent JSON record, rep count, action
class, or attention plot alone does not support H1.

UPS is the direct pose/autoregressive architecture prior; VideoLLM-online
supplies query/time-alignment and silence-objective precedents; the compact pose
systems supply recurrent/graph/SSM controls. PT is the first bounded falsifiable
domain, not the representation boundary. Pose alone must not be asked to infer
pain, force, load, intent, energy, object state, or work quality. No
context-rollover, phone, multi-person, coaching, industrial, or robot-control
claim is required before this first capability measurement.

### Stage 2: implementation equivalence and longer retained state

The Stage-1 model is already causal and stateful; this stage hardens its efficient
incremental path and extends retention. Compare full-prefix and incremental
execution on the same prefixes. Numerical differences should be reported
as a response surface across precision, kernel, sequence length, and state path,
not forced below an arbitrary global tolerance. Measure hidden-state error,
logit/probability error, answer flips, and task-metric changes. CoST-GCN's
`forward_step` tests are a useful engineering pattern; InfoGCN++ and DDCN show why
a causal mask or dormant queue class is insufficient evidence.

### Stage 3: untrimmed event behavior

Only this stage requires continuous sessions with rest, aborts, restarts,
exercise changes, and occlusion. JCR-RNN, SSNet, and gesturelets are the real
controls. Evaluate boundary timing distributions, false commits per hour,
partial/aborted-event errors, and prefix-conditioned semantics rather than only
trimmed observation-ratio accuracy.

### Stage 4: richer proactive decisions and arbitrary-time queries

Extend the Stage-1 analysis interface with explicit semantic response,
abstention, and arbitrary-time query policies. If a learned commit policy is
used, one possible control target is

\[
\mathcal L=\mathcal L_{\mathrm{AR}}
+\lambda\,\mathcal L_{\mathrm{commit}},
\qquad
\mathcal L_{\mathrm{commit}}
=-\sum_t \log p_\theta(A_t^*\mid X_{\leq t},Q,S_{t-1}),
\]

where $A_t^*$ is wait/commit/abstain supervision. The gradient on pre-event
frames pushes probability toward wait; the gradient at sufficiently observable
event frames pushes toward commit. This objective can collapse to always-wait or
over-commit under severe class imbalance, so report precision-recall across
decision costs and false commits per hour. Stream-VLM and VideoLLM-online are
controls, not compact feasibility evidence.

### Stage 5: rollover memory

Compare at least rolling local state, recurrent/SSM state, compressed KV, and a
local-state-plus-event-ledger design. Construct delayed queries for which the
decisive motion has left the raw window. Measure retained-answer accuracy, false
memory, interference after subsequent similar events, and state bytes versus
stream length. StreamingVLM supplies cache mechanics but not a body-event
registration proof.

### Stage 6: sustained edge deployment

Only after a learned capability survives the earlier interventions should phone
or embedded measurements be interpreted as a capability-efficiency result.
Measure the complete camera -> pose -> tokenizer/state -> decoder path: accepted
pose rate, p50/p95 latency, peak memory, energy, dropped timestamps, sustained
thermal throttling, and decision flips under quantization. Pūioio and the
Rockchip system are deployment controls; A100/H100 FPS is not phone evidence.

## Novelty boundary after this search

Prior art rejects the following claims:

- first online or causal skeleton action system;
- first compact stateful pose model;
- first direct pose plus autoregressive Transformer decoder;
- first skeleton event/boundary detector or rep counter;
- first learned “when versus what to say” streaming decoder;
- first long-stream VLM with bounded KV eviction; or
- first on-device pose-analysis pipeline.

A defensible contribution could instead be a measured capability-efficiency
frontier: direct pose plus a compact decoder that learns held-out conditional
body/temporal composition, followed by separately demonstrated incremental
state, untrimmed event semantics, calibrated asynchronous decisions, memory
rollover, and sustained edge operation. Publication value may live in the
representation, shortcut-resistant task construction, state/memory objective,
or the scaling/mechanistic evidence rather than in attaching two known modules.

## Code audit provenance

Official repositories and exact revisions inspected:

- CoST-GCN: `LukasHedegaard/continual-skeletons`,
  `a8fe2937a33f24cce65c1f8c2fc41081bceda721`
- DDCN+SRM: `Powercoder64/DDCN_SRM`,
  `547f415445d8694e2c2ab637d34930462aa77637`
- InfoGCN++: `stnoah1/infogcn2`,
  `8c44085ac134f35dc85f683dcc775fdb43172ad1`
- PoseRAC: `MiracleDance/PoseRAC`,
  `469590b611bde3595eaf163b517263da634e2096`
- USDRL: `wengwanjiang/FoundSkelModel`,
  `0b4d60c3430a6e777eda85790c29c1bc0cbfbe98`
- Stream-VLM: `Qualcomm-AI-research/FitCoach`,
  `a573ee0b98938b81c1b3e19f94c5fab5d1856f26`
- VideoLLM-online: `showlab/videollm-online`,
  `3da632217258b830a9a594bdabd7a40ae99646dc`
- StreamingVLM: `mit-han-lab/streaming-vlm`,
  `a940d8cbfa14eefe651b7f4adeb81825c28cc6c6`
- Online skeleton early-fusion false positive:
  `fabro66/Online-Skeleton-based-Action-Recognition`,
  `de00cbf17ceea98a7d07f68bbbd966bfd02d3b40`

The local clones were inspection artifacts only and are not treated as retained
experimental evidence. Code observations above are reproducible from the named
public revisions. Parameter/latency results not explicitly labeled as measured
code audit are sourced from the papers and remain conditional on their stated
datasets and hardware.
