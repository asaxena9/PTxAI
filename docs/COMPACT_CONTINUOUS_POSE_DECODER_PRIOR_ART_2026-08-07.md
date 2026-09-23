# Compact continual pose-decoder prior-art audit

Date: 2026-08-07  
Status: living sourced audit and research handoff; not a novelty or capability claim

## Correct research question

The connector question is already answered: skeletal pose or motion can be
projected or tokenized into a language/generative decoder. The unresolved
question is the capability--efficiency--continuity intersection:

> Can a compact model incrementally ingest a live, untrimmed human pose stream,
> preserve a useful body state as raw context rolls over, and produce grounded
> temporal answers, predictions, structured events, or language under measured
> on-device constraints?

Physical therapy is a proving ground and source of falsifiable tasks, not the
definition of the underlying capability. One person is sufficient for the first
research claim. Multi-person tracking, clinical intervention policy, and broad
industrial transfer are later questions unless an experiment makes them
direction-changing.

### Operational definitions

- **Direct/native pose input:** coordinates, confidences, timestamps, or learned
  per-pose embeddings enter the learned sequence model. A pipeline that supplies
  only handcrafted cycle summaries is not direct pose decoding.
- **Continual/online:** at time `t`, the prediction is a function of observations
  available by `t`, and the implementation updates persistent state without
  re-encoding a completed future-containing clip.
- **Decoder intelligence:** query-conditioned or generative behavior that tests
  compositional body and temporal information, not only a fixed action class.
- **Edge evidence:** sustained measurements of the complete camera/pose/model
  path on the target device. Parameter count or A100 throughput is not edge
  evidence.

The word *continuous* in continuous sign-language recognition usually means an
unsegmented sentence rather than live incremental inference. It must not be
treated as evidence of a continual deployment implementation.

## What the closest systems actually establish

| System | Size and input | What it establishes | What it does not establish |
|---|---|---|---|
| [FiGMo](https://arxiv.org/html/2606.20888) (2026) | VideoLLaMA3-7B; every 2D/3D pose is independently encoded by a MotionBERT-derived spatial encoder, translated by a two-layer MLP into an LLM token, and paired with explicit timestamp text | Strong offline fine-grained body and temporal QA, duration/order/start/end supervision, dense temporal captioning, and detected-2D exercise feedback. Its timestamp ablation falls from 0.758 to 0.711 BABEL-QA accuracy; the broader supervision mixture matters more than staged scheduling. | It receives a bounded clip. Input is limited to 800 poses at 30 Hz and longer clips are uniformly subsampled. It has no persistent live state, context rollover, compact model, or phone evidence. Code/data release is promised rather than presently available in the paper. |
| [AutoSign](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Johnny_AutoSign_Direct_Pose-to-Text_Translation_for_Continuous_Sign_Language_Recognition_ICCVW_2025_paper.pdf) (ICCVW 2025) | **87.9M reported** for the selected system: frame-wise body/hand/face 2D pose, two stride-2 temporal CNNs, then an AraGPT2-family autoregressive gloss decoder | A compact, direct pose-to-language decoder can learn useful sequence semantics. It reports 20.5% test WER on Isharah-1000 versus 33.2% for its pose Transformer+CTC baseline. | Each padded/truncated pose sequence is supplied as a whole. No incremental state/cache, arbitrary-time queries, context rollover, calibrated commit/wait behavior, mobile measurement, or released implementation was located. The paper's training descriptions are internally inconsistent, so reproduction requires clarification. |
| [MotionGPT](https://arxiv.org/abs/2306.14795) (NeurIPS 2023) | T5-small/base/large language backbones reported as 60M/220M/770M, plus a separately trained VQ motion tokenizer | Compact motion-language modeling already exists. A shared text/motion vocabulary supports motion captioning, text-to-motion, prediction, and in-betweening. The 220M row was the best overall compromise on the authors' data; 770M was not uniformly better. | It compresses completed motion into discrete codes and evaluates offline clips. The reported LLM size does not include a fully audited end-to-end perception system. It does not test continual observed-pose understanding or edge deployment. |
| [Unified Pose Sequence Modeling (UPS)](https://openaccess.thecvf.com/content/CVPR2023/html/Foo_Unified_Pose_Sequence_Modeling_CVPR_2023_paper.html) (CVPR 2023) | Direct coordinate pose tokenizer, dynamic Transformer encoder, three-layer autoregressive Transformer decoder; total parameters unreported | Direct pose plus an autoregressive unified output vocabulary predates this project. One model emits action tokens, quantized 3D coordinates, and end-of-task tokens for recognition, pose estimation, and early action prediction. | It is an offline task/prefix model, not a language-reasoning model or a persistent live deployment. No official implementation or exact total parameter count was located in this audit. |
| [MotionGPT3](https://arxiv.org/abs/2506.24086) (2025/26) | Reported 238M dual text/motion decoder **backbone** with a continuous motion VAE latent. Local official-code audit measured an 18.03M VAE and 27.95M diffusion head; the derived motion-understanding path is approximately 256.8M and the unified path approximately 284.7M plus small extras. | A sub-350M continuous-motion/decoder architecture exists, including shared attention between modality-specific paths. | The complete HumanML3D clip is compressed to one non-causal `1 x 256` latent before decoding. There is no incremental observed-pose state or edge result. |
| [Human Motion Instruction Tuning / LLaMo](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Human_Motion_Instruction_Tuning_CVPR_2025_paper.pdf) (CVPR 2025) | Native motion features and optional video are selected/fused by a query-conditioned module before an autoregressive LLM | Motion/video QA, fine-grained body reasoning, swing analysis, and repetition counting can be instruction-tuned. | Full clips and selected key poses are used; explicit pose time and live state are absent. The model is not compact or edge-demonstrated. The official repository currently exposes documentation rather than a complete runnable release. |
| [BiomechGPT](https://arxiv.org/abs/2505.18465) (revised 2026) | Gemma-family 1B--12B systems; shared biomechanical tokens, including an ablation that projects continuous soft motion tokens into Gemma-3 | Clinical motion-to-language and multi-task biomechanics modeling have been attempted at meaningful data scale. Continuous projected motion embeddings are not themselves a new connector. | Offline trials, large models, and no continual edge deployment. The paper also shows that raw continuous task-specific baselines can remain strong, so language-tokenization is not automatically superior. |
| [Stream-VLM / QEVD](https://proceedings.neurips.cc/paper_files/paper/2024/file/8aa695b5d6bf9e958d4e56b22c9b859d-Paper-Datasets_and_Benchmarks_Track.pdf) (NeurIPS 2024) | 3D CNN video stream fused into LLaMA-2-7B | A generative model can learn asynchronous `<next>` versus `<feedback>` behavior over a live fitness stream. QEVD contributes hundreds of hours and long-session annotations. | Raw video rather than direct pose; 7B; bounded training segments; no compact or phone deployment proof. |
| [BioCoach](https://arxiv.org/abs/2603.26938) (2026 preprint) | Streaming video plus extracted 3D skeletal kinematics and structured biomechanical context into a 7B language model | Streaming skeletal context can improve generated exercise feedback over a video-language baseline. | Much of the body state is built by exercise-specific degree-of-freedom selection, cycle analysis, and constraints. It is not a compact pose-native decoder and has no phone evidence. |
| [VideoLLM-online](https://arxiv.org/abs/2406.11816) (CVPR 2024) | Llama-2/3 7B with one projected video token per accepted frame and a continuous KV cache | The exact continual dialogue mechanism exists for RGB: causal frame ingestion, cached state, temporally aligned response/silence training, and buffering while text is generated. | It is 7B RGB video, reported at over 10 FPS on an A100, not compact direct body pose or phone deployment. |
| [HumMUSS](https://openaccess.thecvf.com/content/CVPR2024/papers/Mondal_HumMUSS_Human_Motion_Understanding_using_State_Space_Models_CVPR_2024_paper.pdf) (CVPR 2024) | 16M causal state-space body model operating on 2D pose | A compact model can update body state with constant time/memory per frame and outperform a matched causal Transformer on pose/mesh tasks while adapting to frame-rate changes. | Task-specific pose/mesh heads, not autoregressive semantic queries or structured language; no context registration/recall evaluator. |
| [AIGaitor](https://arxiv.org/abs/2605.21421) (2026 preprint) | On-device iPhone 14 motion-capture and downstream skeleton-analysis pipeline; models include 24M ViTPose-S and lightweight skeleton classifiers | The component technologies can run locally. ViTPose-S is reported at 8 ms/frame and skeleton classifiers at sub-millisecond latency. | Its complete 10-second 4K/60 pipeline takes 77 seconds, so the whole system is not live. It does not contain a compact continual semantic decoder. |

Additional broad precedents include [MotionLLM](https://arxiv.org/abs/2405.20340),
[SkeletonLLM](https://arxiv.org/abs/2603.18003),
[EAI-LLM](https://ojs.aaai.org/index.php/AAAI/article/view/34597),
[UniPose](https://openaccess.thecvf.com/content/CVPR2025/html/Li_UniPose_A_Unified_Multimodal_Framework_for_Human_Pose_Comprehension_Generation_and_Editing_CVPR_2025_paper.html),
[MG-MotionLLM](https://arxiv.org/abs/2504.02478), and
[Superman](https://arxiv.org/abs/2602.02401). They strengthen the conclusion
that pose/motion-language fusion, unified vocabularies, and generative body
tasks already exist. They do not close the compact continual-deployment gap.

## Defensible novelty boundary

The following broad claims are rejected by sourced prior art:

- first body pose plus Transformer decoder;
- first direct pose-to-language decoder below 150M;
- first motion-language model near 60M;
- first streaming generative fitness interaction;
- first compact stateful causal pose model; or
- first on-device phone pose-analysis pipeline.

No primary source located in this audit establishes the complete conjunction:

1. direct 2D/3D pose, confidence, missingness, and real timestamps;
2. causal incremental processing of an untrimmed stream;
3. persistent state across a finite raw-context window;
4. grounded body/temporal queries, predictions, and structured asynchronous
   outputs rather than only a fixed classifier;
5. a compact model, especially below 150M active parameters; and
6. sustained end-to-end phone measurements plus causal body-grounding tests.

That conjunction is an **UNKNOWN/open empirical gap**, not proof of uniqueness.
Its contribution would initially be systems/empirical research, not new
mathematics. A mathematical contribution would require a new consistent
objective, identifiability result, state-compression result, or comparable
formal advance that survives peer scrutiny.

## The model the experiment is actually testing

Let `X_t` contain the accepted pose coordinates, confidence/missingness, camera
or root channels, and elapsed time at physical time `t`. Let `Q_t` be any query
or instruction available by `t`. Offline clip models estimate

\[
q_\theta(Y\mid X_{1:T},Q),
\]

where the completed sequence length `T` is known and future observations are
available. The continual model instead estimates

\[
q_\theta(Y_t,A_t,M_t\mid X_{\le t},Q_{\le t},M_{t-1}),
\]

where `A_t` includes wait/silent/commit behavior and `M_t` is persistent memory.
The implementation must satisfy future-suffix invariance: changing
`X_{t+1:T}` cannot change an output committed at `t`.

“Registering” a completed context does not have to mean generating a prose
summary. It may be a learned recurrent state, selected/compressed KV entries, a
small set of memory tokens, an explicit event ledger, or a hybrid. Four live
hypotheses should be compared rather than choosing one in advance:

1. rolling local KV with no long memory;
2. compressed/retrieved KV, following streaming-video memory work such as
   [LiveVLM](https://arxiv.org/abs/2505.15269);
3. recurrent or state-space memory, following the causal HumMUSS result; and
4. local KV plus a slower event/semantic memory that is written when evidence
   becomes sufficient.

At one accepted pose token per 100 ms, a 1,024-position raw-pose window spans
102.4 seconds before reserving positions for queries and outputs. This is a
derived capacity calculation, not a recommendation. The experiment must vary
pose-token rate, token granularity, memory policy, and model size because a
four-token pose representation would reduce that gross span to 25.6 seconds.

## Staged research program implied by the evidence

The authoritative capability definition is
`NATIVE_BODY_DECODER_CANONICAL_BUILD_2026-08-07.md`. In particular, fixed event
registration is supporting state and an isolated analytic query benchmark is a
diagnostic; neither is the reason to use a decoder.

The large papers are capability anchors and possible teachers, not reasons to
stop the build. The compact papers are controls, not predetermined winners.

The final system conjunction must **not** become the first experiment. Each
stage establishes one additional claim; a failure is allowed to change the next
stage rather than merely block a fixed queue.

### Stage 1: prove native conditional physical analysis

The first hypothesis is deliberately narrower than continual deployment:

> A compact decoder receiving direct pose tokens can maintain movement state and
> perform task/query/protocol-conditioned physical analysis that cannot be
> recovered from a fixed event logger, the last frame, a permutation-invariant
> pose bag, timestamps alone, or handcrafted motion summaries.

Include paired sequences `(X, X')` with the same pose multiset, timestamps,
endpoints and subject but different pose--time assignments, and paired prompts
`(Q,Q')` that apply different protocols to the same motion. Target families must
cover state/facts, temporal/body comparisons, protocol application and grounded
body/time evidence. Hold out compositions of learned operators rather than only
wording. Pair-exact and counterfactual-consistency scores test conditional
physical information use rather than generic answer accuracy.

The first model may be the strongest admitted 270--350M candidate to maximize
the chance of observing the capability. It should emit a structured multi-field
analysis plus evidence time/body-part tokens. Controls are fixed event/state
heads, query-conditioned temporal encoders, last-frame, bag/time, handcrafted
summary, label-only and pose-deranged inputs. This stage does not claim long-
context rollover, edge viability, clinical assessment or decoder superiority
over every sequence model.

### Stage 2: prove continual causal state

Only after Stage 1 establishes temporally grounded body semantics, convert the
same capability to prefix-by-prefix inference and compare rolling KV,
compressed/retrieved KV, recurrent/SSM state, and hybrid event memory. The claim
is incremental equivalence, correct wait/commit behavior, and retention after
raw-context eviction—not new body semantics.

### Stage 3: descend the capability frontier

Once a strong model supplies an achievable reference, test which parts survive
at approximately 150M, 90M, 60M, 25M, and smaller through architecture changes,
distillation, or motion-specific pretraining. These are response-surface points,
not mandatory gates. The claim is capability per parameter/FLOP/byte, not edge
deployment yet.

### Stage 4: prove the deployed observation and device path

Replace clean motion with synchronized phone pose/RGB observations, quantify
the representation loss, and then measure the complete camera+pose+decoder path
on the device. PT untrimmed sessions become the real-world proving task. More
advanced assessment, intervention, RL, multiple people, and industrial transfer
remain separate later claims.

### Capability and data anchor

FiGMo offers the clearest reproducible conceptual curriculum:

- isolated-pose captions and analytic pose QA;
- body-part and counterfactual pose descriptions;
- motion captioning;
- action order, duration, start/end, and in-between QA; and
- detected-2D downstream transfer.

Its published module table contains approximately 1.013M supervision rows, but
many rows are alternate annotations or generated descendants of the same source
motions. They increase optimization signal and task coverage, not independent
human support. A local data engine must retain the ancestor motion and split all
descendants together.

### Candidate model families after the first proof

Run the same causal pose stream and output suite across, at minimum:

- a 270--350M hybrid/decoder capability model;
- a 125--150M direct GPT/attention model, with AutoSign as the closest task
  precedent;
- a 60M T5-style motion-language control, following MotionGPT;
- a causal SSM/recurrent model informed by HumMUSS;
- a graph/spatial body encoder coupled to each temporal family; and
- direct discriminative heads using the identical hidden state.

This is not a size staircase or permission gate. Allocation should be adaptive,
but every claim about a decoder benefit requires a matched hidden-state/direct
head comparison. A language decoder may improve conditional composition while
adding no value to boundary timing; those are separable results.

### Later continual/deployment test suite

1. **Static body queries:** analytically verifiable joint relation, symmetry,
   angle-bin, and counterfactual questions.
2. **Local temporal queries:** before/after/in-between, duration, phase order,
   reversals, repetitions, and future motion at several physical horizons.
3. **Causal availability:** reveal the stream prefix by prefix; the target is
   answer, wait, or abstain at each time, with no future-derived label leakage.
4. **Untrimmed state:** partial, aborted, resumed, restarted, rest, exercise
   change, dropped frames, and occlusion/reacquisition.
5. **Memory rollover:** ask questions whose decisive evidence has left the raw
   window; measure retained answer accuracy and false memory as stream length
   grows.
6. **Grounding interventions:** remove or counterfactually edit the relevant
   joint/time span and compare with equal-sized irrelevant edits.
7. **Edge measurement:** sustained camera+pose+decoder p50/p95 latency, memory,
   dropped timestamps, power, and thermal behavior.

The principal scientific comparison is not “decoder versus no decoder.” It is
which representation, state update, memory-write mechanism, and output objective
preserve the most causally usable body information per unit of compute and per
independent motion episode.

## Immediate handoff correction

Do not halt top-down LFM/MobileLLM/motion-model experimentation because FiGMo or
AutoSign exists. Their existence changes the claim and supplies controls:

- FiGMo is a large offline capability/teacher anchor.
- AutoSign is the direct sub-150M offline pose-to-language baseline.
- MotionGPT-60M is the compact discrete-motion-language baseline.
- HumMUSS-16M is the compact continual body-state baseline.
- VideoLLM-online/Stream-VLM supply the continual response/cache mechanisms.
- AIGaitor supplies the current phone-pipeline deployment reference.

The constructive research task is to combine and then ablate these properties,
not to assume their combination will work. Preserve failed runs and let the
results change the representation, memory mechanism, objective, model family,
or target capability.
