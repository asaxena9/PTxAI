# The approximately one-billion-parameter body-native backbone frontier

Date: 2026-08-13  
Status: founder-directed field review and execution decision  
Evidence boundary: model architecture, release, parameter, context, and license
facts below are **SOURCED** from official primary pages accessed 2026-08-13.
Their suitability for body-native intelligence is **HYPOTHESIS** until measured
on the same admitted body-data surface. Generic language or vision benchmarks
are not body-understanding evidence.

## Decision

Start the next new decoder tier near one billion parameters, as a small
architecture portfolio rather than a single preferred-company model:

1. `Qwen/Qwen3.5-0.8B-Base` — current hybrid, unified vision-language candidate;
2. `LiquidAI/LFM2.5-1.2B-Base` — direct continuity with the existing LFM2.5
   350M body-input seam;
3. `allenai/OLMo-2-0425-1B` — dense, fully open scientific control.

Use one screening seed per family. Do not spend the first three allocations on
three seeds of one architecture. Replicate only a positive or
decision-relevant ambiguous screening result; require at least three completed
seeds before a reproducible-winner claim.

The deployment goal remains below 100M. The approximately 1B tier is a
capability-discovery instrument and possible teacher. A positive result sends
the next scale experiment downward. If the valid first-wave screens yield no
positive, run one bounded same-family 2--3B diagnostic; a larger success then
becomes an upper bound for descent by a measured scale curve, direct training,
distillation, pruning, state compression, or their controlled comparison.

## Why these three first

| Candidate | Official facts | Why it is in the first wave | Important boundary |
|---|---|---|---|
| Qwen3.5-0.8B-Base | Official card calls the language model 0.8B, hidden size 1,024, 24 layers, with 18 Gated DeltaNet and 6 gated-attention layers; tied 248,320-token embedding/head; native 262,144-token context; causal LM plus vision encoder; Apache-2.0 | Most current hybrid and only first-wave checkpoint with an official unified vision path. It permits a same-family body-stem versus RGB/vision null rather than comparing unrelated backbones. | Hugging Face reports approximately 0.9B total model size, so the vision encoder and all bypassed/executed parameters must be measured rather than inferred from the name. Body-input compatibility is not yet proven. |
| LFM2.5-1.2B-Base | Exact released tensor count 1,170,340,608; 16 layers = 10 double-gated LIV convolution blocks + 6 GQA blocks; 28T pretraining tokens; 32,768 context; 65,536 vocabulary | Lowest engineering-risk scale-up because the genuine 350M LFM2.5 continuous-embedding route already exists. It directly tests whether the 350M result was capacity-limited within one family. | LFM Open License v1.0, not Apache-2.0. Commercial use is conditioned on the legal entity remaining below the stated $10M annual-revenue threshold; exact entity/license review remains required. |
| OLMo-2-0425-1B | Exact released tensor count 1,484,916,736; Transformer-style autoregressive model; 16 layers, hidden size 2,048, 16 attention heads, context 4,096; 4T-token first pretraining stage; code/model Apache-2.0; intermediate checkpoints released | Dense fully open control. It prevents a result from being misread as a universal verdict on one-billion-parameter models when it may be specific to recurrent/linear-attention machinery. Intermediate checkpoints also support training-dynamics research. | Short context relative to Qwen/LFM. The body adapter and padding/position behavior require a separate compatibility measurement. |

Use base checkpoints for the primary representation comparison. Instruction
variants may be evaluated later for language usability, but post-training must
not be confounded with the question of whether body-native alignment was learned.

## Measured adapter contract for the non-Liquid first wave

The following is **MEASURED CPU/meta compatibility evidence**, not a capability
result. It was replayed against the pinned official configs/tokenizers and
Transformers 5.9.0 without loading model weights.

- `Qwen/Qwen3.5-0.8B-Base` is a multimodal
  `Qwen3_5ForConditionalGeneration`, not a drop-in causal-LM replacement. Body
  embeddings must enter `model.language_model`; `model.visual.*` remains frozen;
  and the tied `lm_head` is used for decoding. A generic `all-linear` PEFT call
  on the wrapper is prohibited because it would also adapt 50 vision linears.
  The intended text-core rank-8 condition wraps 186 DeltaNet, attention, and
  MLP linears (5,411,328 LoRA parameters). The pinned revision is
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- `allenai/OLMo-2-0425-1B` is a plain `Olmo2ForCausalLM`, but its released config
  has `tie_word_embeddings=false`; the native output head is therefore a
  separate parameter and the Liquid tied-head assertion must not be transferred.
  Its intended text-core rank-8 condition wraps 112 attention/MLP linears
  (6,029,312 LoRA parameters), excluding the output head unless head adaptation
  is separately factored. The pinned revision is
  `a1847dff35000b4271fa70afc5db10fd29fedbdf`.
- Both tokenizers exactly round-trip all 30 canonical target strings, but neither
  shares Liquid token IDs. Each backbone therefore receives its own sealed token
  grammar while the canonical text and parsed-field metrics remain identical.
  Raw token NLL is not a cross-backbone metric. Including EOS, Qwen targets use
  14--18 tokens and OLMo targets use 13--17.
- Cached generation is not admitted merely because teacher forcing runs. The
  reference is uncached direct-backbone decoding with explicit absolute physical
  positions; cached/batched decoding must match its first- and second-token
  logits. Qwen also needs a batch-one padding-state regression and must bypass
  mutable multimodal-wrapper `rope_deltas` by calling the text tower directly.

The smallest A100 gate for each backbone is one official-weight maximum-shape
forward/backward step for both all-core LoRA and full tuning, with exact optimizer
ownership, padding perturbation, teacher/free alignment, memory, timing, and
automatic shutdown. Passing that gate establishes interface/trainability only.

Primary implementation sources: the official
[Qwen model/config](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base),
[Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
[OLMo model/config](https://huggingface.co/allenai/OLMo-2-0425-1B), and
[Transformers OLMo2 implementation](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/olmo2/modeling_olmo2.py), accessed 2026-08-13.

## Not discarded, but second wave

- **Falcon-H1-1.5B-Base** is a hybrid Transformer--Mamba comparator and has a
  matched official 3B family point. Its Falcon license and integration surface
  make it a second-wave candidate after the Apache Qwen/OLMo and already-used
  Liquid path.
- **Mamba2-1.3B** is an Apache-2.0 pure state-space mechanistic comparator. Its
  older/base-only release makes it weaker as the headline semantic-capability
  model, but useful if the retained-state mechanism is the unknown.
- **RWKV7-Goose-World3-1.5B-HF** is an Apache-2.0, 1.527B-tensor,
  attention-free recurrent/linear-attention candidate trained on 3.119T tokens.
  It is the strongest additional pretrained constant-state arm, but its custom
  code must be pinned and audited before execution.
- **Qwen2.5-1.5B** is an Apache-2.0 dense decoder with 1,543,714,304 released
  tensor parameters. It is a useful dense semantic-ceiling alternative if the
  Qwen3.5 vision/hybrid path prevents a clean body-stem comparison.
- **InternVL3.5-1B-Pretrained** is an Apache-2.0 RGB observability control with
  an official approximately 0.3B vision + 0.8B language = 1.1B decomposition.
  It samples visual frames and does not by itself establish continuous retained
  physical state; use it to test what RGB access adds, not as a body-native win.
- **Llama-3.2-1B** and **Gemma-3-1B** remain credible dense reference points.
  OLMo is preferred initially because it is fully open and exposes more of the
  training process; Qwen is preferred for the current multimodal/hybrid surface.
- **BrainChip TENNs-1.239B** remains a valuable recurrent-mechanics datum, not
  the one-billion-model answer. The public checkpoint's noncommercial license
  and custom-code/security boundary prevent treating it as the default product
  backbone.
- **MotionBricks** is a motion tokenizer/generator and potential executor or
  counterfactual generator. Its approximately 23.5M tokenizer, 150M pose model,
  and 50M root model do not constitute a one-billion semantic decoder result.

## One diagnostic approximately 3B escalation after a negative 1B wave

Let `S_m(C)` be the held capability score for model family `m` after compute
`C`, accompanied by the body-selectivity contrast

`D_m(C) = Delta_error(relevant body edit) - Delta_error(matched irrelevant edit)`.

The founder's scale policy is directional:

- a real approximately-1B positive sends the program downward immediately;
- no positive across the well-executed approximately-1B screening wave sends
  one same-family branch upward to 2--3B as a capacity diagnostic;
- an invalid or infrastructure-failed run is repaired and is not a scientific
  negative.

The larger diagnostic is most interpretable when all of the following are
measured:

1. the approximately 1B run completes correctly and its training objective is
   still improving at the compute ceiling;
2. held `S_m(C)` or `D_m(C)` has a favorable nonzero trend rather than a flat
   shortcut surface;
3. the model has not already fit training while failing new people/sources;
4. target observability and label/data validity have not failed;
5. the larger run is in the same family or otherwise isolates capacity from an
   architecture change.

Preferred paired escalations are:

- Qwen3.5 0.8B -> Qwen3.5 2B;
- LFM2.5 1.2B -> LFM2 2.6B (noting this changes LFM generation and training
  history as well as capacity, so interpretation is family-level, not a pure
  parameter-only ablation);
- Falcon-H1 1.5B -> Falcon-H1 3B if a second-wave Falcon screen has information
  value.

The single larger cell is a diagnostic of capacity versus formulation, not a
rescue attempt. If it preserves the same pose-zero equivalence, shortcut
dependence, unobservability, or held-person collapse, stop the scale ladder and
route to data, interface, objective, or observability work.

## Seed allocation and training duration

One seed is the correct screening unit, not the confirmation unit.

- **Screen:** one seed per materially different model family under the same
  admitted data, target, evaluation, and bounded budget.
- **Repeat:** run a second/third seed only for a positive result, a result near a
  decision boundary, or a suspected stochastic failure whose resolution changes
  the next allocation.
- **Stop/reformulate:** do not repeat a clear shortcut, unobservability,
  catastrophic held degradation, or infrastructure-independent null merely to
  hope for a favorable seed.
- **Promote:** require at least three completed seeds plus fresh-person/source
  evidence. Seed replication measures optimizer variance; it does not create new
  human support.

Training stops by a held learning curve, not by training loss or a round update
count. At log-spaced compute checkpoints, estimate

`g_k = (S(C_k) - S(C_(k-1))) / log2(C_k / C_(k-1))`.

Call a plateau only when the conservative upper gain is below a calibrated
minimum useful effect for two successive intervals across confirmatory seeds,
with no continuing improvement in calibration, subgroups, or causal
selectivity. A still-rising held curve at the ceiling is **budget-limited**, not
plateaued. Pose-zero equivalence, sustained held degradation, or shortcut
dependence can stop a run earlier.

The executable training-sufficiency contract is
`experiments/autoresearch/training_sufficiency_v0/`.

## A100 memory envelope

For full AdamW tuning with BF16 parameters and gradients plus FP32 master weights
and moments, a common static approximation is

`M_static(P) = (2 + 2 + 4 + 4 + 4)P = 16P bytes`,

before activations, temporary buffers, and framework overhead. Therefore:

- 1.0B -> approximately 16 GB static;
- 1.17B -> approximately 18.72 GB static;
- 2.0B -> approximately 32 GB static;
- 2.57B -> approximately 41.1 GB static;
- 3.0B -> approximately 48 GB static.

These are **DERIVED planning estimates**, not measured peaks. A one-billion LoRA
screen easily fits one high-RAM A100; full tuning should also be feasible with
activation checkpointing and calibrated microbatches. Approximately 3B LoRA
fits one A100, while full tuning may require memory-saving optimizer/state
sharding or two to three A100s. Every candidate must profile actual resident,
executed-active, trainable, optimizer, activation, and recurrent/KV state before
the scientific run.

**Measured update, 2026-08-13:** the official pinned LFM2.5-1.2B full-training
path passed a one-step A100-SXM4-80GB calibration with FP32 optimizer-owned
weights, BF16 autocast, activation checkpointing, and microbatch one. Peak Torch
reserved memory was 19.402 GiB, leaving 59.848 GiB device reserve; the largest
tested one-step-safe learning rate was `1e-5`. This closes feasibility for that
exact path only. It is not a body-semantic outcome, a throughput projection, or
evidence that Qwen/OLMo share the same memory behavior.

**Measured Qwen update, 2026-08-13:** the official pinned
`Qwen/Qwen3.5-0.8B-Base` text-tower full-training path passed a one-step
A100-SXM4-80GB calibration. The release loaded 852,985,920 unique parameters;
the executed/trainable text+body route contains 752,567,488 parameters and the
100,592,896 vision parameters were frozen and unexecuted. Peak Torch reserved
memory was 16.125 GB, every one of 334 optimizer-owned tensors had a finite
gradient and sampled nonzero update (331 had nonzero raw gradients), and the
bounded first-step rule selected exploratory LR `1e-5`. This is feasibility,
not capability. A separate cache probe found token-1 max logit difference zero
but token-2 difference 0.29296875 under BF16, exceeding the 0.03 tolerance;
therefore cached generation is not admitted and the Qwen science path must use
the uncached direct text-tower reference.

## Immediate execution order

1. The previous 350M/recurrent allocations are closed without a complete
   scientific outcome: recurrent failed evidence closure and Round-2 attempt
   004 lost both final result archives after remote completion. Do not rerun
   them automatically; carry their transport and adaptation limits into the
   approximately-1B runner.
2. Keep the completed 350M semantic-canary seam; parameterize the checkpoint
   identity instead of launching it as the next new tier.
3. While GPUs are occupied, prove no-allocation compatibility for the three
   first-wave base checkpoints, beginning with LFM2.5-1.2B because it reuses the
   current interface and Qwen3.5-0.8B because it is the strongest field-wide
   alternative.
4. When the current portfolio releases GPUs, run one screening seed per family,
   up to three independent A100 allocations, with identical data/evaluation and
   automatic termination.
5. Use the measured failure signature—not generic leaderboards—to choose
   replicate, scale up, descend, or reformulate.

## Primary sources (accessed 2026-08-13)

- Qwen3.5-0.8B-Base official model card:
  https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- Qwen3.5-2B-Base official model card:
  https://huggingface.co/Qwen/Qwen3.5-2B-Base
- LFM2.5-1.2B-Base official model card:
  https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base
- LFM2 family official model card (including exact 2.6B point):
  https://huggingface.co/LiquidAI/LFM2-2.6B
- LFM Open License v1.0:
  https://www.liquid.ai/lfm-license
- OLMo-2-0425-1B official model card and config:
  https://huggingface.co/allenai/OLMo-2-0425-1B
  https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json
- Falcon-H1 official 1.5B and 3B cards:
  https://huggingface.co/tiiuae/Falcon-H1-1.5B-Base
  https://huggingface.co/tiiuae/Falcon-H1-3B-Base
- Mamba2-1.3B official checkpoint and implementation:
  https://huggingface.co/state-spaces/mamba2-1.3b
  https://github.com/state-spaces/mamba
- RWKV7-Goose-World3-1.5B-HF official model card:
  https://huggingface.co/RWKV/RWKV7-Goose-World3-1.5B-HF
- Qwen2.5-1.5B official model card:
  https://huggingface.co/Qwen/Qwen2.5-1.5B
- InternVL3.5-1B-Pretrained official model card:
  https://huggingface.co/OpenGVLab/InternVL3_5-1B-Pretrained
- Llama-3.2-1B official model card:
  https://huggingface.co/meta-llama/Llama-3.2-1B
- Gemma 3 official model card:
  https://ai.google.dev/gemma/docs/core/model_card_3
