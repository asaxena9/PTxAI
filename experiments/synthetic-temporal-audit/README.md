# Synthetic temporal-information audit

Status: preregistered generator/model audit. This experiment cannot establish behavior on people.

## Question

The current model reports high phase F1 on generated data, but each input frame
already contains handcrafted first-difference features. Does a 24-frame
order-sensitive TCN add predictive information, or is almost all performance
available from the final engineered feature vector or an order-invariant
summary?

For a window (W\in\mathbb{R}^{T\times F}), exercise (E), and phase (Y),
compare representations:

\[
R_{last}=W_T,
\quad
R_{last,no\Delta}=W_{T,1:18},
\quad
R_{bag}=[T^{-1}\sum_t W_t;W_T],
\quad
R_{tcn}=\operatorname{TCN}(W).
\]

All models minimize the same class-weighted joint exercise-phase
cross-entropy. The primary estimand is paired held-out phase macro-F1 when the
exercise is known, because Relay already has the confirmed prescription:

\[
\Delta_m=
F1_{phase}(m)-F1_{phase}(R_{tcn}).
\]

The experiment estimates these differences across five training seeds; it does
not impose a founder-supplied pass threshold.

## Competing hypotheses

- **H1, long-context value:** the TCN materially exceeds both last-frame and
  order-invariant models, indicating useful information in ordered history.
- **H2, engineered-derivative sufficiency:** the last-frame model with
  derivatives approaches the TCN, while removing derivatives mainly damages
  moving-versus-returning discrimination.
- **H3, bag sufficiency:** mean-plus-last pooling approaches the TCN, suggesting
  temporal order is not the source of the reported gain.

The models are not exactly parameter matched, so parameter counts are reported.
Any result is evidence about generator/model structure only. The synthetic
generator may encode labels through unrealistic shortcuts.

## Fixed protocol

- Existing archive and virtual-subject splits are unchanged.
- Five declared seeds: 1103, 2207, 3301, 4409, 5519.
- Same normalization, weighted CE, AdamW, batch size, maximum epochs, and early stopping.
- Model selection uses validation phase macro-F1 conditioned on known exercise.
- Held-out data are evaluated only after selecting the validation checkpoint.
- Report per-seed values, means, standard deviations, paired differences,
  parameter counts, and per-class phase F1.

Run:

```bash
python3 experiments/synthetic-temporal-audit/run.py
```

## Follow-up: parameter-matched order test

The initial result showed that a 1,935-parameter mean-plus-last MLP trailed the
2,351-parameter TCN. That difference may reflect temporal order, nonlinear
per-frame encoding, or capacity. Before running the follow-up, we define a
2,347-parameter DeepSets-style comparator:

\[
R_{set}=\left[
T^{-1}\sum_t\phi(W_t);\phi(W_T)
\right],
\qquad
\hat Y=\operatorname{softmax}(W R_{set}+b).
\]

The shared nonlinear map (\phi\) has width 44. This model is invariant to the
order of historical frames except for the separately preserved current frame,
and has four fewer parameters than the TCN. It is trained with the identical
splits, objective, seeds, and validation selection rule. If the TCN retains a
paired advantage, the evidence for ordered-history value is stronger; if the
gap closes, the original comparison attributed capacity/nonlinearity to time.

```bash
python3 experiments/synthetic-temporal-audit/order_matched.py
```
