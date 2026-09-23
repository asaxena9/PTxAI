# Synthetic temporal-information audit

Run date: 2026-08-07  
Evidence class: `MEASURED-PILOT` on one procedural generator only

## Question

Does the deployed 24-frame TCN's synthetic phase score arise from ordered
temporal history, or can a current-frame or order-invariant model recover the
same labels?

All models used the same generated archive, normalization, weighted joint
exercise-phase cross-entropy, virtual-ID splits, validation selection rule, and
five seeds. The primary metric was held-out phase macro-F1 conditioned on the
known exercise.

## Results

| Representation/model | Parameters | Phase macro-F1, mean ± seed SD | Paired difference from TCN, mean | 95% t-interval of difference |
|---|---:|---:|---:|---:|
| Last frame, no derivative features | 1,103 | 0.6220 ± 0.0098 | -0.2651 | [-0.2815, -0.2486] |
| Last frame, with derivative features | 1,231 | 0.7778 ± 0.0037 | -0.1093 | [-0.1247, -0.0938] |
| Raw mean plus current-frame MLP | 1,935 | 0.8490 ± 0.0041 | -0.0380 | [-0.0575, -0.0185] |
| Parameter-matched nonlinear DeepSets mean/current | 2,347 | 0.8533 ± 0.0020 | -0.0337 | [-0.0503, -0.0172] |
| Causal TCN | 2,351 | 0.8870 ± 0.0121 | 0 | reference |

Mean per-class F1:

| Model | ready | moving | holding | returning | paused |
|---|---:|---:|---:|---:|---:|
| Last, no derivatives | 0.824 | 0.303 | 0.691 | 0.306 | 0.986 |
| Last, derivatives | 0.815 | 0.687 | 0.711 | 0.686 | 0.990 |
| Raw mean/current | 0.863 | 0.800 | 0.802 | 0.811 | 0.970 |
| Parameter-matched DeepSets | 0.868 | 0.811 | 0.798 | 0.805 | 0.986 |
| Causal TCN | 0.902 | 0.867 | 0.850 | 0.865 | 0.952 |

## Interpretation

Three conclusions survive within this generator:

1. Handcrafted derivatives carry substantial phase information. Adding four
   last-step difference features raises mean F1 from 0.622 to 0.778, primarily
   by separating outward `moving` from `returning`.
2. Most of the TCN score is available without historical order. A
   parameter-matched order-invariant model reaches 0.853 versus the TCN's
   0.887.
3. Ordered causal history still contributes in this task. The paired TCN
   advantage over the four-parameter-smaller DeepSets model is 0.0337, and the
   five-seed t-interval excludes zero.

This strengthens only the claim that ordered history is useful for the
procedural phase labels. It weakens any description of the current result as
long-context reasoning: an order-invariant summary recovers about 96% of the
TCN's absolute macro-F1.

## Validity failure in the underlying generator

The reported 30/8/12 “virtual subjects” are not persistent synthetic people.
The seed used to sample thigh, shank, and torso includes exercise and
repetition, so the same subject ID receives new anatomy in each sequence. IDs
are disjoint, but identity-level anatomy is not held constant. Adjacent
24-frame examples also overlap by 23 frames.

Therefore this audit must not be interpreted as subject generalization or as
having 12,339 independent held-out samples. The independent held-out structure
is at most 180 generated one-repetition sequences from one generator family.

## Updated belief

- `MEASURED-PILOT`: temporal order improves phase classification inside the current generator.
- `UNKNOWN`: whether the benefit transfers to real pose-estimator trajectories or continuous sessions.
- `UNKNOWN`: whether longer than local kinematic history is needed for partial/restart/error/feedback decisions.
- `REJECTED INTERPRETATION`: the current synthetic experiment demonstrates generalization to coherent unseen virtual subjects.

The next evidence-bearing experiment must use real subject-held-out motion and
must include a competitive order-invariant or kinematic temporal baseline.

Raw results:

- `results/2026-08-07/results.json`, SHA-256 `c8c6ed16c23003264175a70128362896af2e802884dd65fee711c978afd9211c`
- `results/2026-08-07-order-matched/results.json`, SHA-256 `4ebbf3779dbce4abebe921bc621744f0bedfcda6a8d0db6d971477642297c6a4`
