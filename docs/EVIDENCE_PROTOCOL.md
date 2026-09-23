# Evidence and autoresearch protocol

Status: repository research contract, 2026-08-07

## 1. Purpose

This project is researching whether a compact pose-native causal decoder can observe continuous human motion, maintain temporal state, emit reliable structured events, and run within phone or edge constraints. The answer cannot be obtained from model size or analogy to a VLM. It must be decomposed into falsifiable claims.

The operating rule is **no unlabeled guesswork**. Hypotheses are necessary, but they remain hypotheses until a reproducible experiment promotes them.

## 2. Evidence vocabulary

| Label | Meaning | Minimum artifact |
|---|---|---|
| `MEASURED-PILOT` | A reproducible measurement with intentionally limited scope | command, manifest, raw result, environment, limitations |
| `MEASURED-REPLICATED` | Repeated measurement on the declared seeds/data/hardware | all pilot artifacts plus aggregate and variance |
| `SOURCED` | Claim directly supported by a primary source | URL/DOI, access date, supported claim |
| `DERIVED` | Calculation from measured/sourced inputs | equation, units, inputs, rounding |
| `HYPOTHESIS` | Falsifiable but untested proposal | null, experiment, promotion threshold |
| `UNKNOWN` | Not established or not observable | reason and next discriminating experiment |

Confidence words such as "can," "works," "real time," "understands," "uncertain," or "safe" require the scope to be stated. For example: “the implementation sustained 15 pose tokens/s on an M2 Pro for these three clips” is allowed; “the model runs in real time” is not.

## 3. Claim graph for the pose-native decoder

The proposed system is a conjunction of capabilities. Each node gets a separate gate.

1. **Sensor observability**: the camera and pose stack preserve enough information for the target.
2. **Temporal sampling**: accepted pose packets capture the relevant event bandwidth and timestamp accuracy.
3. **Representation**: pose tokens preserve joint identity, visibility, geometry, time delta, and camera state.
4. **State estimation**: a causal model improves over instantaneous and finite-state baselines on held-out continuous sessions.
5. **Event semantics**: completed, partial, paused, restarted, invalid, and unknown events can be distinguished.
6. **Structured generation**: the decoder emits schema-valid, exactly-once events without materially degrading control-rate outputs.
7. **Forecasting**: future-motion outputs beat persistence and kinematic baselines at declared horizons.
8. **Uncertainty**: confidence predicts error and supports a useful abstention/coverage curve under shift.
9. **Deployment**: sustained latency, memory, energy, and thermal limits are met on each intended device.
10. **Intervention policy**: cue choice and timing improve a predeclared outcome without increasing safety or nuisance failures.

A shared network may implement several nodes. Shared parameters do not merge their evidence.

### Mathematical record required for a claim

Each claim should identify an estimand or objective rather than only a metric name. At minimum record:

- observed variables (X), targets (Y), context/intervention (A), subject/session/device grouping (G), and any latent state (Z);
- source/training distribution (P_{tr}(X,Y,A,G)), evaluation distribution (P_{ev}), and intended deployment distribution (P_*);
- representation (R=f(X)) and the information or symmetry it intentionally removes;
- objective (\mathcal{L}(\theta)=\mathbb{E}_{P_{tr}}[\ell(f_\theta(X),Y)]), including weighting/sampling that changes the effective distribution;
- the behavior the resulting gradient should encourage, plausible shortcut minimizers, and an ablation that separates them;
- uncertainty of the reported estimator, clustered at the independent unit rather than at correlated frames.

Sample complexity is an empirical function, not a guessed dataset number. For a metric (M), estimate a learning curve (M(n,s,d,e)) over independent subjects (n), sessions (s), devices/sites (d), and error-event coverage (e), with repeated subset draws and seeds. Report whether performance is data-limited, coverage-limited, representation-limited, or irreducibly ambiguous under the observed modality.

## 4. Experiment manifest

Every run directory must contain a machine-readable `manifest.json` with:

```json
{
  "experiment_id": "unique-stable-id",
  "claim": "one falsifiable sentence",
  "evidence_before": "HYPOTHESIS",
  "null_hypothesis": "what result rejects the proposed change",
  "primary_metric": {"name": "metric", "direction": "min", "threshold": 0.0},
  "guardrails": [],
  "data": [{"name": "dataset", "version": "version", "sha256": "hash", "license": "terms"}],
  "split_unit": ["subject", "session", "device"],
  "code": {"revision": "git sha or source-tree hash", "entrypoint": "command"},
  "environment": {"hardware": "device", "software": {}},
  "random_seeds": [0],
  "budget": {"wall_clock_minutes": 5, "gpu_hours": 0.0},
  "stop_conditions": [],
  "created_at": "RFC3339 timestamp"
}
```

The result directory must add raw predictions where legally possible, `metrics.json`, stdout/stderr logs, elapsed time, peak memory, and a disposition of `keep`, `reject`, `inconclusive`, or `failed`.

## 5. Autoresearch loop adapted to this project

Karpathy's autoresearch loop holds the dataset/evaluator/time budget fixed, lets an agent alter an in-scope training file, evaluates one objective, and keeps or discards the change. This project needs the same bounded feedback loop, with additional guardrails because a single scalar can hide temporal and safety failures.

For each research track:

1. Freeze the evaluator, data split, compute budget, and output schema.
2. Establish a reproducible baseline.
3. Allow changes only in the declared model/training surface.
4. Run one bounded trial in an isolated output directory.
5. Compare the primary metric and every guardrail with the baseline.
6. Keep the change only when the preregistered rule passes; otherwise reject it and retain the result row.
7. Replicate promoted changes on new seeds, then on the frozen external split.

The ledger is append-only. A trial is never deleted because it failed. Architecture search, dataset changes, evaluator changes, and compute-budget changes are separate experiment families; changing more than one prevents attribution.

### Choosing the next experiment and its gates

Experiment choice is adaptive. Let (\Theta) represent uncertain facts about the problem, (a\) a research or product decision, (U(a,\Theta)) its scientific/deployment utility, and (Y_e) the possible outcome of experiment (e). Prefer experiments with high expected value of information:

\[
\operatorname{EVI}(e)=
\mathbb{E}_{Y_e\mid D}\left[
\max_a\mathbb{E}[U(a,\Theta)\mid D,Y_e]
\right]
-
\max_a\mathbb{E}[U(a,\Theta)\mid D]
-C(e).
\]

This is a decision principle, not a claim that the expectations are always known exactly. When they are not, use explicit ordinal estimates and test sensitivity to them. An experiment is valuable when plausible outcomes would change the next decision, not merely add another metric.

For a comparative metric difference (\Delta=M_{candidate}-M_{baseline}), define promotion using a minimum meaningful effect (\delta), uncertainty clustered at the independent participant/session unit, and non-inferiority limits for guardrails. A typical promotion statement is (P(\Delta>\delta\mid D)\geq 1-\alpha), or the corresponding prespecified confidence-interval criterion. The choices of (\delta) and (\alpha) must be justified by measurement noise, downstream costs, and the stage of research. A founder-proposed number is not that justification.

## 6. Required baselines

Before any decoder tier is credited, compare against:

- instantaneous geometry/rules;
- geometry plus hysteretic finite-state machine;
- a small causal TCN or GRU;
- the same pose-token input with a size-matched causal decoder;
- ablations for RGB, visibility, timestamps, history length, and autoregressive output;
- persistence/constant-velocity future prediction;
- temperature scaling and a visibility-only abstention baseline;
- supervised/imitation policy before any RL policy.

Parameter-matched and compute-matched comparisons answer different questions; report both when architecture is the independent variable.

Model capacity and checkpoint pretraining are separate independent variables.
The current research search envelope extends to 350M total active parameters,
but a larger tier is not run merely because it is in the envelope. It must protect
a decision about capacity, transfer or teacher value and must be paired with data
whose independent subject/session/event support can test that decision. The
capacity-data protocol is `docs/SCALE_AND_DATA_AUTORESEARCH_2026.md`.

## 7. Promotion ladder

1. **Signal pilot**: 1–3 public or first-party recordings; exposes impossibility and instrumentation failures only.
2. **Learnability pilot**: subject-exclusive development corpus; at least three seeds for stochastic training comparisons.
3. **Internal replication**: held-out subjects, sessions, views, and devices; frozen metrics.
4. **External replication**: unseen site and target population; prospective capture.
5. **Deployment validation**: sustained on-device run with camera, pose, decoder, UI, and thermal load.
6. **Clinical/product validation**: intended-use protocol, human factors, adverse-event and subgroup analysis.

Results never automatically climb levels. A strong result at one level creates a hypothesis for the next.

## 8. First falsification queue

The immediate order is:

1. Measure temporal information loss at 5/7.5/10/15/30 Hz on real pose trajectories; use higher-rate synchronized capture for jumps and landings.
2. Label a small continuous set corpus with pauses, partial repetitions, and restarts; measure inter-rater boundary variance.
3. Compare FSM, causal TCN/GRU, and Nexus-style decoder under a fixed token/compute budget.
4. Add parallel heads one at a time and test negative transfer; do not infer head feasibility from parameter count.
5. Test event-triggered structured decoding versus JSON every frame, measuring exact-schema rate, event duplication, control latency, and context consumption.
6. Distill the best real-data teacher into 25M/10M/2M students and measure the full accuracy-calibration-latency frontier.
7. Introduce offline RL only for cue selection/timing after the perception/event interface is frozen and logged-policy support is sufficient.

The current sampling calculations and their limitations are recorded in `docs/POSE_NATIVE_CAPABILITY_AUDIT_2026.md`.
