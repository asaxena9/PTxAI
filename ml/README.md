# Relay temporal movement intelligence

This workstream trains a small causal temporal model on pose-derived sequences,
compares its exported int8 form with an auditable kinematic baseline, and runs
the same artifact in the browser through a dependency-free TypeScript edge.

The current dataset is entirely synthetic. The result proves a reproducible
learned component and integration path; it does **not** establish accuracy on
people, phone-camera recordings, or rehabilitation movements.

## Reproduce

Use Python 3.11 and the recorded dependency versions:

```bash
python3 -m pip install -r ml/requirements.txt
python3 ml/run_experiment.py
```

Or run the stages separately:

```bash
python3 ml/prepare_data.py
python3 ml/train_temporal.py
python3 ml/evaluate_temporal.py
```

The prepared dataset and float checkpoint are disposable outputs under
`ml/.generated/`. The durable browser artifact is:

```text
public/models/temporal/relay-temporal-v1.json
```

Run edge tests and the desktop V8 microbenchmark:

```bash
node --experimental-strip-types --test ml/tests/temporal-model.test.mjs
node --experimental-strip-types ml/benchmark_edge.mjs --iterations 5000
```

## Product integration

```ts
const temporal = await loadTemporalMovementModel();

// At approximately 10 pose frames per second:
const evidence = temporal.observe(frame, side, prescription.exerciseId);

if (evidence.shouldApplyPhase) {
  // Feed the phase as evidence to the exercise-specific state machine.
}
```

Reset the model between exercises. Pass the confirmed exercise so phase
probability is conditioned on the user-confirmed instruction. Use
`state === "caution"` as supporting evidence only and
`state === "unavailable"` as an abstention.

The learned output must not replace:

- The current pose-quality and camera-position gate.
- Image/world knee geometry and its explicit uncertainty checks.
- Confirmed targets, ranges, and hard restrictions.
- Timestamp-based holds, tempo, set/rep bookkeeping, and hysteresis.
- The instruction to pause angle assessment when tracking is unreliable.

The model helps recognize temporal phase and detect an obvious mismatch between
expected and observed exercise. The checked-in engine remains the authority for
repetition completion and personalized constraints until representative human
recordings validate a broader learned role.

See `MODEL_CARD.md`, `EXPERIMENT_REPORT.md`, and `capture/README.md`.
