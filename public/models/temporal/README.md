# Relay temporal model assets

`relay-temporal-v1.json` is the browser-deployed, int8-weight temporal
exercise/phase model. It was trained only on generated pose-landmark sequences.
Its metadata and limitations are part of the artifact and are validated at
runtime.

Do not replace this file without rerunning:

```bash
python3 ml/evaluate_temporal.py
node --experimental-strip-types --test ml/tests/temporal-model.test.mjs
node --experimental-strip-types ml/benchmark_edge.mjs --iterations 5000
```

See `ml/MODEL_CARD.md` for the evidence boundary.
