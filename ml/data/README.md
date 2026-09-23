# Relay motion-data foundation

The checked-in experiment has **no real-human recordings**. It generates
pose-derived landmark sequences with disjoint virtual-subject train,
validation, and held-out splits. That makes the pipeline reproducible and
useful for software regression, but it is not empirical evidence about people.

Generate the disposable archive:

```bash
python3 ml/prepare_data.py
```

The archive is written to `ml/.generated/` and intentionally ignored by Git.
Its adjacent manifest records frame, sequence, subject, split, feature, and
generator details. The durable provenance declaration is
`ml/data/provenance.json`.

## Replacing synthetic data

Consented recordings should follow the schemas in `ml/capture/`. Extract pose
landmarks through the same provider contract used by live Relay, retain source
video only in the access-controlled capture environment, and export
pseudonymous landmark/annotation records. Split by participant—not clip—and
run `python3 ml/capture/validate_records.py <records...>` before training.

Real and synthetic test results must remain separate. Never merge synthetic
clips into a reported “participant” count.
