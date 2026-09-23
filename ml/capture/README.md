# Human capture and annotation scaffold

No consented human recordings were available for this MVP experiment. This
folder defines the workflow required to collect them without quietly treating
generated landmarks as people.

## Capture protocol

1. Obtain protocol-specific consent before recording. Use a pseudonymous
   participant ID; keep the re-identification key outside this repository.
2. Record each of the three exercises from the supported side view, then
   deliberately vary yaw, distance, lighting, clothing, speed, holds,
   incomplete returns, and short occlusions.
3. Include a synchronized angle reference for the heel-slide 90-degree study
   when the protocol supports it.
4. Export pose landmarks from the exact same provider/runtime contract used by
   Relay live sessions. Do not train directly from a separate landmark stack.
5. Annotate phases, complete/incomplete repetition intervals, holds, tracking
   quality, and personalized-constraint events.
6. Assign all recordings from one participant to exactly one of train,
   validation, or held-out. Held-out assignment happens before model tuning.

`record.schema.json` defines provenance and annotation metadata.
`landmarks.schema.json` defines the frame export. Copy `example-record.json`,
replace every placeholder, and store raw media under access control rather
than committing it. This repository ignores `ml/capture/raw/`.

Run the dependency-free integrity check:

```bash
python3 ml/capture/validate_records.py ml/capture/records/*.json
```

The checker verifies consent metadata, file hashes, monotonically increasing
timestamps, annotation intervals, and participant-level split isolation. A
future ingestion adapter should convert only validated landmark exports to the
same base features in `ml/relay_temporal/features.py`.
