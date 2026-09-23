# Temporal sampling audit: measured pilot

Run date: 2026-08-07

## Outcome

The preregistered hypothesis was **not rejected** for either 10 or 15 Hz on
these three slow MobiPhysio E07 hip-flexion recordings.

| Independent pose rate | Aggregate crossing error p95 vs 30 Hz | Matched reference crossings | Missed | Extra | Disposition |
|---:|---:|---:|---:|---:|---|
| 10 Hz | 29.52 ms | 57/57 | 0 | 0 | keep for replication |
| 15 Hz | 16.05 ms | 57/57 | 0 | 0 | keep for replication |

Per-clip 10 Hz angle reconstruction MAE was 1.01°, 3.00°, and 1.78° relative
to the 30 Hz pose stream. Per-clip p95 angular-velocity error was 29.64,
69.06, and 42.90°/s. The latter shows that preserving a mid-range event count
does not mean preserving motion dynamics.

## What is established

`MEASURED-PILOT`: under the exact pipeline, clips, smoothing, active-side rule,
and midpoint-crossing detector recorded in the manifest, the separately run
10 and 15 Hz pose streams produced the reported differences relative to the
separately run 30 Hz pose stream.

## What is not established

- The 30 Hz stream is MediaPipe output, not synchronized mocap or force truth.
- Centered 200 ms smoothing is non-causal and is unsuitable as a live latency measurement.
- The clips contain slow hip flexion, not a squat, jump takeoff, flight, or landing transient.
- Three recordings from two people cannot estimate population performance.
- This does not test any learned temporal decoder, structured outputs, future motion, uncertainty, or intervention policy.
- It does not establish phone, ESP32, or sustained thermal performance.

The next sampling experiment for jumps requires 120–240 FPS video synchronized
with a contact mat or force plate, with pose evaluated at 30/60 FPS and event
truth evaluated at the sensor rate. Until that is run, 10 Hz jump/landing
support remains `UNKNOWN`, not supported by this pilot.

Raw artifacts are in `results/2026-08-07-pilot/`; independent pose streams are
in `runs/pose-15hz/` and `runs/pose-30hz/`. Dataset and model provenance remain
in `../pose-benchmark/provenance.json`.
