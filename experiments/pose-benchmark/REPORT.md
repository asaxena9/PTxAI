# Relay pose-provider benchmark

Run date: 2026-07-27

## Recommendation

Use **MediaPipe Pose Landmarker Full** as Relay's default MVP provider.

Use **MediaPipe Lite only as an explicit low-performance-device fallback**
after a setup benchmark shows that Full cannot sustain Relay's required update
rate. Do not switch providers during a repetition.

Do not use MoveNet Lightning as the initial horizontal knee-rehabilitation
provider. It was substantially faster on this host, but it produced a complete
active hip-knee-ankle chain much less often on the lying exercise proxy and
lacks the heel, foot-index, visibility/presence, and world-coordinate outputs
that Relay can use for camera setup and confidence corroboration.

This benchmark supports a provider choice. It does **not** establish anatomical
knee-angle accuracy or validate Relay's 90-degree restriction behavior.

## Public sample provenance and scope

The common input is a deliberately small subset of
[MobiPhysio](https://doi.org/10.7910/DVN/XSI0QN), a smartphone-recorded
physiotherapy video dataset. Harvard Dataverse reports the release under
**CC0 1.0**. The associated article is
[doi:10.1016/j.dib.2026.112635](https://doi.org/10.1016/j.dib.2026.112635).

The three exact files, Dataverse IDs, byte sizes, labels, and SHA-256 hashes are
recorded in `provenance.json` and verified by `download_assets.py`.

| Sample | View / condition | Duration | Sampled frames |
|---|---|---:|---:|
| `E07_P02_AF_VFL_GM.mp4` | P02, front, full light | 37.16 s | 372 |
| `E07_P02_AR_VFL_GM.mp4` | P02, right, full light | 37.96 s | 382 |
| `E07_P17_AR_VML_GM.mp4` | P17, right, medium light | 27.24 s | 274 |

All are portrait 1080×1920 at approximately 30 FPS. The benchmark sampled
every third frame, for 1,028 common timestamps at approximately 10 Hz.

MobiPhysio E07 is **supine hip flexion with the knee bent and drawn toward the
torso**. It is useful for horizontal-body pose detection, large knee excursions,
leg overlap, occlusion, and side identity. It is not a heel slide, straight-leg
raise, or seated knee extension. Results must therefore be treated as a proxy.

UL-RED was investigated first because it includes `SeatedLegStretch` and
`LegExtensionForward`. Its public `video/` files are rendered composites of
marker skeleton, marker-less skeleton, and depth imagery rather than ordinary
RGB recordings of the person. Feeding those renderings to RGB pose models
would not be a valid phone-camera comparison, so UL-RED was not benchmarked.
Its catalogue remains useful for future skeleton/ground-truth evaluation:
[UL-RED, CC BY 4.0](https://datacat.liverpool.ac.uk/2729/).

## Models

| Provider | Asset | Output relevant to Relay |
|---|---:|---|
| MediaPipe Lite | 5,777,746 B | 33 image landmarks, visibility/presence, inferred world landmarks, heel and foot index |
| MediaPipe Full | 9,398,198 B | Same contract as Lite; higher-capacity landmark model |
| MoveNet Lightning float16 v4 | 4,758,512 B | 17 image landmarks and per-keypoint scores; no heel/foot index or world landmarks |

Exact model URLs and SHA-256 hashes are in `provenance.json`.

The measured implementations were:

- MediaPipe Python Tasks 0.10.31 in video mode, one pose, default detector and
  tracker, detection/presence/tracking thresholds of 0.5.
- MoveNet SinglePose Lightning float16 TFLite v4 through TensorFlow 2.20,
  XNNPACK, four interpreter threads.
- MoveNet used a 192×192 square input. The first frame was full-frame padded;
  subsequent frames used a confidence-gated, hip-centered tracked crop.

The MoveNet crop and inverse-coordinate mapping are part of measured
preprocessing. Video decoding is excluded from pose latency.

## Host and method

Measurements were made on a 10-core Apple M2 Pro MacBook Pro with 16 GB RAM,
not on a phone. They compare the models under one controlled host; they do not
predict iPhone Safari or Android Chrome latency.

- The first ten sampled frames of each clip were excluded from latency.
- Total pose time includes RGB/model-image conversion or MoveNet cropping plus
  inference.
- Sequential throughput is evaluated pose frames divided by accumulated pose
  processing time; it excludes decode, rendering, application logic, and
  speech.
- A leg chain is "available" when hip, knee, and ankle are inside a 5% extended
  frame boundary and each provider's raw joint score meets the threshold.
- Raw scores are not calibrated and are not directly comparable between
  providers. Coverage is therefore reported across several thresholds.
- Knee flexion proxy is `180° - angle(hip, knee, ankle)` in aspect-correct
  source-pixel coordinates.
- Low-motion transitions are the bottom quartile of model-independent
  thumbnail frame change within each clip.
- One active/more observable leg was visually selected per clip and then held
  constant for every model. This avoids rewarding a provider for switching to
  the overlapped, inactive leg.

## Actual results

Primary result at a raw chain threshold of 0.5:

| Metric | MediaPipe Lite | MediaPipe Full | MoveNet Lightning |
|---|---:|---:|---:|
| Active-chain coverage | **100.0%** | **100.0%** | 28.4% |
| Median raw chain score | 0.966 | 0.953 | 0.405 |
| Pose time p50 | 13.65 ms | 19.31 ms | **5.40 ms** |
| Pose time p95 | 30.09 ms | 30.36 ms | **11.19 ms** |
| Derived sequential throughput | 62.2 FPS | 47.1 FPS | **149.8 FPS** |
| Isolated runner creation p50 | 289 ms | 292 ms | **5 ms** |
| Low-motion angle step p50 | 1.56° | **1.32°** | 2.17° |
| Low-motion angle step p95 | 10.70° | **8.23°** | 12.65° |
| Median angle second difference | 6.93° | **4.07°** | 4.10° |

Runner creation excludes Python imports, model download, browser/WASM startup,
and application initialization. It is not page-start time.

MoveNet active-chain coverage by raw threshold was:

| Raw chain threshold | Coverage |
|---:|---:|
| 0.2 | 87.5% |
| 0.3 | 68.8% |
| 0.4 | 51.2% |
| 0.5 | 28.4% |
| 0.6 | 5.6% |
| 0.7 | 0.2% |

Both MediaPipe variants remained at 100% through 0.7 on the selected active
chains. This does not mean a MediaPipe score of 0.7 is an accuracy probability;
it only describes provider output behavior on these clips.

### Angle consistency and identity failures

Pairwise image-angle agreement, on frames where both providers passed the 0.5
chain gate:

| Pair | Common samples | Median absolute difference | p95 absolute difference |
|---|---:|---:|---:|
| Full vs Lite | 1,028 | 9.42° | 68.72° |
| Full vs MoveNet | 292 | 3.72° | 17.33° |
| Lite vs MoveNet | 292 | 7.45° | 63.96° |

The Full–MoveNet result is conditioned on MoveNet's limited high-confidence
coverage and therefore has strong selection bias.

Qualitative review of the largest common-frame disagreements found that
MediaPipe Lite sometimes followed the straight, non-active leg while retaining
high raw confidence. For example:

- P02 front, frame 534: Full 96.6°, MoveNet 94.4°, Lite 11.1°; Lite's raw chain
  score was 0.966.
- P02 right, frame 783: Full 95.7°, MoveNet 111.4°, Lite 17.6°; Lite's raw chain
  score was 0.815.

The annotated evidence is in
`results/2026-07-27-m2-pro/disagreement_examples.jpg`. These are visible
identity/side failures, not ground-truth angle comparisons. They reinforce the
need for Relay's guided selected-leg movement, identity continuity, and
angle/displacement agreement checks.

Full had lower low-motion jitter and substantially lower temporal roughness
than Lite on the common active chain. Bone-length consistency was mixed:
Full's thigh-length coefficient of variation was lower, while its shank value
was higher. No accuracy conclusion is drawn from either.

## Input-orientation experiment

A second pass rotated the P02 right-view clip 90° counterclockwise before
inference so the lying person's head appeared at the top of the model input.

| Provider | Native active-chain coverage | Head-up rotated coverage |
|---|---:|---:|
| MediaPipe Lite | 100.0% | 100.0% |
| MediaPipe Full | 100.0% | 100.0% |
| MoveNet Lightning | 27.7% | **2.4%** |

On this clip, Full's low-motion angle-step p95 changed from 9.98° to 12.30°.
Lite changed from 22.44° to 16.87°, but the rotation did not resolve its
active-leg identity behavior. MoveNet had no consecutive low-motion transitions
passing the 0.5 chain gate after rotation.

Do not force a generic "make the person upright" transform. Keep the native
camera orientation for the MVP and only introduce provider/exercise-specific
rotation after broader evidence.

## Deployment complexity

### MediaPipe Lite / Full

- Same downstream contract and nearly identical integration; selecting Full
  adds about 3.62 MB of model asset.
- Built-in person detection, crop tracking, video mode, and normalized/world
  landmark output.
- 33 points include heel and foot index, useful for Relay's heel-slide
  displacement and camera-placement checks.
- The web Tasks call is synchronous and should run off the UI thread.
- The model/provider output still needs Relay-specific side identity,
  confidence, geometry, and temporal logic.

### MoveNet Lightning

- Smallest and fastest measured model.
- Requires explicit crop management, crop reset behavior, inverse-coordinate
  mapping, and provider-specific score calibration.
- A web product adds TensorFlow.js model/runtime/backend deployment, or a native
  product adds LiteRT integration. The TensorFlow 2.20 Python interpreter used
  here is already marked deprecated in favor of LiteRT.
- Only 17 points: no heel or foot-index landmark and no world-coordinate
  corroboration.
- Low active-chain coverage on these horizontal proxies is the blocking result,
  not raw speed.

## Limitations

- Three clips and two people are enough to expose failure modes, not to estimate
  population performance.
- The exercise is a proxy, not one of Relay's exact supported movements.
- There is no synchronized anatomical angle ground truth. Stability,
  plausibility, and cross-model agreement are not accuracy.
- Clothing, body shape, skin tone, camera distance, phones, browsers, bedding,
  chairs, braces, and ordinary-home backgrounds are underrepresented.
- MediaPipe video mode includes internal detection/tracking behavior; MoveNet's
  lightweight crop tracker is benchmark-specific. The comparison reflects
  deployable pipelines, not isolated neural-network kernels.
- Provider score distributions differ; threshold curves are descriptive rather
  than calibrated confidence.
- Host latency is not phone latency.
- The active leg was selected by visual review. Relay must perform this
  interactively and retain temporal identity.
- An exact 90° camera estimate cannot be released as reliable from this
  benchmark.

## Next experiment

Record synchronized right heel slides around 70–100° on representative iPhones
and Android phones, with an electrogoniometer or optical motion-capture
reference, deliberate camera yaw, clothing variation, and controlled
occlusion. Run the exact browser providers and Relay confidence gate. The key
outcomes are signed angle error, peak error, late maximum warnings,
high-confidence identity failures, abstention rate, and end-to-end cue latency.
