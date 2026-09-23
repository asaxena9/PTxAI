# Known limitations

Relay is a credible technical MVP, not a validated rehabilitation device.

## Claims Relay cannot make

- Camera-derived angles, lift, extension, posture, and timing are estimates,
  not clinical measurements.
- A hard-limit warning cannot guarantee that an anatomical restriction was not
  crossed. Camera error and cue latency remain unmeasured on target phones.
- Relay cannot assess diagnosis, injury, tissue healing, pain, load, muscle
  activation, exercise appropriateness, or readiness to progress.
- Relay observes selected visible relationships. It cannot certify overall
  exercise form or replace a clinician.

## Instruction understanding

- The MVP uses a deterministic local parser, not a general language or vision
  model. It recognizes the three supported exercises and a bounded vocabulary
  of dose, side, hold, tempo, angle, target, minimum, maximum, range, and
  straight-knee instructions.
- Unfamiliar exercise names, complex prose, conditional instructions,
  handwriting, tables, or multiple interdependent restrictions may be missed.
- Photograph OCR and PDF text extraction can transpose or omit numbers and
  sides. The editable review is mandatory by design.
- Speech recognition depends on browser support and may not be local in every
  browser implementation. Typing remains the reliable fallback.
- The structured schema and evaluator can represent provider-defined body
  metrics, but an arbitrary joint or new exercise is not automatically
  observable. It still needs a feature extractor and validated setup/phase
  policy.

## Camera and body perception

- MediaPipe inferred-world landmarks are monocular model estimates, not phone
  depth-sensor measurements. Image/world agreement is corroboration between two
  outputs of one provider, not independent ground truth.
- Perspective, camera yaw, short projected limbs, occlusion, bedding, chairs,
  braces, loose clothing, lighting, body shape, skin tone, and unusual movement
  strategies can change or break estimates.
- Setup and calibration require at least 1.5 seconds of bounded feature spread,
  but they still assume that the user begins in a representative starting pose.
  A stable but poor starting position shifts relative lift, extension, and
  compensation features.
- Straight-leg-raise compensation is currently limited to broad visible knee
  bend and shoulder–hip change. Pelvic rotation, lumbar strategy, effort, and
  subtle lag are not reliably assessed.
- Seated extension is reported as progress from the calibrated seated start,
  not anatomical extension range. Straight-leg lift is a visible chain angle,
  not clinical hip flexion.
- The current phase thresholds and confidence gates were engineered and tested
  with synthetic replay. They have not been tuned against a representative
  participant dataset.
- Only one person and one selected side are tracked. The public-video benchmark
  exposed inactive-leg identity failures in MediaPipe Lite; robust identity
  continuity across real occlusion remains unvalidated even with Full.
- Camera setup fully checks the first exercise. Between exercises, Relay gives
  new placement guidance and recalibrates in the active flow, but it does not
  present a second dedicated setup rehearsal.

## Runtime and device coverage

- The live Web implementation calls MediaPipe video inference synchronously on
  the UI thread at about 15 Hz. A worker, adaptive model selection, thermal
  testing, and dropped-frame monitoring are not implemented.
- GPU is requested with CPU fallback, but no representative iPhone Safari or
  Android Chrome latency, power, heat, memory, or long-session study has been
  run.
- The temporal edge benchmark is a desktop Node/V8 microbenchmark and excludes
  the dominant camera and pose costs.
- Browser speech input, speech synthesis, Web Share, camera permission recovery,
  PWA installation, and service-worker behavior vary across platforms.
- First-use photograph OCR downloads Tesseract worker, core, and English
  language assets. Recognition runs in the browser and Relay does not
  intentionally upload the selected image, but that path is not first-use
  offline and executes remotely obtained code.
- The service worker helps only after assets have loaded successfully at least
  once. First-use offline operation is not supported.

## Temporal model

- Relay Temporal Synthetic v1 was trained and evaluated only on generated pose
  sequences. Its scores are not accuracy on people.
- Synthetic occlusion and noise do not reproduce real pose-SDK identity swaps,
  recovery, anatomy, devices, clothing, or home environments.
- Raw transition precision is too low for direct rep segmentation.
- In the current product applicable temporal output can corroborate only
  geometrically plausible hold and return transitions. It cannot start or
  complete a rep, establish the endpoint, or evaluate a constraint.
- The model expects a nominal 10 Hz cadence and 24-frame window. Large cadence
  changes alter its input distribution.

## Evaluation coverage

- The recorded benchmark contains three clips from two people performing a
  proxy movement, not Relay’s supported exercises.
- There is no anatomical angle ground truth, exact heel-slide 90° dataset,
  consented Relay participant set, held-out real participant split, or
  clinician-rated movement reference.
- Replay verifies software decisions on simulated landmarks. It cannot measure
  camera failure, demographic fairness, user comprehension, spoken-cue timing,
  or actual home setup.
- Automated browser tests use JSDOM for the product journey. Real-device
  camera, speech, sharing, offline, accessibility, and several-feet-away
  usability need dedicated testing.

## Privacy and data

- Relay application code does not persist or upload continuous camera video,
  but a formal device-network audit of the browser, hosting layer, and third
  party runtime has not yet been completed.
- Google states that MediaPipe image and video inputs remain on device, while
  its upstream [privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice)
  says the runtime may send performance and utilization metrics to Google.
  Relay has not yet independently verified or blocked that telemetry.
- Session history is ordinary browser `localStorage`, not encrypted storage.
  Anyone with access to the same browser profile may be able to read it.
- Clearing Relay history removes those structured records, not browser caches
  or a JSON export the user already downloaded/shared.
- Exported JSON contains the confirmed plan and movement results. The user must
  choose an appropriate destination.
- There is no account recovery, cross-device sync, organization policy,
  clinician access control, retention policy enforcement, or audit trail.

## Before a real-world release

At minimum:

1. Freeze the current pipeline and evaluate synchronized target-exercise
   recordings on representative phones with independent movement reference.
2. Measure failure and abstention by participant, environment, device, view,
   clothing, and occlusion—not only aggregate error.
3. Set restriction-warning margins from measured signed error and end-to-end
   cue latency.
4. Move live inference off the UI thread and run performance, thermal, and
   long-session tests.
5. Conduct usability/accessibility studies at actual phone placement distance.
6. Complete privacy, threat, dependency, and network reviews with explicit
   consent and retention rules for any human data.
7. Have clinical and regulatory experts review claims, intended use, and
   release controls.
