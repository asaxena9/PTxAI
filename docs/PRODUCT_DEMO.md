# Guided product demonstration

## Start

```bash
npm install
npm run dev
```

Open the local address printed by the server in a current browser.

## Demonstration 1: complete three-exercise journey

1. On the home screen choose **Try the guided demo**.
2. On **Check what Relay understood**, inspect all three editable exercises:
   - right heel slide: 1 set × 3 reps, 2-second hold, 80° target, hard 90°
     maximum;
   - right straight-leg raise: 1 × 3, 2-second hold, visible knee-straightness
     and shoulder–hip stability rules;
   - right seated knee extension: 1 × 3, 2-second hold, at least 85% extension
     progress.
3. Expand an exercise and change a set, rep, hold, side, target, or maximum to
   demonstrate that the interpretation is user-controlled. Restore the demo
   value if you want the shortest path.
4. Select **Confirm & set up camera**.
5. Confirm that **Guided replay** is selected and the scenario is **Clean
   repetitions**. Confirm the supported phone position and selected side, then
   select **Run camera check**. The start button remains disabled until good
   geometry is stable for at least 1.5 seconds.
6. Select **Start session**. Watch calibration, phase status, rep/hold counters,
   estimated exercise-specific feature, visible/spoken coach message, and
   tracking state.
7. At each exercise transition, read the new positioning instruction and
   continue. Relay recalibrates before counting the next exercise.
8. Open the session summary. Inspect completed/planned reps, hold completion,
   tracking coverage, the exercise-specific peak/range, targets/restrictions,
   and broad posture observations.
9. Choose **Export or share**. If native file sharing is unavailable, Relay
   downloads a JSON export. Then open **View local history**, reopen the result,
   delete it, or clear all records.

The guided replay is labeled throughout as simulated landmarks. It uses the
production measurement, confidence, constraint, coaching, progression, and
summary pipeline; it does not demonstrate camera accuracy.

## Demonstration 2: personalized 90° maximum

1. Return home and choose **Load the 90° limit proof**, or paste:

   > Right heel slides, three sets of ten, hold for two seconds, and do not bend
   > my knee beyond 90 degrees.

2. Select **Create my session**.
3. Confirm that Relay interpreted:
   - right heel slide;
   - 3 sets × 10 reps;
   - 2-second hold;
   - a hard hip–knee–ankle `knee-flexion` maximum of 90°, not a target.
4. Confirm the session, switch setup from **Live camera** to **Guided replay**,
   and choose **Approach maximum**.
5. Run the check and start. Relay should issue an approaching-maximum event
   without a reached-maximum event.
6. Repeat with **Reach maximum**. Relay should say that its estimate reached the
   configured limit and tell the user to stop moving farther.
7. Repeat with **Uncertain angle**. Image and inferred-world geometry disagree
   during the fault window, so angle-based rules pause rather than creating a
   limit event from unreliable geometry.

The replay reaches deliberately constructed values. It proves the personalized
constraint behavior, not the real-camera accuracy or safety of a 90° warning.

## Demonstration 3: live camera

Use an ordinary exercise plan that is already considered appropriate for the
person; Relay does not make that decision.

1. Type or speak supported instructions, review them, and keep **Live camera**
   selected.
2. Secure the phone in landscape about 6–8 feet away, near the moving joints.
   Choose the front camera to see Relay’s coaching or the rear camera for a
   usually sharper view. Turn the selected side toward the camera.
3. For heel slides and seated extension keep the selected hip, knee, and ankle
   visible. For straight-leg raises also keep the shoulder–hip relationship
   visible.
4. Confirm that the phone is supported and that the prescribed side faces the
   camera. Allow camera access and choose **Run camera check**. Follow the
   specific repositioning reason, then hold still until the status remains
   **Good · body features active** for at least 1.5 seconds.
5. Test audio, choose sound on/off, and start. Hold the beginning pose while
   Relay calibrates.
6. Move only under the original plan. Observe that covering a required joint,
   leaving the frame, becoming too small, turning away from side-on, or causing
   strong image/world disagreement pauses the relevant assessment.
7. Use pause/resume and confirm that paused or unobserved time is not credited
   to the hold. Stop early to see a partial summary.

The live path runs the locally served MediaPipe Full model and Relay’s temporal
edge in the browser. It stores structured results, not continuous video.

## Other creation paths

- **Speak**: browser speech recognition fills the same editable instruction
  composer. Unsupported browsers show a typing fallback.
- **Upload plan**: accepts text/Markdown, PDF, or an image. PDF extraction and
  English OCR run in browser. Photograph OCR downloads its Tesseract runtime
  and English language assets on first use; Relay does not intentionally
  upload the selected image. Review every number, side, and restriction.
- **Build it manually**: opens all three exercises with editable side, dose,
  hold, tempo, targets, and maxima.
- **Ambiguity check**: enter
  `Right heel slides, 3 x 10, hold 2 seconds, to 90 degrees.` Relay asks whether
  90° is a target, hard maximum, or both.
- **Multi-exercise text**: enter
  `Right straight-leg raises, two sets of eight, hold for two seconds, keep the
  knee straight. Then right LAQ, 2 x 10.` Relay creates two editable exercises
  with different body-feature scopes.

## Reviewer checklist

- No account is required.
- The user can correct the structured plan before camera access.
- A target and hard maximum are visually and behaviorally distinct.
- Each exercise shows a different primary body feature.
- Bad confidence pauses assessment and gives a repositioning reason.
- Spoken guidance matches the visible coach message.
- Pause, stop, sets, and exercise transitions are operable at mobile size.
- Replay is never presented as camera footage or human-performance evidence.
- Summary, history, deletion, and export contain structured data only.
