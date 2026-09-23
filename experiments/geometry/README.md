# Relay geometry sensitivity experiment

This deterministic experiment quantifies three implementation risks:

1. treating normalized `x` and `y` coordinates as though they have the same
   scale in a 16:9 image;
2. projecting an out-of-plane joint hinge into a single camera plane; and
3. adding independent two-pixel landmark noise at different projected limb
   lengths.

Run it from the repository root:

```bash
node experiments/geometry/angle-sensitivity.mjs \
  > experiments/geometry/results.json
```

The committed [`results.json`](./results.json) is the output of that command.
The random experiment uses a fixed seed (`20260727`).

## Evidence boundary

These are idealized synthetic geometry tests. They support implementation
choices such as aspect-correct pixel coordinates, side-on camera guidance, and
minimum projected segment length. They do **not** measure MediaPipe accuracy,
human anatomy, phone-camera accuracy, or safety around a configured movement
restriction.
