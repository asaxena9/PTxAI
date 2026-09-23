#!/usr/bin/env node

/**
 * Synthetic geometry sensitivity analysis for Relay.
 *
 * This evaluates projection and landmark-noise behavior only. It does not
 * evaluate MediaPipe, people, phone cameras, or anatomical accuracy.
 */

const DEG = 180 / Math.PI;
const RAD = Math.PI / 180;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function angle2d(a, vertex, c) {
  const av = [a[0] - vertex[0], a[1] - vertex[1]];
  const cv = [c[0] - vertex[0], c[1] - vertex[1]];
  const denominator = Math.hypot(...av) * Math.hypot(...cv);
  return (
    Math.acos(
      clamp((av[0] * cv[0] + av[1] * cv[1]) / denominator, -1, 1),
    ) * DEG
  );
}

function flexion2d(hip, knee, ankle) {
  return 180 - angle2d(hip, knee, ankle);
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function round(value, digits = 3) {
  return Number(value.toFixed(digits));
}

function makeHinge(flexion, orientationDeg, length = 200) {
  const orientation = orientationDeg * RAD;
  const knee = [0, 0, 0];
  const hipAngle = orientation + Math.PI;
  const ankleAngle = orientation + flexion * RAD;
  return {
    hip: [
      Math.cos(hipAngle) * length,
      Math.sin(hipAngle) * length,
      0,
    ],
    knee,
    ankle: [
      Math.cos(ankleAngle) * length,
      Math.sin(ankleAngle) * length,
      0,
    ],
  };
}

function rotateYaw(point, yawDeg) {
  const yaw = yawDeg * RAD;
  return [
    point[0] * Math.cos(yaw) + point[2] * Math.sin(yaw),
    point[1],
    -point[0] * Math.sin(yaw) + point[2] * Math.cos(yaw),
  ];
}

function aspectRatioExperiment() {
  const width = 1280;
  const height = 720;
  const rows = [];
  for (const flexion of [30, 60, 90, 120]) {
    for (const orientation of [0, 30, 60, 90]) {
      const hinge = makeHinge(flexion, orientation);
      const translateAndNormalize = (point) => [
        (point[0] + width / 2) / width,
        (point[1] + height / 2) / height,
      ];
      const normalizedEstimate = flexion2d(
        translateAndNormalize(hinge.hip),
        translateAndNormalize(hinge.knee),
        translateAndNormalize(hinge.ankle),
      );
      const correctedEstimate = flexion2d(
        [
          translateAndNormalize(hinge.hip)[0] * width,
          translateAndNormalize(hinge.hip)[1] * height,
        ],
        [
          translateAndNormalize(hinge.knee)[0] * width,
          translateAndNormalize(hinge.knee)[1] * height,
        ],
        [
          translateAndNormalize(hinge.ankle)[0] * width,
          translateAndNormalize(hinge.ankle)[1] * height,
        ],
      );
      rows.push({
        trueFlexionDeg: flexion,
        limbOrientationDeg: orientation,
        normalizedCoordinateEstimateDeg: round(normalizedEstimate),
        normalizedCoordinateErrorDeg: round(normalizedEstimate - flexion),
        aspectCorrectedEstimateDeg: round(correctedEstimate),
        aspectCorrectedErrorDeg: round(correctedEstimate - flexion),
      });
    }
  }
  const wrongErrors = rows.map((row) =>
    Math.abs(row.normalizedCoordinateErrorDeg),
  );
  const correctedErrors = rows.map((row) =>
    Math.abs(row.aspectCorrectedErrorDeg),
  );
  const worstCase = [...rows].sort(
    (a, b) =>
      Math.abs(b.normalizedCoordinateErrorDeg) -
      Math.abs(a.normalizedCoordinateErrorDeg),
  )[0];
  return {
    frame: { width, height, aspectRatio: round(width / height) },
    casesEvaluated: rows.length,
    representativeCases: rows.filter(
      (row) => row.trueFlexionDeg === 90,
    ),
    worstCase,
    summary: {
      normalizedCoordinatesMedianAbsoluteErrorDeg: round(
        percentile(wrongErrors, 0.5),
      ),
      normalizedCoordinatesMaximumAbsoluteErrorDeg: round(
        Math.max(...wrongErrors),
      ),
      aspectCorrectedMaximumAbsoluteErrorDeg: round(
        Math.max(...correctedErrors),
      ),
    },
  };
}

function yawProjectionExperiment() {
  const rows = [];
  for (const yaw of [0, 15, 30, 45]) {
    for (const flexion of [30, 60, 90, 120]) {
      for (const orientation of [-60, -30, 0, 30, 60]) {
        const hinge = makeHinge(flexion, orientation);
        const project = (point) => {
          const rotated = rotateYaw(point, yaw);
          return [rotated[0], rotated[1]];
        };
        const estimate = flexion2d(
          project(hinge.hip),
          project(hinge.knee),
          project(hinge.ankle),
        );
        rows.push({
          yawDeg: yaw,
          trueFlexionDeg: flexion,
          limbOrientationDeg: orientation,
          projectedEstimateDeg: round(estimate),
          errorDeg: round(estimate - flexion),
        });
      }
    }
  }
  const byYaw = Object.fromEntries(
    [0, 15, 30, 45].map((yaw) => {
      const errors = rows
        .filter((row) => row.yawDeg === yaw)
        .map((row) => Math.abs(row.errorDeg));
      return [
        yaw,
        {
          medianAbsoluteErrorDeg: round(percentile(errors, 0.5)),
          p95AbsoluteErrorDeg: round(percentile(errors, 0.95)),
          maximumAbsoluteErrorDeg: round(Math.max(...errors)),
        },
      ];
    }),
  );
  return {
    casesEvaluated: rows.length,
    summaryByYawDeg: byYaw,
    worstCaseByYawDeg: Object.fromEntries(
      [0, 15, 30, 45].map((yaw) => [
        yaw,
        [...rows]
          .filter((row) => row.yawDeg === yaw)
          .sort(
            (a, b) => Math.abs(b.errorDeg) - Math.abs(a.errorDeg),
          )[0],
      ]),
    ),
  };
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

function gaussian(random) {
  const u1 = Math.max(random(), Number.EPSILON);
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function pixelNoiseExperiment() {
  const sigmaPx = 2;
  const trials = 10000;
  const seed = 20260727;
  const random = seededRandom(seed);
  const rows = [];
  for (const lengthPx of [50, 75, 100, 150]) {
    const errors = [];
    for (let trial = 0; trial < trials; trial += 1) {
      const noisy = (point) => [
        point[0] + gaussian(random) * sigmaPx,
        point[1] + gaussian(random) * sigmaPx,
      ];
      const estimate = flexion2d(
        noisy([-lengthPx, 0]),
        noisy([0, 0]),
        noisy([0, lengthPx]),
      );
      errors.push(estimate - 90);
    }
    const mean = errors.reduce((sum, value) => sum + value, 0) / trials;
    const variance =
      errors.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
      (trials - 1);
    rows.push({
      segmentLengthPx: lengthPx,
      meanErrorDeg: round(mean),
      standardDeviationDeg: round(Math.sqrt(variance)),
      p95AbsoluteErrorDeg: round(
        percentile(errors.map((value) => Math.abs(value)), 0.95),
      ),
    });
  }
  return { seed, sigmaPx, trialsPerLength: trials, trueFlexionDeg: 90, rows };
}

const result = {
  experiment: "relay-angle-geometry-sensitivity",
  version: 1,
  evidenceBoundary:
    "Synthetic projection and landmark-noise sensitivity only; not pose-model or anatomical accuracy.",
  aspectRatio: aspectRatioExperiment(),
  outOfPlaneYaw: yawProjectionExperiment(),
  pixelLandmarkNoise: pixelNoiseExperiment(),
};

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
