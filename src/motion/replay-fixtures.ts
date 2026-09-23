import type { BodySide, ExerciseId } from "../domain/types";
import type {
  JointName,
  LandmarkSet,
  PoseFrame,
  PoseLandmark,
} from "./types";

export type ReplayScenarioId =
  | "correct"
  | "incomplete-range"
  | "incomplete-return"
  | "fast"
  | "slow"
  | "tracking-loss"
  | "occlusion"
  | "poor-camera"
  | "posture-variation"
  | "approach-limit"
  | "exceed-limit"
  | "uncertain-angle";

export interface ReplayFixtureOptions {
  exerciseId: ExerciseId;
  side: BodySide;
  scenario?: ReplayScenarioId;
  reps?: number;
  holdSeconds?: number;
  intervalMs?: number;
}

function landmark(
  x: number,
  y: number,
  z = 0,
  confidence = 0.96,
): PoseLandmark {
  return {
    x,
    y,
    z,
    visibility: confidence,
    presence: confidence,
  };
}

function withSide(
  side: BodySide,
  values: Record<"shoulder" | "hip" | "knee" | "ankle" | "heel", PoseLandmark>,
): LandmarkSet {
  return Object.fromEntries(
    Object.entries(values).map(([key, value]) => [
      `${side}-${key}` as JointName,
      value,
    ]),
  ) as LandmarkSet;
}

function withBodyScaffold(
  selected: LandmarkSet,
  side: BodySide,
  posture: "lying" | "seated",
  confidence: number,
): LandmarkSet {
  const oppositeSide: BodySide = side === "left" ? "right" : "left";
  const shoulder = selected[`${side}-shoulder`];
  const hip = selected[`${side}-hip`];
  const knee = selected[`${side}-knee`];
  const ankle = selected[`${side}-ankle`];
  if (!shoulder || !hip || !knee || !ankle) return selected;

  const scaffoldConfidence = Math.min(confidence, 0.78);
  const add = (
    set: LandmarkSet,
    name: JointName,
    x: number,
    y: number,
  ) => {
    if (!set[name]) set[name] = landmark(x, y, 0, scaffoldConfidence);
  };
  const result: LandmarkSet = { ...selected };
  const sideOffset = posture === "seated" ? -0.035 : 0.042;
  const otherShoulder = {
    x: shoulder.x + (posture === "seated" ? sideOffset : 0.004),
    y: shoulder.y + (posture === "seated" ? 0.008 : sideOffset),
  };
  const otherHip = {
    x: hip.x + (posture === "seated" ? sideOffset : 0.004),
    y: hip.y + (posture === "seated" ? 0.008 : sideOffset),
  };
  const otherKnee = {
    x: knee.x + (posture === "seated" ? sideOffset : 0.004),
    y: knee.y + (posture === "seated" ? 0.022 : sideOffset),
  };
  const otherAnkle = {
    x: ankle.x + (posture === "seated" ? sideOffset : 0.004),
    y: ankle.y + (posture === "seated" ? 0.022 : sideOffset),
  };

  add(
    result,
    `${oppositeSide}-shoulder`,
    otherShoulder.x,
    otherShoulder.y,
  );
  add(result, `${oppositeSide}-hip`, otherHip.x, otherHip.y);
  add(result, `${oppositeSide}-knee`, otherKnee.x, otherKnee.y);
  add(result, `${oppositeSide}-ankle`, otherAnkle.x, otherAnkle.y);
  add(
    result,
    `${oppositeSide}-heel`,
    otherAnkle.x - 0.018,
    otherAnkle.y + 0.008,
  );
  add(
    result,
    `${oppositeSide}-foot-index`,
    otherAnkle.x + 0.035,
    otherAnkle.y + 0.01,
  );
  add(
    result,
    `${side}-foot-index`,
    ankle.x + 0.035,
    ankle.y + 0.01,
  );

  if (posture === "seated") {
    add(result, `${side}-elbow`, shoulder.x + 0.09, shoulder.y + 0.1);
    add(result, `${side}-wrist`, shoulder.x + 0.075, shoulder.y + 0.21);
    add(
      result,
      `${oppositeSide}-elbow`,
      otherShoulder.x - 0.07,
      otherShoulder.y + 0.11,
    );
    add(
      result,
      `${oppositeSide}-wrist`,
      otherShoulder.x - 0.045,
      otherShoulder.y + 0.21,
    );
  } else {
    add(result, `${side}-elbow`, shoulder.x + 0.055, shoulder.y - 0.105);
    add(result, `${side}-wrist`, shoulder.x + 0.145, shoulder.y - 0.12);
    add(
      result,
      `${oppositeSide}-elbow`,
      otherShoulder.x + 0.055,
      otherShoulder.y + 0.105,
    );
    add(
      result,
      `${oppositeSide}-wrist`,
      otherShoulder.x + 0.145,
      otherShoulder.y + 0.12,
    );
  }

  ([side, oppositeSide] as BodySide[]).forEach((bodySide) => {
    const wrist = result[`${bodySide}-wrist`];
    if (!wrist) return;
    add(result, `${bodySide}-pinky`, wrist.x + 0.026, wrist.y + 0.008);
    add(result, `${bodySide}-index`, wrist.x + 0.03, wrist.y);
    add(result, `${bodySide}-thumb`, wrist.x + 0.018, wrist.y - 0.014);
  });

  const head =
    posture === "seated"
      ? { x: (shoulder.x + otherShoulder.x) / 2, y: shoulder.y - 0.095 }
      : { x: shoulder.x - 0.045, y: shoulder.y - 0.008 };
  add(result, "nose", head.x, head.y);
  add(result, "left-eye-inner", head.x - 0.012, head.y - 0.012);
  add(result, "left-eye", head.x - 0.02, head.y - 0.014);
  add(result, "left-eye-outer", head.x - 0.029, head.y - 0.012);
  add(result, "right-eye-inner", head.x + 0.012, head.y - 0.012);
  add(result, "right-eye", head.x + 0.02, head.y - 0.014);
  add(result, "right-eye-outer", head.x + 0.029, head.y - 0.012);
  add(result, "left-ear", head.x - 0.042, head.y - 0.002);
  add(result, "right-ear", head.x + 0.042, head.y - 0.002);
  add(result, "mouth-left", head.x - 0.014, head.y + 0.022);
  add(result, "mouth-right", head.x + 0.014, head.y + 0.022);
  return result;
}

function heelOrSeatedPose(
  flexion: number,
  side: BodySide,
  seated: boolean,
  confidence = 0.96,
  worldFlexion = flexion,
): { image: LandmarkSet; world: LandmarkSet } {
  const width = 1280;
  const height = 720;
  const hip = seated ? { x: 0.25, y: 0.46 } : { x: 0.22, y: 0.48 };
  const knee = seated ? { x: 0.5, y: 0.46 } : { x: 0.5, y: 0.48 };
  const lengthPx = seated ? 210 : 260;
  const worldLength = seated ? 0.26 : 0.3;
  const imageRadians = (flexion * Math.PI) / 180;
  const worldRadians = (worldFlexion * Math.PI) / 180;
  const imageAnkle = {
    x: knee.x + (Math.cos(imageRadians) * lengthPx) / width,
    y: knee.y + (Math.sin(imageRadians) * lengthPx) / height,
  };
  const worldAnkle = {
    x: knee.x + Math.cos(worldRadians) * worldLength,
    y: knee.y + Math.sin(worldRadians) * worldLength,
  };
  const shoulder = seated
    ? { x: hip.x - 0.02, y: hip.y - 0.24 }
    : { x: hip.x - 0.13, y: hip.y };

  const image = withSide(side, {
    shoulder: landmark(shoulder.x, shoulder.y, 0, confidence),
    hip: landmark(hip.x, hip.y, 0, confidence),
    knee: landmark(knee.x, knee.y, 0, confidence),
    ankle: landmark(imageAnkle.x, imageAnkle.y, 0, confidence),
    heel: landmark(imageAnkle.x - 0.02, imageAnkle.y, 0, confidence),
  });
  return {
    image: withBodyScaffold(
      image,
      side,
      seated ? "seated" : "lying",
      confidence,
    ),
    world: withSide(side, {
      shoulder: landmark(shoulder.x, shoulder.y, 0, confidence),
      hip: landmark(hip.x, hip.y, 0, confidence),
      knee: landmark(knee.x, knee.y, 0, confidence),
      ankle: landmark(worldAnkle.x, worldAnkle.y, 0, confidence),
      heel: landmark(worldAnkle.x - 0.02, worldAnkle.y, 0, confidence),
    }),
  };
}

function straightLegRaisePose(
  lift: number,
  kneeBend: number,
  side: BodySide,
  confidence = 0.96,
  trunkTilt = 0,
): { image: LandmarkSet; world: LandmarkSet } {
  const width = 1280;
  const height = 720;
  const hip = { x: 0.28, y: 0.62 };
  const segmentPx = 230;
  const worldSegment = 0.25;
  const liftRadians = (lift * Math.PI) / 180;
  const bendRadians = (kneeBend * Math.PI) / 180;
  const knee = {
    x: hip.x + (Math.cos(liftRadians) * segmentPx) / width,
    y: hip.y - (Math.sin(liftRadians) * segmentPx) / height,
  };
  const ankle = {
    x:
      knee.x +
      (Math.cos(liftRadians - bendRadians) * segmentPx) / width,
    y:
      knee.y -
      (Math.sin(liftRadians - bendRadians) * segmentPx) / height,
  };
  const shoulder = {
    x: hip.x - 0.16,
    y: hip.y - Math.tan((trunkTilt * Math.PI) / 180) * 0.16,
  };
  const imageSet = withBodyScaffold(
    withSide(side, {
      shoulder: landmark(shoulder.x, shoulder.y, 0, confidence),
      hip: landmark(hip.x, hip.y, 0, confidence),
      knee: landmark(knee.x, knee.y, 0, confidence),
      ankle: landmark(ankle.x, ankle.y, 0, confidence),
      heel: landmark(ankle.x - 0.02, ankle.y, 0, confidence),
    }),
    side,
    "lying",
    confidence,
  );
  const worldKnee = {
    x: hip.x + Math.cos(liftRadians) * worldSegment,
    y: hip.y - Math.sin(liftRadians) * worldSegment,
  };
  const worldAnkle = {
    x:
      worldKnee.x +
      Math.cos(liftRadians - bendRadians) * worldSegment,
    y:
      worldKnee.y -
      Math.sin(liftRadians - bendRadians) * worldSegment,
  };
  const worldSet = withSide(side, {
    shoulder: landmark(shoulder.x, shoulder.y, 0, confidence),
    hip: landmark(hip.x, hip.y, 0, confidence),
    knee: landmark(worldKnee.x, worldKnee.y, 0, confidence),
    ankle: landmark(worldAnkle.x, worldAnkle.y, 0, confidence),
    heel: landmark(worldAnkle.x - 0.02, worldAnkle.y, 0, confidence),
  });
  return { image: imageSet, world: worldSet };
}

function interpolate(
  start: number,
  end: number,
  count: number,
): number[] {
  if (count <= 1) return [end];
  return Array.from(
    { length: count },
    (_, index) => start + ((end - start) * index) / (count - 1),
  );
}

function frameValues(
  exerciseId: ExerciseId,
  scenario: ReplayScenarioId,
  holdSeconds: number,
  intervalMs: number,
): number[] {
  const fast = scenario === "fast";
  const slow = scenario === "slow";
  const transitionFrames = fast ? 3 : slow ? 22 : 10;
  const holdFrames = Math.max(
    2,
    Math.round((holdSeconds * 1000) / intervalMs) + 1,
  );
  const restFrames = Math.max(3, Math.round(500 / intervalMs));
  const calibrationFrames = Math.max(8, Math.round(1800 / intervalMs));
  const isSeated = exerciseId === "seated-knee-extension";
  const start = isSeated ? 90 : 4;
  let peak = isSeated ? 10 : exerciseId === "straight-leg-raise" ? 38 : 76;

  if (scenario === "incomplete-range")
    peak = isSeated ? 50 : exerciseId === "straight-leg-raise" ? 18 : 30;
  if (scenario === "approach-limit" && exerciseId === "heel-slide") peak = 86;
  if (scenario === "exceed-limit" && exerciseId === "heel-slide") peak = 94;

  const outbound = interpolate(start, peak, transitionFrames);
  const hold = Array.from({ length: holdFrames }, () => peak);
  const returnEnd =
    scenario === "incomplete-return"
      ? isSeated
        ? 55
        : exerciseId === "straight-leg-raise"
          ? 18
          : 32
      : start;
  const returning = interpolate(peak, returnEnd, transitionFrames);
  const finalRestFrames =
    scenario === "incomplete-return"
      ? Math.max(restFrames, Math.ceil(16000 / intervalMs))
      : restFrames;
  const rest = Array.from({ length: finalRestFrames }, () => returnEnd);
  const calibration = Array.from({ length: calibrationFrames }, () => start);
  return [...calibration, ...outbound, ...hold, ...returning, ...rest];
}

export function createReplayFrames({
  exerciseId,
  side,
  scenario = "correct",
  reps = 1,
  holdSeconds = 2,
  intervalMs = 100,
}: ReplayFixtureOptions): PoseFrame[] {
  const frames: PoseFrame[] = [];
  let timestampMs = 0;
  const baseValues = frameValues(
    exerciseId,
    scenario,
    holdSeconds,
    intervalMs,
  );

  for (let rep = 0; rep < reps; rep += 1) {
    baseValues.forEach((value, index) => {
      const isFaultWindow =
        index >= Math.floor(baseValues.length * 0.35) &&
        index <= Math.floor(baseValues.length * 0.55);
      const confidence =
        (scenario === "tracking-loss" || scenario === "occlusion") &&
        isFaultWindow
          ? scenario === "tracking-loss"
            ? 0.1
            : 0.36
          : 0.96;
      const worldValue =
        scenario === "uncertain-angle" && isFaultWindow ? value + 24 : value;
      let pose =
        exerciseId === "straight-leg-raise"
          ? straightLegRaisePose(
              value,
              (scenario === "occlusion" ||
                scenario === "posture-variation") &&
                isFaultWindow
                ? 22
                : 4,
              side,
              confidence,
              scenario === "posture-variation" && isFaultWindow
                ? 28
                : scenario === "poor-camera"
                  ? 20
                  : 2,
            )
          : heelOrSeatedPose(
              value,
              side,
              exerciseId === "seated-knee-extension",
              confidence,
              worldValue,
            );

      if (scenario === "poor-camera") {
        const shift = -0.2;
        const shiftSet = (set: LandmarkSet): LandmarkSet =>
          Object.fromEntries(
            Object.entries(set).map(([key, point]) => [
              key,
              point ? { ...point, x: point.x + shift } : point,
            ]),
          ) as LandmarkSet;
        pose = { image: shiftSet(pose.image), world: pose.world };
      }

      frames.push({
        timestampMs,
        image: pose.image,
        world: pose.world,
        frameWidth: 1280,
        frameHeight: 720,
        source: "replay",
      });
      timestampMs += intervalMs;
    });
  }
  return frames;
}

export const REPLAY_SCENARIOS: Array<{
  id: ReplayScenarioId;
  label: string;
  description: string;
}> = [
  {
    id: "correct",
    label: "Clean repetitions",
    description: "Complete movement, hold, and return with good tracking.",
  },
  {
    id: "incomplete-range",
    label: "Incomplete movement",
    description: "Movement starts but does not reach the exercise endpoint.",
  },
  {
    id: "incomplete-return",
    label: "Incomplete return",
    description: "Movement stops before returning to the starting position.",
  },
  {
    id: "fast",
    label: "Too fast",
    description: "A complete repetition below the configured tempo.",
  },
  {
    id: "slow",
    label: "Slow movement",
    description: "A deliberately slow repetition with a complete hold.",
  },
  {
    id: "tracking-loss",
    label: "Tracking loss",
    description: "Required landmarks disappear during the movement.",
  },
  {
    id: "occlusion",
    label: "Joint occlusion",
    description: "A required joint becomes unreliable during the movement.",
  },
  {
    id: "poor-camera",
    label: "Poor camera position",
    description: "Required joints are too close to or outside the frame.",
  },
  {
    id: "posture-variation",
    label: "Visible compensation",
    description:
      "A straight-leg raise includes clear knee bend and shoulder–hip movement.",
  },
  {
    id: "approach-limit",
    label: "Approach maximum",
    description: "Heel slide approaches a configured 90° maximum.",
  },
  {
    id: "exceed-limit",
    label: "Reach maximum",
    description: "Heel slide reaches and crosses a configured 90° maximum.",
  },
  {
    id: "uncertain-angle",
    label: "Uncertain angle",
    description: "2D and inferred 3D geometry disagree, pausing angle rules.",
  },
];
