import type { BodySide, ExerciseId } from "../domain/types";
import type {
  JointName,
  LandmarkSet,
  MotionMeasurements,
  PoseFrame,
  PoseLandmark,
} from "./types";

const DEG = 180 / Math.PI;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function sideJoint(side: BodySide, joint: string): JointName {
  return `${side}-${joint}` as JointName;
}

function angleBetween2D(
  a: PoseLandmark,
  vertex: PoseLandmark,
  c: PoseLandmark,
  width: number,
  height: number,
): number | undefined {
  const ax = (a.x - vertex.x) * width;
  const ay = (a.y - vertex.y) * height;
  const cx = (c.x - vertex.x) * width;
  const cy = (c.y - vertex.y) * height;
  const denominator = Math.hypot(ax, ay) * Math.hypot(cx, cy);
  if (denominator < 1e-6) return undefined;
  return Math.acos(clamp((ax * cx + ay * cy) / denominator, -1, 1)) * DEG;
}

function angleBetween3D(
  a: PoseLandmark,
  vertex: PoseLandmark,
  c: PoseLandmark,
): number | undefined {
  const av = [a.x - vertex.x, a.y - vertex.y, a.z - vertex.z];
  const cv = [c.x - vertex.x, c.y - vertex.y, c.z - vertex.z];
  const denominator =
    Math.hypot(...av) * Math.hypot(...cv);
  if (denominator < 1e-6) return undefined;
  const dot = av[0] * cv[0] + av[1] * cv[1] + av[2] * cv[2];
  return Math.acos(clamp(dot / denominator, -1, 1)) * DEG;
}

function kneeFlexionFrom(
  landmarks: LandmarkSet,
  side: BodySide,
  dimensions?: { width: number; height: number },
): number | undefined {
  const hip = landmarks[sideJoint(side, "hip")];
  const knee = landmarks[sideJoint(side, "knee")];
  const ankle = landmarks[sideJoint(side, "ankle")];
  if (!hip || !knee || !ankle) return undefined;
  const included = dimensions
    ? angleBetween2D(
        hip,
        knee,
        ankle,
        dimensions.width,
        dimensions.height,
      )
    : angleBetween3D(hip, knee, ankle);
  return included === undefined ? undefined : clamp(180 - included, 0, 180);
}

function landmarkQuality(
  landmarks: LandmarkSet,
  joints: JointName[],
): {
  minimum: number;
  average: number;
  missing: JointName[];
} {
  const values: number[] = [];
  const missing: JointName[] = [];
  joints.forEach((joint) => {
    const landmark = landmarks[joint];
    if (!landmark) {
      missing.push(joint);
      values.push(0);
      return;
    }
    values.push(Math.min(landmark.visibility, landmark.presence));
  });
  return {
    minimum: Math.min(...values),
    average: values.reduce((sum, value) => sum + value, 0) / values.length,
    missing,
  };
}

function framingScore(
  landmarks: LandmarkSet,
  joints: JointName[],
): { score: number; clipped: boolean } {
  const present = joints
    .map((joint) => landmarks[joint])
    .filter(Boolean) as PoseLandmark[];
  if (present.length !== joints.length) return { score: 0, clipped: true };
  const smallestMargin = Math.min(
    ...present.flatMap((landmark) => [
      landmark.x,
      1 - landmark.x,
      landmark.y,
      1 - landmark.y,
    ]),
  );
  return {
    score: clamp(smallestMargin / 0.12, 0, 1),
    clipped: smallestMargin < 0.055,
  };
}

function minimumProjectedSegmentLength(
  landmarks: LandmarkSet,
  side: BodySide,
  width: number,
  height: number,
): number | undefined {
  const hip = landmarks[sideJoint(side, "hip")];
  const knee = landmarks[sideJoint(side, "knee")];
  const ankle = landmarks[sideJoint(side, "ankle")];
  if (!hip || !knee || !ankle) return undefined;
  const distance = (first: PoseLandmark, second: PoseLandmark) =>
    Math.hypot(
      (first.x - second.x) * width,
      (first.y - second.y) * height,
    );
  return Math.min(distance(hip, knee), distance(knee, ankle));
}

function outOfPlaneRatio(
  landmarks: LandmarkSet,
  side: BodySide,
): number | undefined {
  const hip = landmarks[sideJoint(side, "hip")];
  const knee = landmarks[sideJoint(side, "knee")];
  const ankle = landmarks[sideJoint(side, "ankle")];
  if (!hip || !knee || !ankle) return undefined;
  const depth =
    Math.abs(hip.z - knee.z) + Math.abs(knee.z - ankle.z);
  const length =
    Math.hypot(hip.x - knee.x, hip.y - knee.y, hip.z - knee.z) +
    Math.hypot(knee.x - ankle.x, knee.y - ankle.y, knee.z - ankle.z);
  return length > 0 ? depth / length : undefined;
}

function legLift(
  landmarks: LandmarkSet,
  side: BodySide,
  width: number,
  height: number,
): number | undefined {
  const hip = landmarks[sideJoint(side, "hip")];
  const ankle = landmarks[sideJoint(side, "ankle")];
  if (!hip || !ankle) return undefined;
  const dx = Math.abs((ankle.x - hip.x) * width);
  const dy = (hip.y - ankle.y) * height;
  return clamp(Math.atan2(dy, Math.max(dx, 1e-6)) * DEG, -20, 100);
}

function trunkCompensation(
  landmarks: LandmarkSet,
  side: BodySide,
  width: number,
  height: number,
): number | undefined {
  const shoulder = landmarks[sideJoint(side, "shoulder")];
  const hip = landmarks[sideJoint(side, "hip")];
  if (!shoulder || !hip) return undefined;
  const dx = Math.abs((hip.x - shoulder.x) * width);
  const dy = Math.abs((hip.y - shoulder.y) * height);
  return Math.atan2(dy, Math.max(dx, 1e-6)) * DEG;
}

export function measurePose(
  frame: PoseFrame,
  exerciseId: ExerciseId,
  side: BodySide,
): MotionMeasurements {
  const legJoints: JointName[] = [
    sideJoint(side, "hip"),
    sideJoint(side, "knee"),
    sideJoint(side, "ankle"),
  ];
  const postureJoints: JointName[] = [
    sideJoint(side, "shoulder"),
    sideJoint(side, "hip"),
  ];
  const legQuality = landmarkQuality(frame.image, legJoints);
  const legFraming = framingScore(frame.image, legJoints);
  const postureQuality = landmarkQuality(frame.image, postureJoints);
  const postureFraming = framingScore(frame.image, postureJoints);
  const minimumSegment = minimumProjectedSegmentLength(
    frame.image,
    side,
    frame.frameWidth,
    frame.frameHeight,
  );
  const imageKnee = kneeFlexionFrom(frame.image, side, {
    width: frame.frameWidth,
    height: frame.frameHeight,
  });
  const worldKnee = kneeFlexionFrom(frame.world, side);
  const agreement =
    imageKnee !== undefined && worldKnee !== undefined
      ? Math.abs(imageKnee - worldKnee)
      : undefined;
  const planeRatio = outOfPlaneRatio(frame.world, side);
  const reasons: string[] = [];

  if (legQuality.missing.length) reasons.push("Required joints are not visible.");
  if (legQuality.minimum < 0.45)
    reasons.push("A required joint is not tracked reliably.");
  if (legFraming.clipped)
    reasons.push("Move back so the hip, knee, and ankle stay inside the frame.");
  if (minimumSegment !== undefined && minimumSegment < 75)
    reasons.push("Move closer so the visible limb segments are large enough.");
  if (agreement !== undefined && agreement > 12)
    reasons.push("The 2D and 3D angle estimates do not agree.");
  if (planeRatio !== undefined && planeRatio > 0.3)
    reasons.push("Turn more side-on to the camera.");
  if (
    exerciseId === "straight-leg-raise" &&
    (postureQuality.minimum < 0.45 || postureQuality.missing.length > 0)
  )
    reasons.push("The shoulder–hip relationship is not visible.");

  const angleLost =
    legQuality.minimum < 0.45 || legQuality.missing.length > 0;
  const angleLimited =
    !angleLost &&
    (legQuality.minimum < 0.7 ||
      legFraming.clipped ||
      (minimumSegment !== undefined && minimumSegment < 75) ||
      (agreement !== undefined && agreement > 12) ||
      (planeRatio !== undefined && planeRatio > 0.3));
  const angleStatus = angleLost
    ? "lost"
    : angleLimited
      ? "limited"
      : "good";
  const chainLimited =
    !angleLost &&
    (legQuality.minimum < 0.7 ||
      legFraming.clipped ||
      (minimumSegment !== undefined && minimumSegment < 75) ||
      (planeRatio !== undefined && planeRatio > 0.3));
  const chainStatus = angleLost
    ? "lost"
    : chainLimited
      ? "limited"
      : "good";
  const postureLost =
    postureQuality.minimum < 0.45 || postureQuality.missing.length > 0;
  const postureLimited =
    !postureLost &&
    (postureQuality.minimum < 0.7 || postureFraming.clipped);
  const postureStatus = postureLost
    ? "lost"
    : postureLimited
      ? "limited"
      : "good";
  // The primary movement feature determines whether phase/rep assessment can
  // continue. Secondary features (for example shoulder–hip stability during a
  // straight-leg raise) keep their own quality and abstain independently.
  const status =
    exerciseId === "straight-leg-raise" ? chainStatus : angleStatus;
  const geometryScore =
    agreement === undefined ? 0.6 : clamp(1 - agreement / 30, 0, 1);
  const planeScore =
    planeRatio === undefined ? 0.6 : clamp(1 - planeRatio / 0.5, 0, 1);
  const score = clamp(
    legQuality.average * 0.5 +
      legFraming.score * 0.2 +
      geometryScore * 0.2 +
      planeScore * 0.1,
    0,
    1,
  );
  const canAssessAngle =
    angleStatus === "good" &&
    imageKnee !== undefined &&
    worldKnee !== undefined;
  const canAssessBodyFeatures = status === "good";

  const selectedKnee =
    canAssessAngle && imageKnee !== undefined && worldKnee !== undefined
      ? imageKnee * 0.8 + worldKnee * 0.2
      : undefined;
  const lift =
    exerciseId === "straight-leg-raise" && chainStatus === "good"
      ? legLift(frame.image, side, frame.frameWidth, frame.frameHeight)
      : undefined;
  const extension =
    exerciseId === "seated-knee-extension" && selectedKnee !== undefined
      ? clamp((90 - selectedKnee) / 90, 0, 1) * 100
      : undefined;
  const trunk =
    exerciseId === "straight-leg-raise" && postureStatus === "good"
      ? trunkCompensation(
          frame.image,
          side,
          frame.frameWidth,
          frame.frameHeight,
        )
      : undefined;
  const sideHip = sideJoint(side, "hip");
  const sideKnee = sideJoint(side, "knee");
  const sideAnkle = sideJoint(side, "ankle");
  const sideShoulder = sideJoint(side, "shoulder");

  return {
    kneeFlexion: selectedKnee,
    kneeFlexionImage: imageKnee,
    kneeFlexionWorld: worldKnee,
    angleAgreementDelta: agreement,
    legLift: lift,
    extensionProgress: extension,
    kneeBend:
      exerciseId === "straight-leg-raise" ? selectedKnee : undefined,
    trunkCompensation: trunk,
    values: {
      "knee-flexion": selectedKnee,
      "leg-lift": lift,
      "extension-progress": extension,
      "knee-bend":
        exerciseId === "straight-leg-raise" ? selectedKnee : undefined,
      "trunk-compensation": trunk,
    },
    features: [
      {
        id: "knee-flexion",
        metric: "knee-flexion",
        joints: [sideHip, sideKnee, sideAnkle],
        value: selectedKnee,
        unit: "degrees",
        quality: angleStatus,
        description: "Aspect-correct hip–knee–ankle joint angle",
      },
      {
        id: "leg-chain-lift",
        metric: "leg-lift",
        joints: [sideHip, sideKnee, sideAnkle],
        value: lift,
        unit: "degrees",
        quality: chainStatus,
        description: "Movement of the selected leg chain from horizontal",
      },
      {
        id: "extension-progress",
        metric: "extension-progress",
        joints: [sideHip, sideKnee, sideAnkle],
        value: extension,
        unit: "percent",
        quality: angleStatus,
        description: "Selected leg extension relative to seated start",
      },
      {
        id: "leg-straightness",
        metric: "knee-bend",
        joints: [sideHip, sideKnee, sideAnkle],
        value: exerciseId === "straight-leg-raise" ? selectedKnee : undefined,
        unit: "degrees",
        quality: angleStatus,
        description: "Broad visible alignment of the selected leg chain",
      },
      {
        id: "shoulder-hip-stability",
        metric: "trunk-compensation",
        joints: [sideShoulder, sideHip],
        value: trunk,
        unit: "degrees",
        quality: postureStatus,
        description: "Broad visible shoulder–hip relationship stability",
      },
    ],
    confidence: {
      score,
      status,
      canAssessBodyFeatures,
      canAssessAngle,
      reasons:
        reasons.length > 0
          ? reasons
          : ["Required joints are visible and geometry is consistent."],
    },
  };
}
