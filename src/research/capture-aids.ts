import type {
  JointName,
  LandmarkSet,
  PoseLandmark,
} from "../motion/types";
import type { ResearchPoseFrame } from "./capture-bundle";

export type ResearchAidCoordinateSource = "image-2d" | "world-3d";
export type ResearchAidBodySide = "left" | "right";
export type ResearchAidAngleFamily = "elbow" | "knee" | "hip";
export type ResearchAidAngleName =
  `${ResearchAidBodySide}-${ResearchAidAngleFamily}`;

export interface ResearchObservationAids {
  source: ResearchAidCoordinateSource;
  expectedJointCount: number;
  observedJointCount: number;
  coverageFraction: number;
  meanVisibility: number | null;
}

export interface ResearchAngleAid {
  name: ResearchAidAngleName;
  family: ResearchAidAngleFamily;
  side: ResearchAidBodySide;
  source: ResearchAidCoordinateSource;
  evidenceJoints: readonly [JointName, JointName, JointName];
  angleDegrees: number;
  minimumVisibility: number;
  minimumPresence: number;
  angularRateDegreesPerSecond?: number;
}

export interface ResearchSymmetryAid {
  family: ResearchAidAngleFamily;
  source: ResearchAidCoordinateSource;
  convention: "left-minus-right";
  deltaDegrees: number;
}

export interface ResearchFrameAids {
  frameIndex: number;
  sequenceNumber: number | null;
  tMs: number | null;
  observations: {
    image: ResearchObservationAids;
    world: ResearchObservationAids;
  };
  angles: ResearchAngleAid[];
  symmetry: ResearchSymmetryAid[];
}

export interface ResearchCaptureAids {
  schemaVersion: "relay.research-capture-aids/1";
  evidenceClass: "DERIVED";
  interpretation: "deterministic-analytic-aids-not-ground-truth";
  causal: true;
  causalOrder: "input-frame-order";
  rawCoordinatesIncluded: false;
  methods: {
    angles: "three-point-included-angle";
    angularRates: "adjacent-frame-backward-difference";
    symmetry: "left-minus-right";
  };
  frameCount: number;
  frames: ResearchFrameAids[];
}

interface AngleDefinition {
  name: ResearchAidAngleName;
  family: ResearchAidAngleFamily;
  side: ResearchAidBodySide;
  joints: readonly [JointName, JointName, JointName];
}

type BaseAngleAid = Omit<ResearchAngleAid, "angularRateDegreesPerSecond">;

const RESEARCH_POSE_JOINTS = [
  "nose",
  "left-eye-inner",
  "left-eye",
  "left-eye-outer",
  "right-eye-inner",
  "right-eye",
  "right-eye-outer",
  "left-ear",
  "right-ear",
  "mouth-left",
  "mouth-right",
  "left-shoulder",
  "right-shoulder",
  "left-elbow",
  "right-elbow",
  "left-wrist",
  "right-wrist",
  "left-pinky",
  "right-pinky",
  "left-index",
  "right-index",
  "left-thumb",
  "right-thumb",
  "left-hip",
  "right-hip",
  "left-knee",
  "right-knee",
  "left-ankle",
  "right-ankle",
  "left-heel",
  "right-heel",
  "left-foot-index",
  "right-foot-index",
] as const satisfies readonly JointName[];

const COORDINATE_SOURCES = ["image-2d", "world-3d"] as const;

const ANGLE_DEFINITIONS = [
  {
    name: "left-elbow",
    family: "elbow",
    side: "left",
    joints: ["left-shoulder", "left-elbow", "left-wrist"],
  },
  {
    name: "right-elbow",
    family: "elbow",
    side: "right",
    joints: ["right-shoulder", "right-elbow", "right-wrist"],
  },
  {
    name: "left-knee",
    family: "knee",
    side: "left",
    joints: ["left-hip", "left-knee", "left-ankle"],
  },
  {
    name: "right-knee",
    family: "knee",
    side: "right",
    joints: ["right-hip", "right-knee", "right-ankle"],
  },
  {
    name: "left-hip",
    family: "hip",
    side: "left",
    joints: ["left-shoulder", "left-hip", "left-knee"],
  },
  {
    name: "right-hip",
    family: "hip",
    side: "right",
    joints: ["right-shoulder", "right-hip", "right-knee"],
  },
] as const satisfies readonly AngleDefinition[];

function isFiniteConfidence(value: number): boolean {
  return Number.isFinite(value) && value > 0 && value <= 1;
}

function isObservableLandmark(
  landmark: PoseLandmark | undefined,
): landmark is PoseLandmark {
  return Boolean(
    landmark &&
      Number.isFinite(landmark.x) &&
      Number.isFinite(landmark.y) &&
      Number.isFinite(landmark.z) &&
      isFiniteConfidence(landmark.visibility) &&
      isFiniteConfidence(landmark.presence),
  );
}

function deriveObservationAids(
  landmarks: LandmarkSet,
  source: ResearchAidCoordinateSource,
): ResearchObservationAids {
  let observedJointCount = 0;
  let visibilitySum = 0;

  for (const joint of RESEARCH_POSE_JOINTS) {
    const landmark = landmarks[joint];
    if (!isObservableLandmark(landmark)) continue;
    observedJointCount += 1;
    visibilitySum += landmark.visibility;
  }

  return {
    source,
    expectedJointCount: RESEARCH_POSE_JOINTS.length,
    observedJointCount,
    coverageFraction: observedJointCount / RESEARCH_POSE_JOINTS.length,
    meanVisibility:
      observedJointCount === 0 ? null : visibilitySum / observedJointCount,
  };
}

function coordinatesFor(
  frame: ResearchPoseFrame,
  source: ResearchAidCoordinateSource,
  landmark: PoseLandmark,
): readonly [number, number, number] | null {
  if (source === "world-3d") return [landmark.x, landmark.y, landmark.z];

  if (
    !Number.isFinite(frame.frameWidth) ||
    !Number.isFinite(frame.frameHeight) ||
    frame.frameWidth <= 0 ||
    frame.frameHeight <= 0
  ) {
    return null;
  }

  const x = landmark.x * frame.frameWidth;
  const y = landmark.y * frame.frameHeight;
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y, 0] : null;
}

function includedAngleDegrees(
  first: readonly [number, number, number],
  vertex: readonly [number, number, number],
  third: readonly [number, number, number],
): number | null {
  const firstVector = [
    first[0] - vertex[0],
    first[1] - vertex[1],
    first[2] - vertex[2],
  ] as const;
  const thirdVector = [
    third[0] - vertex[0],
    third[1] - vertex[1],
    third[2] - vertex[2],
  ] as const;
  if (
    !firstVector.every(Number.isFinite) ||
    !thirdVector.every(Number.isFinite)
  ) {
    return null;
  }

  const firstLength = Math.hypot(...firstVector);
  const thirdLength = Math.hypot(...thirdVector);
  if (
    !Number.isFinite(firstLength) ||
    !Number.isFinite(thirdLength) ||
    firstLength === 0 ||
    thirdLength === 0
  ) {
    return null;
  }

  const cosine =
    (firstVector[0] / firstLength) * (thirdVector[0] / thirdLength) +
    (firstVector[1] / firstLength) * (thirdVector[1] / thirdLength) +
    (firstVector[2] / firstLength) * (thirdVector[2] / thirdLength);
  if (!Number.isFinite(cosine)) return null;

  const angleDegrees =
    Math.acos(Math.max(-1, Math.min(1, cosine))) * (180 / Math.PI);
  return Number.isFinite(angleDegrees) ? angleDegrees : null;
}

function deriveBaseAngles(frame: ResearchPoseFrame): BaseAngleAid[] {
  const result: BaseAngleAid[] = [];

  for (const source of COORDINATE_SOURCES) {
    const landmarks = source === "image-2d" ? frame.image : frame.world;
    for (const definition of ANGLE_DEFINITIONS) {
      const points = definition.joints.map((joint) => landmarks[joint]);
      if (!points.every(isObservableLandmark)) continue;

      const observedPoints = points as [PoseLandmark, PoseLandmark, PoseLandmark];
      const coordinates = observedPoints.map((point) =>
        coordinatesFor(frame, source, point),
      );
      if (coordinates.some((point) => point === null)) continue;

      const [first, vertex, third] = coordinates as [
        readonly [number, number, number],
        readonly [number, number, number],
        readonly [number, number, number],
      ];
      const angleDegrees = includedAngleDegrees(first, vertex, third);
      if (angleDegrees === null) continue;

      result.push({
        name: definition.name,
        family: definition.family,
        side: definition.side,
        source,
        evidenceJoints: definition.joints,
        angleDegrees,
        minimumVisibility: Math.min(
          ...observedPoints.map((point) => point.visibility),
        ),
        minimumPresence: Math.min(
          ...observedPoints.map((point) => point.presence),
        ),
      });
    }
  }

  return result;
}

function angleKey(
  source: ResearchAidCoordinateSource,
  name: ResearchAidAngleName,
): string {
  return `${source}:${name}`;
}

function deriveSymmetry(angles: readonly ResearchAngleAid[]): ResearchSymmetryAid[] {
  const byKey = new Map(
    angles.map((angle) => [angleKey(angle.source, angle.name), angle]),
  );
  const result: ResearchSymmetryAid[] = [];

  for (const source of COORDINATE_SOURCES) {
    for (const family of ["elbow", "knee", "hip"] as const) {
      const left = byKey.get(angleKey(source, `left-${family}`));
      const right = byKey.get(angleKey(source, `right-${family}`));
      if (!left || !right) continue;
      const deltaDegrees = left.angleDegrees - right.angleDegrees;
      if (!Number.isFinite(deltaDegrees)) continue;
      result.push({
        family,
        source,
        convention: "left-minus-right",
        deltaDegrees,
      });
    }
  }

  return result;
}

function finiteOrNull(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

export function deriveResearchCaptureAids(
  frames: readonly ResearchPoseFrame[],
): ResearchCaptureAids {
  const derivedFrames: ResearchFrameAids[] = [];
  let previous:
    | {
        tMs: number | null;
        angles: Map<string, BaseAngleAid>;
      }
    | undefined;

  for (const [frameIndex, frame] of frames.entries()) {
    const tMs = finiteOrNull(frame.tMs);
    const baseAngles = deriveBaseAngles(frame);
    const currentAngles = new Map(
      baseAngles.map((angle) => [angleKey(angle.source, angle.name), angle]),
    );
    const deltaSeconds =
      previous && previous.tMs !== null && tMs !== null
        ? (tMs - previous.tMs) / 1000
        : null;
    const angles = baseAngles.map<ResearchAngleAid>((angle) => {
      if (deltaSeconds === null || deltaSeconds <= 0 || !previous) return angle;
      const previousAngle = previous.angles.get(angleKey(angle.source, angle.name));
      if (!previousAngle) return angle;
      const angularRateDegreesPerSecond =
        (angle.angleDegrees - previousAngle.angleDegrees) / deltaSeconds;
      return Number.isFinite(angularRateDegreesPerSecond)
        ? { ...angle, angularRateDegreesPerSecond }
        : angle;
    });

    derivedFrames.push({
      frameIndex,
      sequenceNumber: finiteOrNull(frame.sequenceNumber),
      tMs,
      observations: {
        image: deriveObservationAids(frame.image, "image-2d"),
        world: deriveObservationAids(frame.world, "world-3d"),
      },
      angles,
      symmetry: deriveSymmetry(angles),
    });
    previous = { tMs, angles: currentAngles };
  }

  return {
    schemaVersion: "relay.research-capture-aids/1",
    evidenceClass: "DERIVED",
    interpretation: "deterministic-analytic-aids-not-ground-truth",
    causal: true,
    causalOrder: "input-frame-order",
    rawCoordinatesIncluded: false,
    methods: {
      angles: "three-point-included-angle",
      angularRates: "adjacent-frame-backward-difference",
      symmetry: "left-minus-right",
    },
    frameCount: derivedFrames.length,
    frames: derivedFrames,
  };
}
