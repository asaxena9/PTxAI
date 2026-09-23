import { describe, expect, it } from "vitest";
import {
  deriveResearchCaptureAids,
  type ResearchAngleAid,
} from "../src/research/capture-aids";
import type { ResearchPoseFrame } from "../src/research/capture-bundle";
import type { LandmarkSet, PoseLandmark } from "../src/motion/types";

function landmark(
  x: number,
  y: number,
  z = 0,
  visibility = 1,
  presence = 1,
): PoseLandmark {
  return { x, y, z, visibility, presence };
}

function frame(
  tMs: number,
  world: LandmarkSet = {},
  image: LandmarkSet = {},
  sequenceNumber = tMs,
): ResearchPoseFrame {
  return {
    sequenceNumber,
    tMs,
    sourceTimestampMs: tMs,
    callbackMonotonicTimeMs: tMs,
    frameWidth: 100,
    frameHeight: 100,
    poseDetected: Object.keys(image).length > 0 || Object.keys(world).length > 0,
    imageJointCount: Object.keys(image).length,
    worldJointCount: Object.keys(world).length,
    image,
    world,
  };
}

function bilateralRightAnglePose(): LandmarkSet {
  return {
    "left-shoulder": landmark(0.2, 0.2),
    "left-elbow": landmark(0.2, 0.35),
    "left-wrist": landmark(0.35, 0.35),
    "left-hip": landmark(0.2, 0.5),
    "left-knee": landmark(0.2, 0.7),
    "left-ankle": landmark(0.35, 0.7),
    "right-shoulder": landmark(0.8, 0.2),
    "right-elbow": landmark(0.8, 0.35),
    "right-wrist": landmark(0.65, 0.35),
    "right-hip": landmark(0.8, 0.5),
    "right-knee": landmark(0.8, 0.7),
    "right-ankle": landmark(0.65, 0.7),
  };
}

function elbowAt(degrees: number): LandmarkSet {
  const radians = degrees * (Math.PI / 180);
  return {
    "left-shoulder": landmark(1, 0),
    "left-elbow": landmark(0, 0),
    "left-wrist": landmark(Math.cos(radians), Math.sin(radians)),
  };
}

function findAngle(
  angles: readonly ResearchAngleAid[],
  name: ResearchAngleAid["name"],
  source: ResearchAngleAid["source"],
): ResearchAngleAid {
  const result = angles.find(
    (angle) => angle.name === name && angle.source === source,
  );
  if (!result) throw new Error(`Missing ${source} ${name} angle`);
  return result;
}

function numericValues(value: unknown): number[] {
  if (typeof value === "number") return [value];
  if (Array.isArray(value)) return value.flatMap(numericValues);
  if (value && typeof value === "object") {
    return Object.values(value).flatMap(numericValues);
  }
  return [];
}

describe("research capture derived aids", () => {
  it("derives known bilateral geometry with explicit provenance", () => {
    const pose = bilateralRightAnglePose();
    const result = deriveResearchCaptureAids([frame(0, pose, pose)]);
    const derived = result.frames[0];

    expect(result).toMatchObject({
      evidenceClass: "DERIVED",
      interpretation: "deterministic-analytic-aids-not-ground-truth",
      causal: true,
      rawCoordinatesIncluded: false,
      frameCount: 1,
    });
    expect(derived.observations.image).toMatchObject({
      observedJointCount: 12,
      coverageFraction: 12 / 33,
      meanVisibility: 1,
    });
    expect(derived.observations.world.observedJointCount).toBe(12);
    expect(derived.angles).toHaveLength(12);
    expect(
      findAngle(derived.angles, "left-elbow", "image-2d").angleDegrees,
    ).toBeCloseTo(90);
    expect(
      findAngle(derived.angles, "left-knee", "world-3d").angleDegrees,
    ).toBeCloseTo(90);
    expect(
      findAngle(derived.angles, "left-hip", "world-3d").angleDegrees,
    ).toBeCloseTo(180);
    expect(derived.angles.every((angle) => angle.angularRateDegreesPerSecond === undefined)).toBe(true);
    expect(derived.symmetry).toHaveLength(6);
    expect(derived.symmetry.every((aid) => Math.abs(aid.deltaDegrees) < 1e-10)).toBe(
      true,
    );
    expect(derived).not.toHaveProperty("image");
    expect(derived).not.toHaveProperty("world");
  });

  it("uses only the adjacent earlier frame for angular rates", () => {
    const result = deriveResearchCaptureAids([
      frame(0, elbowAt(90)),
      frame(500, elbowAt(60)),
      frame(1000, {}),
      frame(1500, elbowAt(30)),
    ]);

    const first = findAngle(result.frames[0].angles, "left-elbow", "world-3d");
    const second = findAngle(result.frames[1].angles, "left-elbow", "world-3d");
    const fourth = findAngle(result.frames[3].angles, "left-elbow", "world-3d");
    expect(first.angularRateDegreesPerSecond).toBeUndefined();
    expect(second.angularRateDegreesPerSecond).toBeCloseTo(-60);
    expect(fourth.angularRateDegreesPerSecond).toBeUndefined();
  });

  it("fails closed for missing, zero-confidence, degenerate, and nonfinite data", () => {
    const malformedWorld: LandmarkSet = {
      "left-shoulder": landmark(1, 0),
      "left-elbow": landmark(Number.NaN, 0),
      "left-wrist": landmark(0, 1),
      "right-shoulder": landmark(1, 0, 0, 0),
      "right-elbow": landmark(0, 0),
      "right-wrist": landmark(Number.POSITIVE_INFINITY, 1),
      "left-hip": landmark(0, 0),
      "left-knee": landmark(0, 0),
      "left-ankle": landmark(0, 1),
    };
    const malformedFrame = frame(
      Number.POSITIVE_INFINITY,
      malformedWorld,
      {},
      Number.NaN,
    );
    const result = deriveResearchCaptureAids([malformedFrame]);
    const derived = result.frames[0];

    expect(derived.sequenceNumber).toBeNull();
    expect(derived.tMs).toBeNull();
    expect(derived.observations.image).toMatchObject({
      observedJointCount: 0,
      coverageFraction: 0,
      meanVisibility: null,
    });
    expect(derived.observations.world.observedJointCount).toBe(6);
    expect(derived.angles).toEqual([]);
    expect(derived.symmetry).toEqual([]);
    expect(numericValues(result).every(Number.isFinite)).toBe(true);
  });

  it("is deterministic, prefix-stable, and leaves raw coordinates unchanged", () => {
    const rawFrames = [
      frame(0, elbowAt(90)),
      frame(500, elbowAt(60)),
      frame(1000, elbowAt(30)),
    ];
    const snapshot = structuredClone(rawFrames);
    const full = deriveResearchCaptureAids(rawFrames);

    expect(deriveResearchCaptureAids(rawFrames)).toEqual(full);
    for (let prefixLength = 1; prefixLength <= rawFrames.length; prefixLength += 1) {
      expect(full.frames.slice(0, prefixLength)).toEqual(
        deriveResearchCaptureAids(rawFrames.slice(0, prefixLength)).frames,
      );
    }
    expect(rawFrames).toEqual(snapshot);
    expect(numericValues(full).every(Number.isFinite)).toBe(true);
  });
});
