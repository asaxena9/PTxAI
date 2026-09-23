import { describe, expect, it } from "vitest";
import type { LandmarkSet } from "../src/motion/types";
import {
  executeObservableProtocol,
  type ProtocolExecutorPoseFrame,
} from "../src/research/protocol-executor";
import type {
  CaptureInterval,
  VideoFrameTimingSample,
} from "../src/research/capture-timing";
import {
  freezeObservableProtocol,
  LATERAL_ARM_RAISE_REFERENCE_V1,
} from "../src/research/protocol";

function timing(sequence: number, mediaTimeMs: number): VideoFrameTimingSample {
  return {
    sequence,
    callbackMonotonicTimeMs: 1000 + mediaTimeMs,
    mediaTimeMs,
    presentedFrames: sequence,
  };
}

function interval(
  kind: "reference" | "attempt",
  begin: number,
  end: number,
): CaptureInterval {
  return { kind, begin: timing(begin, begin * 100), end: timing(end, end * 100) };
}

function poseForAngle(degrees: number, confidence = 1): LandmarkSet {
  const radians = degrees * (Math.PI / 180);
  return {
    "left-shoulder": {
      x: 0.5,
      y: 0.5,
      z: 0,
      visibility: confidence,
      presence: confidence,
    },
    "left-hip": {
      x: 0.5,
      y: 0.8,
      z: 0,
      visibility: confidence,
      presence: confidence,
    },
    "left-elbow": {
      x: 0.5 + 0.25 * Math.sin(radians),
      y: 0.5 + 0.25 * Math.cos(radians),
      z: 0,
      visibility: confidence,
      presence: confidence,
    },
  };
}

function frame(
  sequenceNumber: number,
  angle: number,
  confidence = 1,
): ProtocolExecutorPoseFrame {
  const sample = timing(sequenceNumber, sequenceNumber * 100);
  return {
    sequenceNumber,
    tMs: sequenceNumber * 100,
    sourceTimestampMs: 1000 + sequenceNumber * 100,
    callbackMonotonicTimeMs: 1000 + sequenceNumber * 100,
    frameWidth: 100,
    frameHeight: 100,
    poseDetected: true,
    imageJointCount: 3,
    worldJointCount: 0,
    image: poseForAngle(angle, confidence),
    world: {},
    videoTiming: {
      strategy: "nearest-latest-causal-video-frame",
      ...sample,
    },
  };
}

describe("frozen observable protocol executor", () => {
  it("derives a stable reference-relative change with exact evidence frames", async () => {
    const protocol = await freezeObservableProtocol(
      LATERAL_ARM_RAISE_REFERENCE_V1,
    );
    const frames = [
      ...[1, 2, 3, 4, 5].map((value) => frame(value, 70 + value * 4)),
      ...[7, 8, 9, 10, 11].map((value) => frame(value, 35 + value * 2)),
    ];
    const result = executeObservableProtocol(protocol, frames, [
      interval("reference", 1, 5),
      interval("attempt", 7, 11),
    ]);

    expect(result).toMatchObject({
      status: "observable",
      direction: "raise-left-arm-farther-toward-reference",
      sensitivityDirectionStable: true,
      evidence: {
        referenceSequenceNumber: 5,
        attemptSequenceNumber: 11,
      },
    });
    expect(result.referencePeakDegrees).toBeCloseTo(90);
    expect(result.attemptPeakDegrees).toBeCloseTo(57);
    expect(result.targetMinusObservedDegrees).toBeCloseTo(33);
    expect(result.sensitivity).toHaveLength(3);
  });

  it("abstains when confidence support is insufficient", async () => {
    const protocol = await freezeObservableProtocol(
      LATERAL_ARM_RAISE_REFERENCE_V1,
    );
    const frames = [
      ...[1, 2, 3, 4, 5].map((value) => frame(value, 80, 0.7)),
      ...[7, 8, 9, 10, 11].map((value) => frame(value, 50, 0.7)),
    ];
    const result = executeObservableProtocol(protocol, frames, [
      interval("reference", 1, 5),
      interval("attempt", 7, 11),
    ]);
    expect(result.status).toBe("unobservable");
    expect(result.abstainReason).toMatch(/eligible frames/);
  });

  it("excludes a pose callback that arrived after the interval ended", async () => {
    const protocol = await freezeObservableProtocol(
      LATERAL_ARM_RAISE_REFERENCE_V1,
    );
    const lateReference = {
      ...frame(6, 170),
      callbackMonotonicTimeMs: timing(6, 600).callbackMonotonicTimeMs + 1,
    };
    const lateAttempt = {
      ...frame(13, 170),
      callbackMonotonicTimeMs: timing(13, 1300).callbackMonotonicTimeMs + 1,
    };
    const frames = [
      ...[1, 2, 3, 4, 5].map((value) => frame(value, 60 + value)),
      lateReference,
      ...[8, 9, 10, 11, 12].map((value) => frame(value, 40 + value)),
      lateAttempt,
    ];

    const result = executeObservableProtocol(protocol, frames, [
      interval("reference", 1, 6),
      interval("attempt", 8, 13),
    ]);

    expect(result.status).toBe("observable");
    expect(result.evidence).toMatchObject({
      referenceSequenceNumber: 5,
      attemptSequenceNumber: 12,
    });
  });

  it("refuses a different metric instead of applying the pilot formula", async () => {
    const changed = structuredClone(LATERAL_ARM_RAISE_REFERENCE_V1);
    changed.protocolId = "different-protocol";
    const protocol = await freezeObservableProtocol(changed);
    expect(executeObservableProtocol(protocol, [], []).status).toBe(
      "unobservable",
    );
  });
});
