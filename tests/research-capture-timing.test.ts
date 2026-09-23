import { describe, expect, it } from "vitest";
import {
  attachPoseTiming,
  beginCaptureInterval,
  endCaptureInterval,
  validateCaptureIntervals,
  validateCaptureTimingSession,
  validatePoseTimingCoverage,
  validateVideoFrameTiming,
  type CaptureInterval,
  type PoseTimingSample,
  type VideoFrameTimingSample,
} from "../src/research/capture-timing";

function videoFrame(
  sequence: number,
  callbackMonotonicTimeMs: number,
  mediaTimeMs: number,
  presentedFrames: number | null = sequence + 10,
): VideoFrameTimingSample {
  return {
    sequence,
    callbackMonotonicTimeMs,
    mediaTimeMs,
    ...(presentedFrames === null ? {} : { presentedFrames }),
  };
}

function completeIntervals(
  frames: readonly VideoFrameTimingSample[],
): CaptureInterval[] {
  const reference = beginCaptureInterval("reference", frames[0]);
  const attempt = beginCaptureInterval("attempt", frames[2]);
  if (!reference || !attempt) throw new Error("Expected valid interval starts");
  const completedReference = endCaptureInterval(reference, frames[1]);
  const completedAttempt = endCaptureInterval(attempt, frames[4]);
  if (!completedReference || !completedAttempt) {
    throw new Error("Expected valid interval ends");
  }
  return [completedReference, completedAttempt];
}

describe("continuous research capture timing", () => {
  it("validates a monotonic shared-clock session with complete coverage", () => {
    const videoFrames = [
      videoFrame(0, 100.125, 0.25),
      videoFrame(1, 110.5, 10.625),
      videoFrame(2, 120.75, 20.875),
      videoFrame(3, 131.25, 31.375),
      videoFrame(4, 141.5, 41.625),
    ];
    const poses: PoseTimingSample[] = [
      { sequence: 0, callbackMonotonicTimeMs: 105.375 },
      { sequence: 1, callbackMonotonicTimeMs: 125.875 },
    ];

    const result = validateCaptureTimingSession({
      videoFrames,
      poses,
      intervals: completeIntervals(videoFrames),
    });

    expect(result.valid).toBe(true);
    expect(result.issues).toEqual([]);
    expect(result.videoTiming).toMatchObject({
      valid: true,
      extent: {
        firstSequence: 0,
        lastSequence: 4,
        startCallbackMonotonicTimeMs: 100.125,
        endCallbackMonotonicTimeMs: 141.5,
        startMediaTimeMs: 0.25,
        endMediaTimeMs: 41.625,
      },
      coverage: { sequence: "complete", presentedFrames: "complete" },
    });
    expect(result.poseTiming.coverage).toEqual({
      totalPoses: 2,
      causallyAttachedPoses: 2,
      posesWithinTimingExtent: 2,
    });
  });

  it("attaches to the nearest past sample, including an exact-time sample", () => {
    const videoFrames = [
      videoFrame(0, 10.125, 0.25, null),
      videoFrame(1, 20.5, 10.625, null),
      videoFrame(2, 30.875, 21, null),
    ];
    const between = {
      sequence: 7,
      callbackMonotonicTimeMs: 20.499,
      sourceTimestampMs: 987.654321,
    };
    const snapshot = structuredClone({ videoFrames, between });

    const attachedBetween = attachPoseTiming(between, videoFrames);
    const attachedExact = attachPoseTiming(
      { sequence: 8, callbackMonotonicTimeMs: 20.5 },
      videoFrames,
    );

    expect(attachedBetween?.videoFrame).toBe(videoFrames[0]);
    expect(attachedBetween?.pose.callbackMonotonicTimeMs).toBe(20.499);
    expect(attachedBetween?.pose.sourceTimestampMs).toBe(987.654321);
    expect(attachedExact?.videoFrame).toBe(videoFrames[1]);
    expect(
      attachedBetween!.videoFrame.callbackMonotonicTimeMs <=
        attachedBetween!.pose.callbackMonotonicTimeMs,
    ).toBe(true);
    expect(
      attachPoseTiming(
        { sequence: 6, callbackMonotonicTimeMs: 10 },
        videoFrames,
      ),
    ).toBeNull();
    expect({ videoFrames, between }).toEqual(snapshot);
  });

  it("is causally prefix-stable when later video samples arrive", () => {
    const videoFrames = [
      videoFrame(0, 100, 0, null),
      videoFrame(1, 120, 20, null),
      videoFrame(2, 140, 40, null),
      videoFrame(3, 160, 60, null),
    ];
    const pose = { sequence: 0, callbackMonotonicTimeMs: 139.75 };

    const beforeFutureFrames = attachPoseTiming(pose, videoFrames.slice(0, 2));
    const afterFutureFrames = attachPoseTiming(pose, videoFrames);

    expect(beforeFutureFrames).not.toBeNull();
    expect(afterFutureFrames).toEqual(beforeFutureFrames);
    expect(afterFutureFrames?.videoFrame.sequence).toBe(1);
  });

  it("fails closed on backward and nonfinite video or pose times", () => {
    const backwardVideo = [
      videoFrame(0, 10, 0),
      videoFrame(1, 9, 10),
    ];
    const nonfiniteVideo = [
      videoFrame(0, 10, 0),
      videoFrame(1, 20, Number.NaN),
    ];
    const validVideo = [
      videoFrame(0, 10, 0),
      videoFrame(1, 20, 10),
    ];

    expect(validateVideoFrameTiming(backwardVideo)).toMatchObject({
      valid: false,
      issues: expect.arrayContaining([
        expect.objectContaining({ code: "backward-time" }),
      ]),
    });
    expect(validateVideoFrameTiming(nonfiniteVideo)).toMatchObject({
      valid: false,
      extent: null,
      issues: expect.arrayContaining([
        expect.objectContaining({ code: "invalid-time" }),
      ]),
    });
    expect(
      attachPoseTiming(
        { sequence: 0, callbackMonotonicTimeMs: 15 },
        backwardVideo,
      ),
    ).toBeNull();
    expect(
      attachPoseTiming(
        { sequence: 0, callbackMonotonicTimeMs: Number.POSITIVE_INFINITY },
        validVideo,
      ),
    ).toBeNull();

    const poseValidation = validatePoseTimingCoverage(
      [
        { sequence: 0, callbackMonotonicTimeMs: 18 },
        { sequence: 1, callbackMonotonicTimeMs: 17 },
      ],
      validVideo,
    );
    expect(poseValidation.valid).toBe(false);
    expect(poseValidation.attachments).toEqual([]);
    expect(poseValidation.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "backward-time" }),
      ]),
    );
  });

  it("reports pose and presented-frame coverage gaps", () => {
    const incompleteVideo = [
      videoFrame(0, 10, 0, 40),
      videoFrame(1, 20, 10, 42),
    ];
    expect(validateVideoFrameTiming(incompleteVideo)).toMatchObject({
      valid: false,
      coverage: { presentedFrames: "incomplete" },
    });

    const videoFrames = [
      videoFrame(0, 10, 0, null),
      videoFrame(1, 20, 10, null),
      videoFrame(2, 30, 20, null),
    ];
    const result = validatePoseTimingCoverage(
      [
        { sequence: 0, callbackMonotonicTimeMs: 9 },
        { sequence: 1, callbackMonotonicTimeMs: 31 },
      ],
      videoFrames,
    );

    expect(result.valid).toBe(false);
    expect(result.coverage).toEqual({
      totalPoses: 2,
      causallyAttachedPoses: 1,
      posesWithinTimingExtent: 0,
    });
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "uncovered-pose" }),
        expect.objectContaining({ code: "pose-outside-timing-extent" }),
      ]),
    );
  });

  it("validates exact interval bounds, ordering, and non-overlap", () => {
    const videoFrames = [
      videoFrame(0, 100, 0),
      videoFrame(1, 110, 10),
      videoFrame(2, 120, 20),
      videoFrame(3, 130, 30),
      videoFrame(4, 140, 40),
    ];
    const validIntervals = completeIntervals(videoFrames);
    expect(validateCaptureIntervals(validIntervals, videoFrames).valid).toBe(
      true,
    );

    const overlapping: CaptureInterval[] = [
      { kind: "reference", begin: videoFrames[0], end: videoFrames[3] },
      { kind: "attempt", begin: videoFrames[2], end: videoFrames[4] },
    ];
    expect(validateCaptureIntervals(overlapping, videoFrames).issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "interval-overlap" }),
      ]),
    );

    const forgedBoundary = videoFrame(5, 150, 50);
    const outside: CaptureInterval[] = [
      { kind: "reference", begin: videoFrames[0], end: videoFrames[1] },
      { kind: "attempt", begin: videoFrames[2], end: forgedBoundary },
    ];
    expect(validateCaptureIntervals(outside, videoFrames).issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "interval-boundary-outside-timing-extent",
        }),
      ]),
    );

    const open = beginCaptureInterval("reference", videoFrames[2]);
    expect(open).not.toBeNull();
    expect(endCaptureInterval(open!, videoFrames[1])).toBeNull();
    expect(
      beginCaptureInterval(
        "reference",
        videoFrame(8, Number.NaN, 80),
      ),
    ).toBeNull();
  });
});
