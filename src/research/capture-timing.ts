/**
 * A source sample from requestVideoFrameCallback for one continuous video.
 * `mediaTimeMs` is expected to have been converted to milliseconds at capture;
 * these helpers never round, clamp, or otherwise rewrite source timestamps.
 */
export interface VideoFrameTimingSample {
  readonly sequence: number;
  readonly callbackMonotonicTimeMs: number;
  readonly mediaTimeMs: number;
  readonly presentedFrames?: number;
}

/** A pose observation timestamped on the same monotonic clock as video callbacks. */
export interface PoseTimingSample {
  readonly sequence: number;
  readonly callbackMonotonicTimeMs: number;
}

export interface PoseTimingAttachment<
  TPose extends PoseTimingSample = PoseTimingSample,
> {
  readonly strategy: "nearest-latest-causal-video-frame";
  readonly pose: TPose;
  readonly videoFrame: VideoFrameTimingSample;
}

export type CaptureIntervalKind = "reference" | "attempt" | "revision";

export interface OpenCaptureInterval {
  readonly kind: CaptureIntervalKind;
  readonly begin: VideoFrameTimingSample;
}

export interface CaptureInterval extends OpenCaptureInterval {
  readonly end: VideoFrameTimingSample;
}

export interface VideoFrameTimingExtent {
  readonly firstSequence: number;
  readonly lastSequence: number;
  readonly startCallbackMonotonicTimeMs: number;
  readonly endCallbackMonotonicTimeMs: number;
  readonly startMediaTimeMs: number;
  readonly endMediaTimeMs: number;
}

export type CaptureTimingIssueCode =
  | "missing-video-timing"
  | "invalid-sequence"
  | "invalid-time"
  | "backward-time"
  | "sequence-gap"
  | "presented-frame-coverage"
  | "uncovered-pose"
  | "pose-outside-timing-extent"
  | "interval-count"
  | "interval-order"
  | "interval-boundary-outside-timing-extent"
  | "nonpositive-interval"
  | "interval-overlap";

export interface CaptureTimingIssue {
  readonly code: CaptureTimingIssueCode;
  readonly path: string;
  readonly message: string;
}

export interface VideoFrameTimingCoverage {
  readonly sequence: "complete" | "incomplete";
  readonly presentedFrames: "complete" | "unavailable" | "incomplete";
}

export interface VideoFrameTimingValidation {
  readonly valid: boolean;
  readonly issues: readonly CaptureTimingIssue[];
  readonly extent: VideoFrameTimingExtent | null;
  readonly coverage: VideoFrameTimingCoverage;
}

export interface PoseTimingCoverage {
  readonly totalPoses: number;
  readonly causallyAttachedPoses: number;
  readonly posesWithinTimingExtent: number;
}

export interface PoseTimingCoverageValidation<
  TPose extends PoseTimingSample = PoseTimingSample,
> {
  readonly valid: boolean;
  readonly issues: readonly CaptureTimingIssue[];
  readonly attachments: readonly PoseTimingAttachment<TPose>[];
  readonly coverage: PoseTimingCoverage;
}

export interface CaptureIntervalValidation {
  readonly valid: boolean;
  readonly issues: readonly CaptureTimingIssue[];
  readonly extent: VideoFrameTimingExtent | null;
}

export interface ContinuousCaptureTimingSession<
  TPose extends PoseTimingSample = PoseTimingSample,
> {
  readonly videoFrames: readonly VideoFrameTimingSample[];
  readonly poses: readonly TPose[];
  readonly intervals: readonly CaptureInterval[];
}

export interface CaptureTimingSessionValidation<
  TPose extends PoseTimingSample = PoseTimingSample,
> {
  readonly valid: boolean;
  readonly issues: readonly CaptureTimingIssue[];
  readonly videoTiming: VideoFrameTimingValidation;
  readonly poseTiming: PoseTimingCoverageValidation<TPose>;
  readonly intervalTiming: CaptureIntervalValidation;
}

const INTERVAL_ORDER = ["reference", "attempt", "revision"] as const;

function issue(
  code: CaptureTimingIssueCode,
  path: string,
  message: string,
): CaptureTimingIssue {
  return { code, path, message };
}

function validSequence(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0;
}

function validTime(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

function videoSampleIssues(
  sample: VideoFrameTimingSample,
  path: string,
): CaptureTimingIssue[] {
  const issues: CaptureTimingIssue[] = [];
  if (!validSequence(sample.sequence)) {
    issues.push(
      issue(
        "invalid-sequence",
        `${path}.sequence`,
        "Video-frame sequence must be a non-negative safe integer.",
      ),
    );
  }
  if (!validTime(sample.callbackMonotonicTimeMs)) {
    issues.push(
      issue(
        "invalid-time",
        `${path}.callbackMonotonicTimeMs`,
        "Video callback time must be finite and non-negative.",
      ),
    );
  }
  if (!validTime(sample.mediaTimeMs)) {
    issues.push(
      issue(
        "invalid-time",
        `${path}.mediaTimeMs`,
        "Video media time must be finite and non-negative.",
      ),
    );
  }
  if (
    sample.presentedFrames !== undefined &&
    !validSequence(sample.presentedFrames)
  ) {
    issues.push(
      issue(
        "invalid-sequence",
        `${path}.presentedFrames`,
        "Presented-frame count must be a non-negative safe integer when supplied.",
      ),
    );
  }
  return issues;
}

function poseSampleIssues(
  sample: PoseTimingSample,
  path: string,
): CaptureTimingIssue[] {
  const issues: CaptureTimingIssue[] = [];
  if (!validSequence(sample.sequence)) {
    issues.push(
      issue(
        "invalid-sequence",
        `${path}.sequence`,
        "Pose sequence must be a non-negative safe integer.",
      ),
    );
  }
  if (!validTime(sample.callbackMonotonicTimeMs)) {
    issues.push(
      issue(
        "invalid-time",
        `${path}.callbackMonotonicTimeMs`,
        "Pose callback time must be finite and non-negative.",
      ),
    );
  }
  return issues;
}

/** Validate one ordered timing trace and report whether optional frame coverage is complete. */
export function validateVideoFrameTiming(
  samples: readonly VideoFrameTimingSample[],
): VideoFrameTimingValidation {
  const issues: CaptureTimingIssue[] = [];
  if (samples.length === 0) {
    issues.push(
      issue(
        "missing-video-timing",
        "videoFrames",
        "At least one video-frame timing sample is required.",
      ),
    );
    return {
      valid: false,
      issues,
      extent: null,
      coverage: { sequence: "incomplete", presentedFrames: "unavailable" },
    };
  }

  samples.forEach((sample, index) => {
    issues.push(...videoSampleIssues(sample, `videoFrames[${index}]`));
  });

  let sequenceComplete = samples.every((sample) =>
    validSequence(sample.sequence),
  );
  for (let index = 1; index < samples.length; index += 1) {
    const previous = samples[index - 1];
    const current = samples[index];

    if (
      validSequence(previous.sequence) &&
      validSequence(current.sequence) &&
      current.sequence !== previous.sequence + 1
    ) {
      sequenceComplete = false;
      issues.push(
        issue(
          current.sequence <= previous.sequence
            ? "invalid-sequence"
            : "sequence-gap",
          `videoFrames[${index}].sequence`,
          current.sequence <= previous.sequence
            ? "Video-frame sequence must increase strictly."
            : "Video-frame sequence must be contiguous within the retained trace.",
        ),
      );
    }
    if (
      validTime(previous.callbackMonotonicTimeMs) &&
      validTime(current.callbackMonotonicTimeMs) &&
      current.callbackMonotonicTimeMs < previous.callbackMonotonicTimeMs
    ) {
      issues.push(
        issue(
          "backward-time",
          `videoFrames[${index}].callbackMonotonicTimeMs`,
          "Video callback time moved backward.",
        ),
      );
    }
    if (
      validTime(previous.mediaTimeMs) &&
      validTime(current.mediaTimeMs) &&
      current.mediaTimeMs < previous.mediaTimeMs
    ) {
      issues.push(
        issue(
          "backward-time",
          `videoFrames[${index}].mediaTimeMs`,
          "Video media time moved backward.",
        ),
      );
    }
  }

  const suppliedPresentedFrames = samples.filter(
    (sample) => sample.presentedFrames !== undefined,
  ).length;
  let presentedFrameCoverage: VideoFrameTimingCoverage["presentedFrames"] =
    suppliedPresentedFrames === 0
      ? "unavailable"
      : samples.every(
            (sample) =>
              sample.presentedFrames !== undefined &&
              validSequence(sample.presentedFrames),
          )
        ? "complete"
        : "incomplete";

  if (
    suppliedPresentedFrames > 0 &&
    suppliedPresentedFrames !== samples.length
  ) {
    presentedFrameCoverage = "incomplete";
    issues.push(
      issue(
        "presented-frame-coverage",
        "videoFrames",
        "Presented-frame counts must be supplied for every sample or for none.",
      ),
    );
  } else if (suppliedPresentedFrames === samples.length) {
    for (let index = 1; index < samples.length; index += 1) {
      const previous = samples[index - 1];
      const current = samples[index];
      if (
        !validSequence(previous.presentedFrames as number) ||
        !validSequence(current.presentedFrames as number) ||
        !validSequence(previous.sequence) ||
        !validSequence(current.sequence)
      ) {
        presentedFrameCoverage = "incomplete";
        continue;
      }
      const presentedDelta =
        (current.presentedFrames as number) -
        (previous.presentedFrames as number);
      const sequenceDelta = current.sequence - previous.sequence;
      if (presentedDelta !== sequenceDelta || presentedDelta <= 0) {
        presentedFrameCoverage = "incomplete";
        issues.push(
          issue(
            "presented-frame-coverage",
            `videoFrames[${index}].presentedFrames`,
            presentedDelta <= 0
              ? "Presented-frame count must increase strictly."
              : "Presented-frame count indicates an unretained video-frame callback.",
          ),
        );
      }
    }
  }

  const valid = issues.length === 0;
  const first = samples[0];
  const last = samples[samples.length - 1];
  const extent: VideoFrameTimingExtent | null = valid
    ? {
        firstSequence: first.sequence,
        lastSequence: last.sequence,
        startCallbackMonotonicTimeMs: first.callbackMonotonicTimeMs,
        endCallbackMonotonicTimeMs: last.callbackMonotonicTimeMs,
        startMediaTimeMs: first.mediaTimeMs,
        endMediaTimeMs: last.mediaTimeMs,
      }
    : null;

  return {
    valid,
    issues,
    extent,
    coverage: {
      sequence: sequenceComplete ? "complete" : "incomplete",
      presentedFrames: presentedFrameCoverage,
    },
  };
}

function validatePoseSequence(
  poses: readonly PoseTimingSample[],
): CaptureTimingIssue[] {
  const issues: CaptureTimingIssue[] = [];
  poses.forEach((pose, index) => {
    issues.push(...poseSampleIssues(pose, `poses[${index}]`));
  });

  for (let index = 1; index < poses.length; index += 1) {
    const previous = poses[index - 1];
    const current = poses[index];
    if (
      validSequence(previous.sequence) &&
      validSequence(current.sequence) &&
      current.sequence !== previous.sequence + 1
    ) {
      issues.push(
        issue(
          current.sequence <= previous.sequence
            ? "invalid-sequence"
            : "sequence-gap",
          `poses[${index}].sequence`,
          current.sequence <= previous.sequence
            ? "Pose sequence must increase strictly."
            : "Pose sequence must be contiguous within the retained trace.",
        ),
      );
    }
    if (
      validTime(previous.callbackMonotonicTimeMs) &&
      validTime(current.callbackMonotonicTimeMs) &&
      current.callbackMonotonicTimeMs < previous.callbackMonotonicTimeMs
    ) {
      issues.push(
        issue(
          "backward-time",
          `poses[${index}].callbackMonotonicTimeMs`,
          "Pose callback time moved backward.",
        ),
      );
    }
  }
  return issues;
}

function latestCausalVideoFrame(
  poseMonotonicTimeMs: number,
  samples: readonly VideoFrameTimingSample[],
): VideoFrameTimingSample | null {
  let lower = 0;
  let upper = samples.length - 1;
  let match: VideoFrameTimingSample | null = null;

  while (lower <= upper) {
    const middle = lower + Math.floor((upper - lower) / 2);
    const candidate = samples[middle];
    if (candidate.callbackMonotonicTimeMs <= poseMonotonicTimeMs) {
      match = candidate;
      lower = middle + 1;
    } else {
      upper = middle - 1;
    }
  }
  return match;
}

/**
 * Attach a pose to the latest retained video callback no later than that pose.
 * Invalid traces and poses before the first causal video sample return null;
 * the function never extrapolates from a future video frame.
 */
export function attachPoseTiming<TPose extends PoseTimingSample>(
  pose: TPose,
  videoFrames: readonly VideoFrameTimingSample[],
): PoseTimingAttachment<TPose> | null {
  if (poseSampleIssues(pose, "pose").length > 0) return null;
  if (!validateVideoFrameTiming(videoFrames).valid) return null;
  const videoFrame = latestCausalVideoFrame(
    pose.callbackMonotonicTimeMs,
    videoFrames,
  );
  if (!videoFrame) return null;
  return {
    strategy: "nearest-latest-causal-video-frame",
    pose,
    videoFrame,
  };
}

/** Validate pose monotonicity plus causal and final-extent coverage. */
export function validatePoseTimingCoverage<TPose extends PoseTimingSample>(
  poses: readonly TPose[],
  videoFrames: readonly VideoFrameTimingSample[],
): PoseTimingCoverageValidation<TPose> {
  const videoValidation = validateVideoFrameTiming(videoFrames);
  const issues = [
    ...videoValidation.issues,
    ...validatePoseSequence(poses),
  ];
  const attachments: PoseTimingAttachment<TPose>[] = [];

  if (!videoValidation.extent || issues.length > 0) {
    return {
      valid: false,
      issues,
      attachments,
      coverage: {
        totalPoses: poses.length,
        causallyAttachedPoses: 0,
        posesWithinTimingExtent: 0,
      },
    };
  }

  let posesWithinTimingExtent = 0;
  for (let index = 0; index < poses.length; index += 1) {
    const pose = poses[index];
    const videoFrame = latestCausalVideoFrame(
      pose.callbackMonotonicTimeMs,
      videoFrames,
    );
    if (!videoFrame) {
      issues.push(
        issue(
          "uncovered-pose",
          `poses[${index}].callbackMonotonicTimeMs`,
          "Pose precedes the first causal video-frame timing sample.",
        ),
      );
      continue;
    }

    attachments.push({
      strategy: "nearest-latest-causal-video-frame",
      pose,
      videoFrame,
    });
    if (
      pose.callbackMonotonicTimeMs <=
      videoValidation.extent.endCallbackMonotonicTimeMs
    ) {
      posesWithinTimingExtent += 1;
    } else {
      issues.push(
        issue(
          "pose-outside-timing-extent",
          `poses[${index}].callbackMonotonicTimeMs`,
          "Pose follows the final retained video-frame timing sample.",
        ),
      );
    }
  }

  return {
    valid: issues.length === 0,
    issues,
    attachments,
    coverage: {
      totalPoses: poses.length,
      causallyAttachedPoses: attachments.length,
      posesWithinTimingExtent,
    },
  };
}

/** Begin an interval at an exact sample from the shared video-frame clock. */
export function beginCaptureInterval(
  kind: CaptureIntervalKind,
  begin: VideoFrameTimingSample,
): OpenCaptureInterval | null {
  if (!INTERVAL_ORDER.includes(kind)) return null;
  if (videoSampleIssues(begin, "begin").length > 0) return null;
  return { kind, begin };
}

/** End an interval without manufacturing a duration or timestamp. */
export function endCaptureInterval(
  open: OpenCaptureInterval,
  end: VideoFrameTimingSample,
): CaptureInterval | null {
  if (!INTERVAL_ORDER.includes(open.kind)) return null;
  if (videoSampleIssues(open.begin, "begin").length > 0) return null;
  if (videoSampleIssues(end, "end").length > 0) return null;
  if (
    end.sequence <= open.begin.sequence ||
    end.callbackMonotonicTimeMs <= open.begin.callbackMonotonicTimeMs ||
    end.mediaTimeMs <= open.begin.mediaTimeMs
  ) {
    return null;
  }
  return { ...open, end };
}

function sameVideoSample(
  left: VideoFrameTimingSample,
  right: VideoFrameTimingSample,
): boolean {
  return (
    left.sequence === right.sequence &&
    left.callbackMonotonicTimeMs === right.callbackMonotonicTimeMs &&
    left.mediaTimeMs === right.mediaTimeMs &&
    left.presentedFrames === right.presentedFrames
  );
}

function timelineIndexOf(
  boundary: VideoFrameTimingSample,
  videoFrames: readonly VideoFrameTimingSample[],
): number {
  const index = videoFrames.findIndex(
    (sample) => sample.sequence === boundary.sequence,
  );
  if (index < 0 || !sameVideoSample(boundary, videoFrames[index])) return -1;
  return index;
}

/**
 * Validate the final reference/attempt/(optional revision) segmentation against
 * the one retained video-frame timing extent.
 */
export function validateCaptureIntervals(
  intervals: readonly CaptureInterval[],
  videoFrames: readonly VideoFrameTimingSample[],
): CaptureIntervalValidation {
  const videoValidation = validateVideoFrameTiming(videoFrames);
  const issues = [...videoValidation.issues];
  if (!videoValidation.extent) {
    return { valid: false, issues, extent: null };
  }

  if (intervals.length !== 2 && intervals.length !== 3) {
    issues.push(
      issue(
        "interval-count",
        "intervals",
        "A final session requires reference and attempt intervals, plus at most one revision.",
      ),
    );
  }

  const bounds: Array<{ begin: number; end: number } | null> = [];
  intervals.forEach((interval, index) => {
    const expectedKind = INTERVAL_ORDER[index];
    if (interval.kind !== expectedKind) {
      issues.push(
        issue(
          "interval-order",
          `intervals[${index}].kind`,
          `Expected ${expectedKind ?? "no additional"} interval at this position.`,
        ),
      );
    }

    const beginIndex = timelineIndexOf(interval.begin, videoFrames);
    const endIndex = timelineIndexOf(interval.end, videoFrames);
    if (beginIndex < 0) {
      issues.push(
        issue(
          "interval-boundary-outside-timing-extent",
          `intervals[${index}].begin`,
          "Interval begin is not an exact sample from this video timing trace.",
        ),
      );
    }
    if (endIndex < 0) {
      issues.push(
        issue(
          "interval-boundary-outside-timing-extent",
          `intervals[${index}].end`,
          "Interval end is not an exact sample from this video timing trace.",
        ),
      );
    }

    if (
      beginIndex >= 0 &&
      endIndex >= 0 &&
      (endIndex <= beginIndex ||
        interval.end.callbackMonotonicTimeMs <=
          interval.begin.callbackMonotonicTimeMs ||
        interval.end.mediaTimeMs <= interval.begin.mediaTimeMs)
    ) {
      issues.push(
        issue(
          "nonpositive-interval",
          `intervals[${index}]`,
          "Interval end must follow its begin on sequence, callback, and media clocks.",
        ),
      );
    }
    bounds.push(
      beginIndex >= 0 && endIndex >= 0
        ? { begin: beginIndex, end: endIndex }
        : null,
    );
  });

  for (let index = 1; index < intervals.length; index += 1) {
    const previousBounds = bounds[index - 1];
    const currentBounds = bounds[index];
    if (
      previousBounds &&
      currentBounds &&
      previousBounds.end >= currentBounds.begin
    ) {
      issues.push(
        issue(
          "interval-overlap",
          `intervals[${index}]`,
          "Capture intervals must be chronological and separated by at least one retained video frame; a boundary sample cannot belong to both intervals.",
        ),
      );
    }
  }

  return {
    valid: issues.length === 0,
    issues,
    extent: videoValidation.extent,
  };
}

function uniqueIssues(
  issues: readonly CaptureTimingIssue[],
): CaptureTimingIssue[] {
  const seen = new Set<string>();
  return issues.filter((entry) => {
    const key = `${entry.code}:${entry.path}:${entry.message}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/** Validate every timing component of a single continuous same-clock session. */
export function validateCaptureTimingSession<
  TPose extends PoseTimingSample,
>(
  session: ContinuousCaptureTimingSession<TPose>,
): CaptureTimingSessionValidation<TPose> {
  const videoTiming = validateVideoFrameTiming(session.videoFrames);
  const poseTiming = validatePoseTimingCoverage(
    session.poses,
    session.videoFrames,
  );
  const intervalTiming = validateCaptureIntervals(
    session.intervals,
    session.videoFrames,
  );
  const issues = uniqueIssues([
    ...videoTiming.issues,
    ...poseTiming.issues,
    ...intervalTiming.issues,
  ]);
  return {
    valid:
      videoTiming.valid && poseTiming.valid && intervalTiming.valid,
    issues,
    videoTiming,
    poseTiming,
    intervalTiming,
  };
}
