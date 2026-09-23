import type { PoseLandmark } from "../motion/types";
import type { ResearchPoseFrame } from "./capture-bundle";
import type { CaptureInterval } from "./capture-timing";
import type { FrozenObservableProtocol } from "./protocol";

export const EXPLORATORY_PROTOCOL_VISIBILITY_THRESHOLDS = [
  0.55,
  0.65,
  0.75,
] as const;
export const EXPLORATORY_MINIMUM_ELIGIBLE_FRAMES = 5;

export interface ProtocolExecutorPoseFrame extends ResearchPoseFrame {
  readonly callbackMonotonicTimeMs: number;
  readonly videoTiming?: {
    readonly strategy: "nearest-latest-causal-video-frame";
    readonly sequence: number;
    readonly callbackMonotonicTimeMs: number;
    readonly mediaTimeMs: number;
    readonly presentedFrames?: number;
  };
}

export interface ProtocolAnglePeak {
  readonly threshold: number;
  readonly eligibleFrames: number;
  readonly peakDegrees: number;
  readonly evidenceSequenceNumber: number;
  readonly evidenceSessionTimeMs: number;
  readonly evidenceMediaTimeMs: number;
}

export interface ProtocolSensitivityResult {
  readonly threshold: number;
  readonly reference: ProtocolAnglePeak;
  readonly attempt: ProtocolAnglePeak;
  readonly targetMinusObservedDegrees: number;
  readonly directionSign: -1 | 0 | 1;
}

export interface ObservableProtocolExecution {
  readonly schemaVersion: "relay.observable-protocol-execution/1";
  readonly evidenceClass: "DERIVED";
  readonly interpretation:
    | "reference-relative-observable-geometry-not-anatomical-or-clinical-truth";
  readonly protocolId: string;
  readonly protocolVersion: string;
  readonly protocolSha256: string;
  readonly metricId: string;
  readonly status: "observable" | "unobservable";
  readonly abstainReason?: string;
  readonly primaryThreshold: 0.65;
  readonly sensitivity: readonly ProtocolSensitivityResult[];
  readonly sensitivityDirectionStable: boolean;
  readonly referencePeakDegrees?: number;
  readonly attemptPeakDegrees?: number;
  readonly targetMinusObservedDegrees?: number;
  readonly direction?:
    | "raise-left-arm-farther-toward-reference"
    | "lower-left-arm-toward-reference"
    | "matched-reference-peak";
  readonly evidence?: {
    readonly referenceSequenceNumber: number;
    readonly referenceSessionTimeMs: number;
    readonly referenceMediaTimeMs: number;
    readonly attemptSequenceNumber: number;
    readonly attemptSessionTimeMs: number;
    readonly attemptMediaTimeMs: number;
  };
  readonly claimBoundary: readonly string[];
}

function finiteLandmark(
  value: PoseLandmark | undefined,
  threshold: number,
): value is PoseLandmark {
  return Boolean(
    value &&
      Number.isFinite(value.x) &&
      Number.isFinite(value.y) &&
      Number.isFinite(value.z) &&
      Number.isFinite(value.visibility) &&
      Number.isFinite(value.presence) &&
      value.visibility >= threshold &&
      value.presence >= threshold,
  );
}

function leftShoulderAngleDegrees(
  frame: ProtocolExecutorPoseFrame,
  threshold: number,
): number | null {
  const hip = frame.image["left-hip"];
  const shoulder = frame.image["left-shoulder"];
  const elbow = frame.image["left-elbow"];
  if (
    !finiteLandmark(hip, threshold) ||
    !finiteLandmark(shoulder, threshold) ||
    !finiteLandmark(elbow, threshold) ||
    !Number.isFinite(frame.frameWidth) ||
    !Number.isFinite(frame.frameHeight) ||
    frame.frameWidth <= 0 ||
    frame.frameHeight <= 0
  ) {
    return null;
  }

  const firstX = (hip.x - shoulder.x) * frame.frameWidth;
  const firstY = (hip.y - shoulder.y) * frame.frameHeight;
  const secondX = (elbow.x - shoulder.x) * frame.frameWidth;
  const secondY = (elbow.y - shoulder.y) * frame.frameHeight;
  const firstLength = Math.hypot(firstX, firstY);
  const secondLength = Math.hypot(secondX, secondY);
  if (
    !Number.isFinite(firstLength) ||
    !Number.isFinite(secondLength) ||
    firstLength === 0 ||
    secondLength === 0
  ) {
    return null;
  }
  const cosine =
    (firstX * secondX + firstY * secondY) / (firstLength * secondLength);
  if (!Number.isFinite(cosine)) return null;
  const angle = Math.acos(Math.max(-1, Math.min(1, cosine))) * (180 / Math.PI);
  return Number.isFinite(angle) ? angle : null;
}

function intervalPeak(
  frames: readonly ProtocolExecutorPoseFrame[],
  interval: CaptureInterval,
  threshold: number,
): ProtocolAnglePeak | null {
  const eligible: Array<{
    frame: ProtocolExecutorPoseFrame;
    angle: number;
    mediaTimeMs: number;
  }> = [];
  for (const frame of frames) {
    const mediaTimeMs = frame.videoTiming?.mediaTimeMs;
    if (
      mediaTimeMs === undefined ||
      !Number.isFinite(mediaTimeMs) ||
      frame.callbackMonotonicTimeMs < interval.begin.callbackMonotonicTimeMs ||
      frame.callbackMonotonicTimeMs > interval.end.callbackMonotonicTimeMs ||
      mediaTimeMs < interval.begin.mediaTimeMs ||
      mediaTimeMs > interval.end.mediaTimeMs
    ) {
      continue;
    }
    const angle = leftShoulderAngleDegrees(frame, threshold);
    if (angle === null) continue;
    eligible.push({ frame, angle, mediaTimeMs });
  }
  if (eligible.length < EXPLORATORY_MINIMUM_ELIGIBLE_FRAMES) return null;
  const best = eligible.reduce((left, right) =>
    right.angle > left.angle ? right : left,
  );
  return {
    threshold,
    eligibleFrames: eligible.length,
    peakDegrees: best.angle,
    evidenceSequenceNumber: best.frame.sequenceNumber,
    evidenceSessionTimeMs: best.frame.tMs,
    evidenceMediaTimeMs: best.mediaTimeMs,
  };
}

function sign(value: number): -1 | 0 | 1 {
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}

function unobservable(
  protocol: FrozenObservableProtocol,
  reason: string,
  sensitivity: readonly ProtocolSensitivityResult[] = [],
): ObservableProtocolExecution {
  return {
    schemaVersion: "relay.observable-protocol-execution/1",
    evidenceClass: "DERIVED",
    interpretation:
      "reference-relative-observable-geometry-not-anatomical-or-clinical-truth",
    protocolId: protocol.protocol.protocolId,
    protocolVersion: protocol.protocol.protocolVersion,
    protocolSha256: protocol.sha256,
    metricId: protocol.protocol.target.metricId,
    status: "unobservable",
    abstainReason: reason,
    primaryThreshold: 0.65,
    sensitivity,
    sensitivityDirectionStable: false,
    claimBoundary: CLAIM_BOUNDARY,
  };
}

const CLAIM_BOUNDARY = [
  "The captured reference is a comparator, not an ideal, safe, or clinical target.",
  "Angles describe the retained estimator output in this camera view, not anatomical ground truth.",
  "Pain, load, force, contact, intent, diagnosis, treatment, and safety are not observable here.",
] as const;

/**
 * Executes the single frozen pilot metric. This deliberately rejects any other
 * protocol rather than pretending the formula is a general protocol engine.
 */
export function executeObservableProtocol(
  protocol: FrozenObservableProtocol,
  frames: readonly ProtocolExecutorPoseFrame[],
  intervals: readonly CaptureInterval[],
): ObservableProtocolExecution {
  const target = protocol.protocol.target;
  if (
    protocol.protocol.protocolId !==
      "relay.lateral-arm-raise.reference-relative" ||
    target.metricId !== "left-shoulder-lateral-raise-included-angle" ||
    target.coordinateSource !== "image-2d" ||
    target.aggregation !== "maximum-over-interval" ||
    target.directionConvention !== "target-minus-observed" ||
    target.referenceBinding !==
      "reference-interval-from-shared-continuous-session"
  ) {
    return unobservable(protocol, "The supplied protocol is not supported by this frozen executor.");
  }
  const reference = intervals.find((interval) => interval.kind === "reference");
  const attempt = intervals.find((interval) => interval.kind === "attempt");
  if (!reference || !attempt) {
    return unobservable(protocol, "Reference and attempt intervals are both required.");
  }

  const sensitivity: ProtocolSensitivityResult[] = [];
  for (const threshold of EXPLORATORY_PROTOCOL_VISIBILITY_THRESHOLDS) {
    const referencePeak = intervalPeak(frames, reference, threshold);
    const attemptPeak = intervalPeak(frames, attempt, threshold);
    if (!referencePeak || !attemptPeak) {
      return unobservable(
        protocol,
        `At least ${EXPLORATORY_MINIMUM_ELIGIBLE_FRAMES} eligible frames are required in each interval at every declared visibility threshold.`,
        sensitivity,
      );
    }
    const delta = referencePeak.peakDegrees - attemptPeak.peakDegrees;
    sensitivity.push({
      threshold,
      reference: referencePeak,
      attempt: attemptPeak,
      targetMinusObservedDegrees: delta,
      directionSign: sign(delta),
    });
  }

  const directionStable = sensitivity.every(
    (entry) => entry.directionSign === sensitivity[0].directionSign,
  );
  if (!directionStable) {
    return unobservable(
      protocol,
      "The reference-relative direction changes across the declared confidence sensitivity analysis.",
      sensitivity,
    );
  }
  const primary = sensitivity.find((entry) => entry.threshold === 0.65);
  if (!primary) {
    return unobservable(protocol, "The declared primary confidence threshold is missing.", sensitivity);
  }

  return {
    schemaVersion: "relay.observable-protocol-execution/1",
    evidenceClass: "DERIVED",
    interpretation:
      "reference-relative-observable-geometry-not-anatomical-or-clinical-truth",
    protocolId: protocol.protocol.protocolId,
    protocolVersion: protocol.protocol.protocolVersion,
    protocolSha256: protocol.sha256,
    metricId: target.metricId,
    status: "observable",
    primaryThreshold: 0.65,
    sensitivity,
    sensitivityDirectionStable: true,
    referencePeakDegrees: primary.reference.peakDegrees,
    attemptPeakDegrees: primary.attempt.peakDegrees,
    targetMinusObservedDegrees: primary.targetMinusObservedDegrees,
    direction:
      primary.directionSign > 0
        ? "raise-left-arm-farther-toward-reference"
        : primary.directionSign < 0
          ? "lower-left-arm-toward-reference"
          : "matched-reference-peak",
    evidence: {
      referenceSequenceNumber: primary.reference.evidenceSequenceNumber,
      referenceSessionTimeMs: primary.reference.evidenceSessionTimeMs,
      referenceMediaTimeMs: primary.reference.evidenceMediaTimeMs,
      attemptSequenceNumber: primary.attempt.evidenceSequenceNumber,
      attemptSessionTimeMs: primary.attempt.evidenceSessionTimeMs,
      attemptMediaTimeMs: primary.attempt.evidenceMediaTimeMs,
    },
    claimBoundary: CLAIM_BOUNDARY,
  };
}
