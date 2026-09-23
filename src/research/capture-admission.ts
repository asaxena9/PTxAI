import type { JointName, LandmarkSet } from "../motion/types";
import {
  deriveResearchAnnotationReview,
  type ResearchAnnotationAdjudication,
  type ResearchAnnotationReviewEntry,
  type ResearchAnnotationSnapshot,
} from "./annotation-review";
import {
  deriveResearchCaptureAids,
  type ResearchCaptureAids,
} from "./capture-aids";
import {
  validateCaptureTimingSession,
  type CaptureInterval,
  type CaptureTimingSessionValidation,
  type VideoFrameTimingSample,
} from "./capture-timing";
import type { ObservableProtocolExecution } from "./protocol-executor";

export type CaptureAdmissionStatus =
  | "QUARANTINED"
  | "ALLOWED_MEASURED_PILOT_MECHANICS_ONLY";

/**
 * This is deliberately structural rather than coupled to the changing bundle
 * schema. One pose callback and requestVideoFrameCallback must share the same
 * monotonic clock; `tMs` remains the continuous-session-relative pose time used
 * by deterministic aids.
 */
export interface CaptureAdmissionPoseFrame {
  readonly sequenceNumber: number;
  readonly tMs: number;
  readonly sourceTimestampMs: number;
  readonly callbackMonotonicTimeMs: number;
  readonly frameWidth: number;
  readonly frameHeight: number;
  readonly poseDetected: boolean;
  readonly imageJointCount: number;
  readonly worldJointCount: number;
  readonly image: LandmarkSet;
  readonly world: LandmarkSet;
}

export interface CanonicalArtifactIdentity {
  /** The exact retained UTF-8 artifact text, not a pretty-printed surrogate. */
  readonly canonicalJson: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface RawPoseArtifactIdentity extends CanonicalArtifactIdentity {
  readonly frameCount: number;
}

export interface DerivedAidsArtifactIdentity extends CanonicalArtifactIdentity {
  readonly frameCount: number;
  readonly derivedFromRawSha256: string;
  readonly aids: ResearchCaptureAids;
}

export interface FrozenProtocolIdentity {
  readonly protocol: Readonly<{
    protocolId: string;
    protocolVersion: string;
    [key: string]: unknown;
  }>;
  readonly canonicalJson: string;
  readonly sha256: string;
}

export interface FrozenProtocolExecutionIdentity
  extends CanonicalArtifactIdentity {
  /** The exact retained deterministic executor output bound by its digest. */
  readonly execution: ObservableProtocolExecution;
}

export interface RuntimePoseModelIdentity {
  readonly modelName: string;
  readonly runtimeLibrary: string;
  readonly runtimeVersion: string;
  readonly assetPath: string;
  readonly assetBytes: number;
  readonly assetSha256: string;
  readonly realizedDelegate: string;
}

export interface CaptureAppEnvironmentIdentity {
  readonly appName: string;
  readonly appVersion: string;
  readonly appBuildSha256: string;
  readonly collectorSchemaVersion: string;
  readonly userAgent: string;
  readonly platform: string;
  readonly locale: string;
  readonly timeZone: string;
  readonly capturedAt: string;
}

export interface ContinuousVideoAssetIdentity {
  readonly fileName: string;
  readonly mimeType: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly captureTopology: "one-continuous-session";
}

export interface CaptureReplayAudit {
  readonly status: "passed";
  readonly auditedAt: string;
  readonly auditorCode: string;
  readonly rawPoseSha256: string;
  readonly derivedAidsSha256: string;
  readonly videoSha256?: string;
  readonly replayedPoseFrameCount: number;
  readonly continuousVideoDecoded: boolean;
  readonly poseOverlayRendered: boolean;
  readonly referenceAttemptIntervalsReviewed: boolean;
  readonly timingSessionRevalidated: boolean;
  /** Required only after measured source-to-encoded-blob timeline calibration. */
  readonly encodedReplayTimelineCalibrated: true;
}

export interface CaptureCollectionScope {
  readonly participantScope: "adult-researcher-self";
  readonly dataUseScope: "internal-methods-only";
  readonly recordingRightsConfirmed: true;
  readonly nonClinicalResearchAcknowledged: true;
  readonly selfAttestedAt: string;
}

export interface CaptureAdmissionReview {
  /** Review entries and optional adjudication are retained, never merged. */
  readonly storageMode: "append-only";
  readonly claimedFinalState: "adjudicated";
  readonly entries: readonly ResearchAnnotationReviewEntry[];
  readonly adjudication?: ResearchAnnotationAdjudication;
  readonly numericTolerance?: number;
}

export interface ContinuousCaptureAdmissionEpisode {
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly rawVideoEnabled: boolean;
  readonly declaredPoseFrameCount: number;
  readonly declaredVideoFrameTimingCount: number;
  readonly declaredIntervalCount: number;
  readonly rawPoseFrames: readonly CaptureAdmissionPoseFrame[];
  readonly videoFrameTimings: readonly VideoFrameTimingSample[];
  readonly intervals: readonly CaptureInterval[];
  readonly rawPoseArtifact: RawPoseArtifactIdentity;
  readonly derivedAidsArtifact: DerivedAidsArtifactIdentity;
  readonly frozenProtocol: FrozenProtocolIdentity;
  readonly protocolExecution: FrozenProtocolExecutionIdentity;
  readonly runtimePoseModel: RuntimePoseModelIdentity;
  readonly appEnvironment: CaptureAppEnvironmentIdentity;
  readonly video?: ContinuousVideoAssetIdentity;
  readonly replayAudit: CaptureReplayAudit;
  readonly collectionScope: CaptureCollectionScope;
  readonly review: CaptureAdmissionReview;
}

export type CaptureAdmissionGateId =
  | "episode-lineage"
  | "pose-integrity"
  | "exact-frame-counts"
  | "continuous-timing"
  | "raw-pose-artifact"
  | "derived-aids-artifact"
  | "immutable-protocol"
  | "immutable-protocol-execution"
  | "runtime-model-identity"
  | "app-environment-identity"
  | "continuous-video-identity"
  | "replay-audit"
  | "collection-scope"
  | "append-only-independent-review";

export interface CaptureAdmissionGateReport {
  readonly id: CaptureAdmissionGateId;
  readonly passed: boolean;
  readonly issues: readonly string[];
  readonly facts: Readonly<Record<string, string | number | boolean | null>>;
}

export interface CaptureAdmissionReport {
  readonly status: CaptureAdmissionStatus;
  readonly admitted: boolean;
  readonly evidenceClass:
    | "QUARANTINED_UNADMITTED"
    | "MEASURED_PILOT_MECHANICS_ONLY";
  readonly clinicalTruthStatus: "NOT_ESTABLISHED";
  readonly permittedUse:
    | "none-while-quarantined"
    | "bounded-nonclinical-mechanics-methods-pilot";
  readonly prohibitedClaims: readonly [
    "clinical correctness",
    "safety",
    "diagnosis or treatment",
    "pain, force, contact, load, or intent",
    "general body intelligence",
  ];
  readonly gates: readonly CaptureAdmissionGateReport[];
  readonly issues: readonly string[];
}

interface ComputedCanonicalArtifact {
  readonly canonicalJson: string;
  readonly bytes: number;
  readonly sha256: string;
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const KNOWN_JOINTS = new Set<JointName>([
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
]);

const PROHIBITED_CLAIMS = [
  "clinical correctness",
  "safety",
  "diagnosis or treatment",
  "pain, force, contact, load, or intent",
  "general body intelligence",
] as const;

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, stableValue(entry)]),
    );
  }
  return value;
}

/** Canonical form shared by the verifier and collector artifact writer. */
export function canonicalCaptureArtifactJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function sha256CaptureArtifactText(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return bytesToHex(new Uint8Array(digest));
}

export async function identifyCanonicalCaptureArtifact(
  value: unknown,
): Promise<ComputedCanonicalArtifact> {
  const canonicalJson = canonicalCaptureArtifactJson(value);
  return {
    canonicalJson,
    bytes: new TextEncoder().encode(canonicalJson).byteLength,
    sha256: await sha256CaptureArtifactText(canonicalJson),
  };
}

function finiteNonNegative(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

function positiveSafeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function nonNegativeSafeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0;
}

function isSha256(value: string): boolean {
  return SHA256_PATTERN.test(value);
}

function validTimestamp(value: string): boolean {
  return Boolean(value.trim()) && Number.isFinite(Date.parse(value));
}

function addRequiredTextIssue(
  issues: string[],
  value: string,
  path: string,
): void {
  if (!value.trim()) issues.push(`${path} is required.`);
}

function gate(
  id: CaptureAdmissionGateId,
  issues: readonly string[],
  facts: CaptureAdmissionGateReport["facts"] = {},
): CaptureAdmissionGateReport {
  return {
    id,
    passed: issues.length === 0,
    issues: Array.from(new Set(issues)),
    facts,
  };
}

function validateLineage(
  episode: ContinuousCaptureAdmissionEpisode,
): CaptureAdmissionGateReport {
  const issues: string[] = [];
  addRequiredTextIssue(issues, episode.episodeId, "episodeId");
  addRequiredTextIssue(
    issues,
    episode.episodeAncestorId,
    "episodeAncestorId",
  );
  for (const [index, entry] of episode.review.entries.entries()) {
    if (entry.episodeId !== episode.episodeId) {
      issues.push(`review.entries[${index}].episodeId does not match episodeId.`);
    }
    if (entry.episodeAncestorId !== episode.episodeAncestorId) {
      issues.push(
        `review.entries[${index}].episodeAncestorId does not match episodeAncestorId.`,
      );
    }
  }
  if (
    episode.review.adjudication &&
    episode.review.adjudication.episodeId !== episode.episodeId
  ) {
    issues.push("review.adjudication.episodeId does not match episodeId.");
  }
  if (
    episode.review.adjudication &&
    episode.review.adjudication.episodeAncestorId !== episode.episodeAncestorId
  ) {
    issues.push(
      "review.adjudication.episodeAncestorId does not match episodeAncestorId.",
    );
  }
  return gate("episode-lineage", issues, {
    reviewEntryCount: episode.review.entries.length,
    hasAdjudication: Boolean(episode.review.adjudication),
  });
}

function validateLandmarks(
  landmarks: LandmarkSet,
  path: string,
): string[] {
  const issues: string[] = [];
  for (const [joint, landmark] of Object.entries(landmarks)) {
    if (!KNOWN_JOINTS.has(joint as JointName)) {
      issues.push(`${path}.${joint} is not a recognized pose joint.`);
      continue;
    }
    if (!landmark || typeof landmark !== "object") {
      issues.push(`${path}.${joint} must be a landmark object.`);
      continue;
    }
    for (const coordinate of ["x", "y", "z"] as const) {
      if (!Number.isFinite(landmark[coordinate])) {
        issues.push(`${path}.${joint}.${coordinate} must be finite.`);
      }
    }
    for (const confidence of ["visibility", "presence"] as const) {
      const value = landmark[confidence];
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        issues.push(
          `${path}.${joint}.${confidence} must be finite and within [0,1].`,
        );
      }
    }
  }
  return issues;
}

function validatePoseIntegrity(
  frames: readonly CaptureAdmissionPoseFrame[],
  intervals: readonly CaptureInterval[],
): CaptureAdmissionGateReport {
  const issues: string[] = [];
  if (frames.length === 0) issues.push("At least one raw pose frame is required.");

  frames.forEach((frame, index) => {
    const path = `rawPoseFrames[${index}]`;
    if (!nonNegativeSafeInteger(frame.sequenceNumber)) {
      issues.push(`${path}.sequenceNumber must be a non-negative safe integer.`);
    }
    if (frame.sequenceNumber !== index) {
      issues.push(`${path}.sequenceNumber must exactly equal its retained index.`);
    }
    for (const timeField of [
      "tMs",
      "sourceTimestampMs",
      "callbackMonotonicTimeMs",
    ] as const) {
      if (!finiteNonNegative(frame[timeField])) {
        issues.push(`${path}.${timeField} must be finite and non-negative.`);
      }
    }
    if (!Number.isFinite(frame.frameWidth) || frame.frameWidth <= 0) {
      issues.push(`${path}.frameWidth must be finite and positive.`);
    }
    if (!Number.isFinite(frame.frameHeight) || frame.frameHeight <= 0) {
      issues.push(`${path}.frameHeight must be finite and positive.`);
    }
    if (!nonNegativeSafeInteger(frame.imageJointCount)) {
      issues.push(`${path}.imageJointCount must be a non-negative safe integer.`);
    }
    if (!nonNegativeSafeInteger(frame.worldJointCount)) {
      issues.push(`${path}.worldJointCount must be a non-negative safe integer.`);
    }
    const actualImageCount = Object.keys(frame.image).length;
    const actualWorldCount = Object.keys(frame.world).length;
    if (frame.imageJointCount !== actualImageCount) {
      issues.push(
        `${path}.imageJointCount does not equal the retained image landmark count.`,
      );
    }
    if (frame.worldJointCount !== actualWorldCount) {
      issues.push(
        `${path}.worldJointCount does not equal the retained world landmark count.`,
      );
    }
    const actuallyDetected = actualImageCount > 0 || actualWorldCount > 0;
    if (frame.poseDetected !== actuallyDetected) {
      issues.push(`${path}.poseDetected does not match retained landmarks.`);
    }
    issues.push(...validateLandmarks(frame.image, `${path}.image`));
    issues.push(...validateLandmarks(frame.world, `${path}.world`));

    if (index > 0) {
      const previous = frames[index - 1];
      for (const timeField of [
        "tMs",
        "sourceTimestampMs",
        "callbackMonotonicTimeMs",
      ] as const) {
        if (
          finiteNonNegative(previous[timeField]) &&
          finiteNonNegative(frame[timeField]) &&
          frame[timeField] < previous[timeField]
        ) {
          issues.push(`${path}.${timeField} moved backward.`);
        }
      }
    }
  });

  for (const kind of ["reference", "attempt"] as const) {
    const interval = intervals.find((entry) => entry.kind === kind);
    if (!interval) continue;
    const poseBearingCount = frames.filter(
      (frame) =>
        frame.poseDetected &&
        frame.callbackMonotonicTimeMs >= interval.begin.callbackMonotonicTimeMs &&
        frame.callbackMonotonicTimeMs <= interval.end.callbackMonotonicTimeMs,
    ).length;
    if (poseBearingCount === 0) {
      issues.push(`The ${kind} interval must contain at least one detected pose.`);
    }
  }

  return gate("pose-integrity", issues, {
    retainedPoseFrames: frames.length,
    detectedPoseFrames: frames.filter((frame) => frame.poseDetected).length,
  });
}

function validateExactCounts(
  episode: ContinuousCaptureAdmissionEpisode,
): CaptureAdmissionGateReport {
  const issues: string[] = [];
  const checks: Array<[number, number, string]> = [
    [
      episode.declaredPoseFrameCount,
      episode.rawPoseFrames.length,
      "declaredPoseFrameCount",
    ],
    [
      episode.rawPoseArtifact.frameCount,
      episode.rawPoseFrames.length,
      "rawPoseArtifact.frameCount",
    ],
    [
      episode.derivedAidsArtifact.frameCount,
      episode.rawPoseFrames.length,
      "derivedAidsArtifact.frameCount",
    ],
    [
      episode.derivedAidsArtifact.aids.frameCount,
      episode.rawPoseFrames.length,
      "derivedAidsArtifact.aids.frameCount",
    ],
    [
      episode.declaredVideoFrameTimingCount,
      episode.videoFrameTimings.length,
      "declaredVideoFrameTimingCount",
    ],
    [
      episode.declaredIntervalCount,
      episode.intervals.length,
      "declaredIntervalCount",
    ],
  ];
  for (const [declared, actual, path] of checks) {
    if (!nonNegativeSafeInteger(declared)) {
      issues.push(`${path} must be a non-negative safe integer.`);
    } else if (declared !== actual) {
      issues.push(`${path} (${declared}) does not equal retained count (${actual}).`);
    }
  }
  if (episode.derivedAidsArtifact.aids.frames.length !== episode.rawPoseFrames.length) {
    issues.push(
      "derivedAidsArtifact.aids.frames length does not equal the raw pose frame count.",
    );
  }
  return gate("exact-frame-counts", issues, {
    rawPoseFrames: episode.rawPoseFrames.length,
    videoFrameTimings: episode.videoFrameTimings.length,
    intervals: episode.intervals.length,
  });
}

function validateTiming(
  episode: ContinuousCaptureAdmissionEpisode,
): {
  gate: CaptureAdmissionGateReport;
  validation: CaptureTimingSessionValidation | null;
} {
  try {
    const validation = validateCaptureTimingSession({
      videoFrames: episode.videoFrameTimings,
      poses: episode.rawPoseFrames.map((frame) => ({
        sequence: frame.sequenceNumber,
        callbackMonotonicTimeMs: frame.callbackMonotonicTimeMs,
      })),
      intervals: episode.intervals,
    });
    const issues = validation.issues.map(
      (entry) => `${entry.code} at ${entry.path}: ${entry.message}`,
    );
    return {
      gate: gate("continuous-timing", issues, {
        totalPoses: validation.poseTiming.coverage.totalPoses,
        causallyAttachedPoses:
          validation.poseTiming.coverage.causallyAttachedPoses,
        posesWithinTimingExtent:
          validation.poseTiming.coverage.posesWithinTimingExtent,
        videoSequenceCoverage:
          validation.videoTiming.coverage.sequence === "complete",
        presentedFrameCoverage: validation.videoTiming.coverage.presentedFrames,
      }),
      validation,
    };
  } catch (error) {
    return {
      gate: gate("continuous-timing", [
        `Timing validation failed closed: ${error instanceof Error ? error.message : String(error)}`,
      ]),
      validation: null,
    };
  }
}

async function validateRawArtifact(
  episode: ContinuousCaptureAdmissionEpisode,
): Promise<{
  gate: CaptureAdmissionGateReport;
  computed: ComputedCanonicalArtifact | null;
}> {
  const issues: string[] = [];
  try {
    const computed = await identifyCanonicalCaptureArtifact(
      episode.rawPoseFrames,
    );
    if (episode.rawPoseArtifact.canonicalJson !== computed.canonicalJson) {
      issues.push(
        "rawPoseArtifact.canonicalJson is not the exact canonical UTF-8 serialization of rawPoseFrames.",
      );
    }
    if (episode.rawPoseArtifact.bytes !== computed.bytes) {
      issues.push("rawPoseArtifact.bytes does not match canonical raw pose bytes.");
    }
    if (!isSha256(episode.rawPoseArtifact.sha256)) {
      issues.push("rawPoseArtifact.sha256 must be a lowercase SHA-256 digest.");
    } else if (episode.rawPoseArtifact.sha256 !== computed.sha256) {
      issues.push("rawPoseArtifact.sha256 does not match canonical raw pose bytes.");
    }
    return {
      gate: gate("raw-pose-artifact", issues, {
        computedBytes: computed.bytes,
        computedSha256: computed.sha256,
      }),
      computed,
    };
  } catch (error) {
    issues.push(
      `Raw pose canonicalization failed closed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return { gate: gate("raw-pose-artifact", issues), computed: null };
  }
}

async function validateDerivedAidsArtifact(
  episode: ContinuousCaptureAdmissionEpisode,
  rawComputed: ComputedCanonicalArtifact | null,
  poseGatePassed: boolean,
): Promise<CaptureAdmissionGateReport> {
  const issues: string[] = [];
  if (!poseGatePassed) {
    issues.push(
      "Deterministic aids are not trusted because raw pose integrity failed.",
    );
  }
  if (!rawComputed) {
    issues.push("Deterministic aids cannot bind to an unverified raw pose artifact.");
  }

  try {
    const storedComputed = await identifyCanonicalCaptureArtifact(
      episode.derivedAidsArtifact.aids,
    );
    if (episode.derivedAidsArtifact.canonicalJson !== storedComputed.canonicalJson) {
      issues.push(
        "derivedAidsArtifact.canonicalJson is not the exact canonical UTF-8 serialization of stored aids.",
      );
    }
    if (episode.derivedAidsArtifact.bytes !== storedComputed.bytes) {
      issues.push(
        "derivedAidsArtifact.bytes does not match canonical stored aids bytes.",
      );
    }
    if (!isSha256(episode.derivedAidsArtifact.sha256)) {
      issues.push(
        "derivedAidsArtifact.sha256 must be a lowercase SHA-256 digest.",
      );
    } else if (episode.derivedAidsArtifact.sha256 !== storedComputed.sha256) {
      issues.push(
        "derivedAidsArtifact.sha256 does not match canonical stored aids bytes.",
      );
    }

    if (rawComputed) {
      if (
        episode.derivedAidsArtifact.derivedFromRawSha256 !== rawComputed.sha256
      ) {
        issues.push(
          "derivedAidsArtifact.derivedFromRawSha256 does not bind to computed raw pose bytes.",
        );
      }
      if (
        episode.derivedAidsArtifact.derivedFromRawSha256 !==
        episode.rawPoseArtifact.sha256
      ) {
        issues.push(
          "derivedAidsArtifact.derivedFromRawSha256 does not bind to the retained raw pose digest.",
        );
      }
    }

    if (poseGatePassed) {
      const recomputedAids = deriveResearchCaptureAids(
        episode.rawPoseFrames,
      );
      const recomputed = await identifyCanonicalCaptureArtifact(recomputedAids);
      if (recomputed.canonicalJson !== storedComputed.canonicalJson) {
        issues.push(
          "Stored aids canonical bytes do not exactly equal deriveResearchCaptureAids(rawPoseFrames).",
        );
      }
      if (recomputed.sha256 !== episode.derivedAidsArtifact.sha256) {
        issues.push(
          "Stored aids digest does not equal the deterministic recomputation digest.",
        );
      }
    }

    return gate("derived-aids-artifact", issues, {
      storedBytes: storedComputed.bytes,
      storedSha256: storedComputed.sha256,
      rawBindingPresent: isSha256(
        episode.derivedAidsArtifact.derivedFromRawSha256,
      ),
    });
  } catch (error) {
    issues.push(
      `Deterministic aids verification failed closed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return gate("derived-aids-artifact", issues);
  }
}

async function validateProtocol(
  protocol: FrozenProtocolIdentity,
): Promise<CaptureAdmissionGateReport> {
  const issues: string[] = [];
  addRequiredTextIssue(issues, protocol.protocol.protocolId, "protocol.protocolId");
  addRequiredTextIssue(
    issues,
    protocol.protocol.protocolVersion,
    "protocol.protocolVersion",
  );
  if (!isSha256(protocol.sha256)) {
    issues.push("frozenProtocol.sha256 must be a lowercase SHA-256 digest.");
  }
  try {
    const computed = await identifyCanonicalCaptureArtifact(protocol.protocol);
    if (protocol.canonicalJson !== computed.canonicalJson) {
      issues.push(
        "frozenProtocol.canonicalJson is not the exact immutable protocol serialization.",
      );
    }
    if (protocol.sha256 !== computed.sha256) {
      issues.push("frozenProtocol.sha256 does not match the immutable protocol bytes.");
    }
    return gate("immutable-protocol", issues, {
      protocolId: protocol.protocol.protocolId,
      protocolVersion: protocol.protocol.protocolVersion,
      computedSha256: computed.sha256,
    });
  } catch (error) {
    issues.push(
      `Protocol verification failed closed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return gate("immutable-protocol", issues);
  }
}

async function validateProtocolExecution(
  identity: FrozenProtocolExecutionIdentity | undefined,
  protocol: FrozenProtocolIdentity,
): Promise<CaptureAdmissionGateReport> {
  const issues: string[] = [];
  if (!identity || !identity.execution) {
    return gate("immutable-protocol-execution", [
      "A retained canonical protocol-execution artifact is required.",
    ]);
  }

  const execution = identity.execution;
  try {
    const computed = await identifyCanonicalCaptureArtifact(execution);
    if (identity.canonicalJson !== computed.canonicalJson) {
      issues.push(
        "protocolExecution.canonicalJson is not the exact canonical executor-output serialization.",
      );
    }
    if (identity.bytes !== computed.bytes) {
      issues.push("protocolExecution.bytes does not match canonical executor-output bytes.");
    }
    if (!isSha256(identity.sha256)) {
      issues.push("protocolExecution.sha256 must be a lowercase SHA-256 digest.");
    } else if (identity.sha256 !== computed.sha256) {
      issues.push("protocolExecution.sha256 does not match the retained executor output.");
    }
    if (execution.protocolSha256 !== protocol.sha256) {
      issues.push(
        "The retained protocol execution does not bind to frozenProtocol.sha256.",
      );
    }
    if (execution.protocolId !== protocol.protocol.protocolId) {
      issues.push("The retained protocol execution has a different protocolId.");
    }
    if (execution.protocolVersion !== protocol.protocol.protocolVersion) {
      issues.push("The retained protocol execution has a different protocolVersion.");
    }
    if (execution.status === "observable") {
      if (!execution.evidence) {
        issues.push("An observable protocol execution requires retained evidence bindings.");
      }
      for (const [field, value] of [
        ["referencePeakDegrees", execution.referencePeakDegrees],
        ["attemptPeakDegrees", execution.attemptPeakDegrees],
        ["targetMinusObservedDegrees", execution.targetMinusObservedDegrees],
      ] as const) {
        if (!Number.isFinite(value)) {
          issues.push(`Observable protocol execution ${field} must be finite.`);
        }
      }
    } else if (!execution.abstainReason?.trim()) {
      issues.push("An unobservable protocol execution requires an abstention reason.");
    }

    return gate("immutable-protocol-execution", issues, {
      status: execution.status,
      computedSha256: computed.sha256,
      protocolSha256: execution.protocolSha256,
    });
  } catch (error) {
    issues.push(
      `Protocol-execution verification failed closed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return gate("immutable-protocol-execution", issues);
  }
}

function validateRuntimeModel(
  model: RuntimePoseModelIdentity,
): CaptureAdmissionGateReport {
  const issues: string[] = [];
  for (const [path, value] of [
    ["modelName", model.modelName],
    ["runtimeLibrary", model.runtimeLibrary],
    ["runtimeVersion", model.runtimeVersion],
    ["assetPath", model.assetPath],
    ["realizedDelegate", model.realizedDelegate],
  ] as const) {
    addRequiredTextIssue(issues, value, `runtimePoseModel.${path}`);
  }
  if (!positiveSafeInteger(model.assetBytes)) {
    issues.push("runtimePoseModel.assetBytes must be a positive safe integer.");
  }
  if (!isSha256(model.assetSha256)) {
    issues.push(
      "runtimePoseModel.assetSha256 must be a lowercase SHA-256 digest.",
    );
  }
  return gate("runtime-model-identity", issues, {
    assetBytes: model.assetBytes,
    assetSha256: model.assetSha256,
  });
}

function validateAppEnvironment(
  identity: CaptureAppEnvironmentIdentity,
): CaptureAdmissionGateReport {
  const issues: string[] = [];
  for (const [path, value] of [
    ["appName", identity.appName],
    ["appVersion", identity.appVersion],
    ["collectorSchemaVersion", identity.collectorSchemaVersion],
    ["userAgent", identity.userAgent],
    ["platform", identity.platform],
    ["locale", identity.locale],
    ["timeZone", identity.timeZone],
  ] as const) {
    addRequiredTextIssue(issues, value, `appEnvironment.${path}`);
  }
  if (!isSha256(identity.appBuildSha256)) {
    issues.push(
      "appEnvironment.appBuildSha256 must be a lowercase SHA-256 digest.",
    );
  }
  if (!validTimestamp(identity.capturedAt)) {
    issues.push("appEnvironment.capturedAt must be a valid timestamp.");
  }
  return gate("app-environment-identity", issues, {
    appVersion: identity.appVersion,
    collectorSchemaVersion: identity.collectorSchemaVersion,
    appBuildSha256: identity.appBuildSha256,
  });
}

function validateVideo(
  episode: ContinuousCaptureAdmissionEpisode,
): CaptureAdmissionGateReport {
  const issues: string[] = [];
  const video = episode.video;
  if (!episode.rawVideoEnabled) {
    issues.push(
      "The frozen collector-v2 admission class requires one retained continuous video; pose-only fixtures remain a separate quarantined mechanics evidence class.",
    );
  }
  if (episode.rawVideoEnabled && !video) {
    issues.push(
      "rawVideoEnabled requires exactly one continuous video asset identity.",
    );
  }
  if (!episode.rawVideoEnabled && video) {
    issues.push("A video asset must be absent when rawVideoEnabled is false.");
  }
  if (video) {
    addRequiredTextIssue(issues, video.fileName, "video.fileName");
    addRequiredTextIssue(issues, video.mimeType, "video.mimeType");
    if (!positiveSafeInteger(video.bytes)) {
      issues.push("video.bytes must be a positive safe integer.");
    }
    if (!isSha256(video.sha256)) {
      issues.push("video.sha256 must be a lowercase SHA-256 digest.");
    }
    if (video.captureTopology !== "one-continuous-session") {
      issues.push("video.captureTopology must be one-continuous-session.");
    }
  }
  return gate("continuous-video-identity", issues, {
    rawVideoEnabled: episode.rawVideoEnabled,
    retainedVideoAssetCount: video ? 1 : 0,
    videoBytes: video?.bytes ?? 0,
  });
}

function validateReplay(
  episode: ContinuousCaptureAdmissionEpisode,
  timingValidation: CaptureTimingSessionValidation | null,
): CaptureAdmissionGateReport {
  const audit = episode.replayAudit;
  const issues: string[] = [];
  if (audit.status !== "passed") issues.push("replayAudit.status must be passed.");
  addRequiredTextIssue(issues, audit.auditorCode, "replayAudit.auditorCode");
  if (!validTimestamp(audit.auditedAt)) {
    issues.push("replayAudit.auditedAt must be a valid timestamp.");
  }
  if (audit.rawPoseSha256 !== episode.rawPoseArtifact.sha256) {
    issues.push("replayAudit.rawPoseSha256 does not match the raw pose artifact.");
  }
  if (audit.derivedAidsSha256 !== episode.derivedAidsArtifact.sha256) {
    issues.push(
      "replayAudit.derivedAidsSha256 does not match the derived aids artifact.",
    );
  }
  if (episode.rawVideoEnabled) {
    if (!episode.video || audit.videoSha256 !== episode.video.sha256) {
      issues.push("replayAudit.videoSha256 does not match the continuous video.");
    }
    if (!audit.continuousVideoDecoded) {
      issues.push("Replay must decode the retained continuous video.");
    }
  } else if (audit.videoSha256 !== undefined) {
    issues.push("replayAudit.videoSha256 must be absent when video is disabled.");
  }
  if (audit.replayedPoseFrameCount !== episode.rawPoseFrames.length) {
    issues.push(
      "replayAudit.replayedPoseFrameCount does not equal the retained raw pose count.",
    );
  }
  if (!audit.poseOverlayRendered) {
    issues.push("Replay must render the pose overlay.");
  }
  if (!audit.referenceAttemptIntervalsReviewed) {
    issues.push("Replay must review the reference and attempt intervals.");
  }
  if (!audit.timingSessionRevalidated) {
    issues.push("Replay must revalidate the continuous timing session.");
  }
  if (audit.encodedReplayTimelineCalibrated !== true) {
    issues.push(
      "Replay admission requires a measured calibration from the source timeline to the encoded MediaRecorder blob timeline.",
    );
  }
  if (!timingValidation?.valid) {
    issues.push("Replay cannot pass while current timing validation fails.");
  }
  return gate("replay-audit", issues, {
    replayedPoseFrameCount: audit.replayedPoseFrameCount,
    currentTimingValid: timingValidation?.valid ?? false,
    continuousVideoDecoded: audit.continuousVideoDecoded,
    encodedReplayTimelineCalibrated:
      audit.encodedReplayTimelineCalibrated === true,
  });
}

function validateCollectionScope(
  scope: CaptureCollectionScope,
): CaptureAdmissionGateReport {
  const issues: string[] = [];
  if (scope.participantScope !== "adult-researcher-self") {
    issues.push(
      "collectionScope.participantScope must be adult-researcher-self for this pilot.",
    );
  }
  if (scope.dataUseScope !== "internal-methods-only") {
    issues.push(
      "collectionScope.dataUseScope must be internal-methods-only for this pilot.",
    );
  }
  if (!scope.recordingRightsConfirmed) {
    issues.push("collectionScope.recordingRightsConfirmed must be true.");
  }
  if (!scope.nonClinicalResearchAcknowledged) {
    issues.push("collectionScope.nonClinicalResearchAcknowledged must be true.");
  }
  if (!validTimestamp(scope.selfAttestedAt)) {
    issues.push("collectionScope.selfAttestedAt must be a valid timestamp.");
  }
  return gate("collection-scope", issues, {
    participantScope: scope.participantScope,
    dataUseScope: scope.dataUseScope,
    legalOrIRBApprovalEstablished: false,
  });
}

function validateAnnotationSourceBinding(
  annotation: ResearchAnnotationSnapshot,
  episode: ContinuousCaptureAdmissionEpisode,
  context: string,
): string[] {
  const issues: string[] = [];
  if (
    annotation.assessment !== "acceptable" &&
    annotation.assessment !== "needs-correction"
  ) {
    return issues;
  }

  const execution = episode.protocolExecution?.execution;
  if (!execution) {
    return [`${context} cannot bind to a missing protocol-execution artifact.`];
  }
  if (annotation.protocolSha256 !== episode.frozenProtocol.sha256) {
    issues.push(`${context} protocolSha256 does not match frozenProtocol.sha256.`);
  }
  if (annotation.protocolExecutionSha256 !== episode.protocolExecution.sha256) {
    issues.push(
      `${context} protocolExecutionSha256 does not match the retained protocol-execution artifact.`,
    );
  }
  if (execution.status !== "observable" || !execution.evidence) {
    issues.push(`${context} makes an observable assessment from an unobservable execution.`);
    return issues;
  }
  if (
    annotation.referenceEvidenceSequenceNumber !==
    execution.evidence.referenceSequenceNumber
  ) {
    issues.push(
      `${context} referenceEvidenceSequenceNumber does not match the retained executor evidence.`,
    );
  }
  if (
    annotation.attemptEvidenceSequenceNumber !== execution.evidence.attemptSequenceNumber
  ) {
    issues.push(
      `${context} attemptEvidenceSequenceNumber does not match the retained executor evidence.`,
    );
  }
  if (annotation.metric !== execution.metricId) {
    issues.push(`${context} metric does not match the retained executor metricId.`);
  }
  if (annotation.unit !== "degrees") {
    issues.push(`${context} unit must be degrees for this retained executor output.`);
  }
  if (annotation.observedValue !== execution.attemptPeakDegrees) {
    issues.push(`${context} observedValue does not match the retained attempt peak.`);
  }
  if (annotation.targetValue !== execution.referencePeakDegrees) {
    issues.push(`${context} targetValue does not match the retained reference peak.`);
  }
  if (annotation.delta !== execution.targetMinusObservedDegrees) {
    issues.push(`${context} delta does not match the retained executor delta.`);
  }
  if (
    annotation.evidenceStartMs === undefined ||
    annotation.evidenceEndMs === undefined ||
    annotation.evidenceStartMs > execution.evidence.attemptSessionTimeMs ||
    annotation.evidenceEndMs < execution.evidence.attemptSessionTimeMs
  ) {
    issues.push(`${context} evidence interval does not contain the retained attempt evidence time.`);
  }
  return issues;
}

function validateAppendOnlyReview(
  episode: ContinuousCaptureAdmissionEpisode,
): CaptureAdmissionGateReport {
  const review = episode.review;
  const issues: string[] = [];
  if (review.storageMode !== "append-only") {
    issues.push("review.storageMode must be append-only.");
  }
  if (review.claimedFinalState !== "adjudicated") {
    issues.push("review.claimedFinalState must be adjudicated.");
  }
  review.entries.forEach((entry, index) => {
    issues.push(
      ...validateAnnotationSourceBinding(
        entry.annotation,
        episode,
        `review.entries[${index}].annotation`,
      ),
    );
  });
  if (review.adjudication) {
    issues.push(
      ...validateAnnotationSourceBinding(
        review.adjudication.annotation,
        episode,
        "review.adjudication.annotation",
      ),
    );
  }

  let derivedState = "invalid";
  let resolution = "none";
  try {
    const result = deriveResearchAnnotationReview(review.entries, {
      numericTolerance: review.numericTolerance,
      adjudication: review.adjudication,
    });
    derivedState = result.state;
    resolution = result.resolution;
    if (result.state !== "adjudicated") {
      issues.push(
        `Independent review derives ${result.state}, not final adjudicated status.`,
      );
    }
    if (result.clinicalTruthStatus !== "not-established-by-review") {
      issues.push("Review result attempted to establish clinical truth.");
    }

    const orderedEvents = [
      ...review.entries.map((entry) => ({
        annotationCreatedAt: entry.annotation.createdAt,
        recordedAt:
          entry.independentReviewDeclaration?.declaredAt ??
          entry.annotation.createdAt,
      })),
      ...(review.adjudication
        ? [
            {
              annotationCreatedAt: review.adjudication.annotation.createdAt,
              recordedAt: review.adjudication.declaredAt,
            },
          ]
        : []),
    ];
    let previousRecordedAt = Number.NEGATIVE_INFINITY;
    orderedEvents.forEach((event, index) => {
      const createdAt = Date.parse(event.annotationCreatedAt);
      const recordedAt = Date.parse(event.recordedAt);
      if (!Number.isFinite(createdAt)) {
        issues.push(`review append-only event ${index} has an invalid createdAt.`);
      }
      if (!Number.isFinite(recordedAt)) {
        issues.push(`review append-only event ${index} has an invalid recorded time.`);
      }
      if (Number.isFinite(createdAt) && Number.isFinite(recordedAt) && recordedAt < createdAt) {
        issues.push(
          `review append-only event ${index} was recorded before its annotation was created.`,
        );
      }
      if (Number.isFinite(recordedAt) && recordedAt < previousRecordedAt) {
        issues.push(`review append-only event ${index} moved backward in time.`);
      }
      if (Number.isFinite(recordedAt)) previousRecordedAt = recordedAt;
    });
  } catch (error) {
    issues.push(
      `Independent review validation failed closed: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  return gate("append-only-independent-review", issues, {
    primaryReviewEntries: review.entries.length,
    hasThirdRaterAdjudication: Boolean(review.adjudication),
    derivedState,
    resolution,
    clinicalTruthEstablished: false,
  });
}

/**
 * Pure, read-only admission decision. Passing every gate permits only a bounded
 * MEASURED-PILOT mechanics/methods use. It does not establish clinical truth,
 * generalization, safety, or body-native intelligence.
 */
export async function verifyContinuousCaptureAdmission(
  episode: ContinuousCaptureAdmissionEpisode,
): Promise<CaptureAdmissionReport> {
  const lineageGate = validateLineage(episode);
  const poseGate = validatePoseIntegrity(
    episode.rawPoseFrames,
    episode.intervals,
  );
  const countGate = validateExactCounts(episode);
  const timing = validateTiming(episode);
  const rawArtifact = await validateRawArtifact(episode);
  const aidsGate = await validateDerivedAidsArtifact(
    episode,
    rawArtifact.computed,
    poseGate.passed,
  );
  const protocolGate = await validateProtocol(episode.frozenProtocol);
  const protocolExecutionGate = await validateProtocolExecution(
    episode.protocolExecution,
    episode.frozenProtocol,
  );
  const modelGate = validateRuntimeModel(episode.runtimePoseModel);
  const environmentGate = validateAppEnvironment(episode.appEnvironment);
  const videoGate = validateVideo(episode);
  const replayGate = validateReplay(episode, timing.validation);
  const collectionGate = validateCollectionScope(episode.collectionScope);
  const reviewGate = validateAppendOnlyReview(episode);

  const gates = [
    lineageGate,
    poseGate,
    countGate,
    timing.gate,
    rawArtifact.gate,
    aidsGate,
    protocolGate,
    protocolExecutionGate,
    modelGate,
    environmentGate,
    videoGate,
    replayGate,
    collectionGate,
    reviewGate,
  ] as const;
  const issues = gates.flatMap((entry) =>
    entry.issues.map((issue) => `[${entry.id}] ${issue}`),
  );
  const admitted = gates.every((entry) => entry.passed);

  return {
    status: admitted
      ? "ALLOWED_MEASURED_PILOT_MECHANICS_ONLY"
      : "QUARANTINED",
    admitted,
    evidenceClass: admitted
      ? "MEASURED_PILOT_MECHANICS_ONLY"
      : "QUARANTINED_UNADMITTED",
    clinicalTruthStatus: "NOT_ESTABLISHED",
    permittedUse: admitted
      ? "bounded-nonclinical-mechanics-methods-pilot"
      : "none-while-quarantined",
    prohibitedClaims: PROHIBITED_CLAIMS,
    gates,
    issues,
  };
}
