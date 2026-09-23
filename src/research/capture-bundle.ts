import type { JointName, LandmarkSet, PoseFrame } from "../motion/types";
import {
  deriveResearchCaptureAids,
  type ResearchCaptureAids,
} from "./capture-aids";
import type {
  CaptureInterval,
  VideoFrameTimingSample,
} from "./capture-timing";
import { validateCaptureTimingSession } from "./capture-timing";
import type { ObservableProtocolExecution } from "./protocol-executor";
import { executeObservableProtocol } from "./protocol-executor";
import type { FrozenObservableProtocol } from "./protocol";
import type { LocalReplayIntegrityAudit } from "./capture-replay";
import {
  auditReplayIntervalBoundaryCoverage,
  RECORDED_BLOB_TIMELINE_MAPPING,
  REPLAY_INTEGRITY_EVIDENCE_SCOPE,
  validateDecodedContinuousVideoEvidence,
  validateHumanReplayConfirmation,
  validateReplayPoseAttachments,
  validateReplayVideoTimingTrace,
  type ReplayPoseAttachment,
} from "./capture-replay";

export type CaptureKind = "reference" | "attempt" | "revision";
export type ObservableAssessment =
  | "acceptable"
  | "needs-correction"
  | "uncertain"
  | "unobservable";

export const RESEARCH_COLLECTOR_BUILD_ID =
  "relay.body-native-research-collector/2.0.0" as const;

export interface ResearchPoseVideoTimingAttachment {
  strategy: "nearest-latest-causal-video-frame";
  sequence: number;
  callbackMonotonicTimeMs: number;
  mediaTimeMs: number;
  presentedFrames?: number;
}

export interface ResearchPoseFrame {
  sequenceNumber: number;
  tMs: number;
  sourceTimestampMs: number;
  callbackMonotonicTimeMs: number;
  frameWidth: number;
  frameHeight: number;
  poseDetected: boolean;
  imageJointCount: number;
  worldJointCount: number;
  image: LandmarkSet;
  world: LandmarkSet;
  videoTiming?: ResearchPoseVideoTimingAttachment;
}

/**
 * Build the one canonical replay attachment shape retained in an episode.
 * Pose geometry deliberately stays in the raw pose trace instead of being
 * duplicated inside every boundary-audit record.
 */
export function buildResearchReplayAttachments(
  frames: readonly ResearchPoseFrame[],
): ReplayPoseAttachment[] {
  return frames.flatMap((frame) =>
    frame.videoTiming
      ? [
          {
            strategy: "nearest-latest-causal-video-frame" as const,
            pose: {
              sequence: frame.sequenceNumber,
              callbackMonotonicTimeMs: frame.callbackMonotonicTimeMs,
            },
            videoFrame: frame.videoTiming,
          },
        ]
      : [],
  );
}

export interface StoredResearchAids {
  payload: ResearchCaptureAids;
  derivedFromRawPoseSha256: string;
  canonicalBytes: number;
  canonicalSha256: string;
}

export interface ResearchVideoAsset {
  fileName: string;
  mimeType: string;
  bytes: number;
  sha256: string;
  continuousEpisode: true;
}

export interface ResearchReplayAudit {
  status: "PASS" | "FAIL";
  videoAssetSha256: string;
  poseFrameCount: number;
  videoTimingCount: number;
  intervalBoundaryCoverage: boolean;
  causalOverlaySelection: boolean;
  encodedReplayTimelineMapping: typeof RECORDED_BLOB_TIMELINE_MAPPING;
  encodedReplayTimelineCalibrated: false;
  renderedOverlayAlignment: "APPROXIMATE_UNCALIBRATED";
  humanConfirmation: {
    reviewerCode: string;
    watchedContinuousVideo: true;
    overlaysVisible: true;
    referenceIntervalConfirmed: true;
    attemptIntervalConfirmed: true;
    declaredAt: string;
    evidenceScope: typeof REPLAY_INTEGRITY_EVIDENCE_SCOPE;
  };
  issues: string[];
  integrityAudit: LocalReplayIntegrityAudit;
}

export interface ResearchContinuousCaptureSession {
  sessionCaptureId: string;
  startedAt: string;
  endedAt: string;
  durationMs: number;
  cameraFacing: "user" | "environment";
  cameraSettings: Record<string, string | number | boolean | undefined>;
  clocks: {
    performanceTimeOriginMs: number;
    sessionStartedMonotonicMs: number;
    videoRecorderStartedMonotonicMs: number;
    videoFrameClock: "requestVideoFrameCallback.mediaTime";
    poseClock: "performance.now";
    poseVideoSynchronization: "latest-past-causal-video-frame-callback";
  };
  poseModel: {
    name: "MediaPipe Pose Landmarker Full";
    asset: "/models/pose_landmarker_full.task";
    assetBytes: 9398198;
    assetSha256: "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad";
    inferenceCadenceMs: 66;
    requestedDelegate: "GPU";
    realizedDelegate: "unobserved-gpu-with-cpu-fallback";
  };
  frameCount: number;
  poseFrames: ResearchPoseFrame[];
  videoFrameTimings: VideoFrameTimingSample[];
  intervals: CaptureInterval[];
  derivedAids: StoredResearchAids;
  protocolExecution: ObservableProtocolExecution;
  video: ResearchVideoAsset;
  replayAudit: ResearchReplayAudit;
}

export interface ResearchEpisodeMetadata {
  participantCode: string;
  sessionCode: string;
  episodeCode: string;
  movementFamily: string;
  cameraView: "front" | "left" | "right" | "oblique" | "other";
  deviceNotes: string;
  calibrationNotes: string;
}

/**
 * The v2 pilot is intentionally limited to an adult researcher recording themself.
 * This is an engineering collection scope, not a substitute for an approved
 * consent process and not permission to recruit or distribute participant data.
 */
export interface ResearchConsent {
  adultResearcherSelfConfirmed: true;
  collectorIsParticipant: true;
  recordingRightsConfirmed: true;
  nonClinicalResearchAcknowledged: true;
  dataUse: "internal-methods-validation-only";
  rawVideoIncluded: true;
  authorizationStatus: "self-attested-limited-pilot";
  formVersion: "relay-research-self-capture-v2";
  confirmedAt: string;
}

export interface ResearchAnnotation {
  annotationId: string;
  raterCode: string;
  raterRole: "collector" | "movement-professional" | "independent-reviewer";
  reviewStatus: "draft-unreviewed" | "independently-reviewed";
  movementIdentity: string;
  assessment: ObservableAssessment;
  phaseStartMs?: number;
  phaseEndMs?: number;
  realizedErrorType?: string;
  evidenceJoints: JointName[];
  evidenceStartMs?: number;
  evidenceEndMs?: number;
  evidenceClock?:
    | "session-pose-callback-offset-ms"
    | "encoded-replay-approximate-ms";
  metric?: string;
  observedValue?: number;
  targetValue?: number;
  delta?: number;
  unit?: string;
  desiredTargetSource:
    | "captured-reference"
    | "supplied-protocol"
    | "analytic-oracle"
    | "unknown";
  cue?: string;
  confidence: number;
  abstainReason?: string;
  notes?: string;
  protocolSha256?: string;
  protocolExecutionSha256?: string;
  referenceEvidenceSequenceNumber?: number;
  attemptEvidenceSequenceNumber?: number;
  createdAt: string;
}

export interface ResearchRuntimeEnvironment {
  collectorBuildId: typeof RESEARCH_COLLECTOR_BUILD_ID;
  userAgent: string;
  platform: string;
  language: string;
  timezone: string;
  secureContext: boolean;
  origin: string;
  recordedAt: string;
}

export interface ResearchEpisodeDraft {
  metadata: ResearchEpisodeMetadata;
  consent: ResearchConsent;
  protocol: FrozenObservableProtocol;
  session: ResearchContinuousCaptureSession;
  annotations: ResearchAnnotation[];
  environment: ResearchRuntimeEnvironment;
}

export interface ResearchBundle extends ResearchEpisodeDraft {
  schemaVersion: "relay.body-native-episode/2";
  evidenceClass: "FIRST_PARTY_SELF_CAPTURE_MECHANICS_PILOT";
  admissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW";
  createdAt: string;
  lineage: {
    episodeAncestorId: string;
    syntheticDescendantsCountAsIndependent: false;
  };
  observationContract: {
    nativeSignals: string[];
    complementarySignals: string[];
    derivedSignals: string[];
    absentSignals: string[];
  };
  contentDigests: Array<{
    path: string;
    bytes: number;
    sha256: string;
  }>;
  claimBoundary: string[];
}

export interface ResearchExport {
  bundle: ResearchBundle;
  json: string;
  sha256: string;
  sha256Sidecar: string;
}

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

export function canonicalJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function sha256Bytes(bytes: Uint8Array): Promise<string> {
  const owned = Uint8Array.from(bytes);
  const digest = await crypto.subtle.digest("SHA-256", owned.buffer);
  return bytesToHex(new Uint8Array(digest));
}

export async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

export function serializePoseFrame(
  frame: PoseFrame,
  sessionStartedAtMs: number,
  sequenceNumber = 0,
  videoTiming?: ResearchPoseVideoTimingAttachment,
): ResearchPoseFrame {
  const imageJointCount = Object.keys(frame.image).length;
  const worldJointCount = Object.keys(frame.world).length;
  return {
    sequenceNumber,
    tMs: Math.max(0, frame.timestampMs - sessionStartedAtMs),
    sourceTimestampMs: frame.timestampMs,
    callbackMonotonicTimeMs: frame.timestampMs,
    frameWidth: frame.frameWidth,
    frameHeight: frame.frameHeight,
    poseDetected: imageJointCount > 0 || worldJointCount > 0,
    imageJointCount,
    worldJointCount,
    image: frame.image,
    world: frame.world,
    videoTiming,
  };
}

function finite(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value);
}

function validIsoInstant(value: string): boolean {
  return Boolean(value) && Number.isFinite(Date.parse(value));
}

function validateAnnotation(annotation: ResearchAnnotation): string[] {
  const errors: string[] = [];
  if (!annotation.raterCode.trim()) errors.push("Rater code is required.");
  if (!annotation.movementIdentity.trim())
    errors.push("The observed movement identity is required.");
  if (!Number.isFinite(annotation.confidence) || annotation.confidence < 0 || annotation.confidence > 1)
    errors.push("Annotation confidence must be finite and between 0 and 1.");
  for (const field of [
    "phaseStartMs",
    "phaseEndMs",
    "evidenceStartMs",
    "evidenceEndMs",
    "observedValue",
    "targetValue",
    "delta",
  ] as const) {
    const value = annotation[field];
    if (value !== undefined && !Number.isFinite(value)) {
      errors.push(`${field} must be finite when present.`);
    }
  }
  if (annotation.assessment === "needs-correction") {
    if (!annotation.realizedErrorType?.trim())
      errors.push("A realized reference-relative difference is required.");
    if (annotation.evidenceJoints.length === 0)
      errors.push("At least one evidence joint is required for a correction.");
    if (!annotation.metric?.trim())
      errors.push("An observable metric is required for a correction.");
    if (!finite(annotation.observedValue))
      errors.push("A finite observed value is required for a correction.");
    if (!finite(annotation.targetValue))
      errors.push("A finite target value is required for a correction.");
    if (!finite(annotation.delta))
      errors.push("A finite target-minus-observed delta is required for a correction.");
    if (!finite(annotation.evidenceStartMs) || !finite(annotation.evidenceEndMs))
      errors.push("A finite evidence time range is required for a correction.");
    if (!annotation.unit?.trim())
      errors.push("A defined unit is required for a correction.");
    if (!annotation.cue?.trim())
      errors.push("A grounded reference-relative cue is required for a correction.");
    if (!annotation.protocolSha256?.trim() || !annotation.protocolExecutionSha256?.trim())
      errors.push("Correction annotations must bind the protocol and executor output hashes.");
  }
  if (annotation.assessment === "acceptable" || annotation.assessment === "needs-correction") {
    errors.push(
      "This uncalibrated mechanics pilot cannot label an estimator difference acceptable or in need of correction.",
    );
  }
  if (annotation.assessment === "uncertain") {
    if (!annotation.realizedErrorType?.trim())
      errors.push("An uncertain annotation must name the observed construct limitation.");
    if (annotation.evidenceJoints.length === 0)
      errors.push("An uncertain observation requires at least one evidence joint.");
    if (!annotation.metric?.trim())
      errors.push("An uncertain numeric observation requires its estimator metric.");
    if (!finite(annotation.observedValue) || !finite(annotation.targetValue) || !finite(annotation.delta))
      errors.push("An uncertain numeric observation requires finite observed, reference, and difference values.");
    if (!finite(annotation.evidenceStartMs) || !finite(annotation.evidenceEndMs))
      errors.push("An uncertain numeric observation requires a finite evidence time range.");
    if (!annotation.unit?.trim())
      errors.push("An uncertain numeric observation requires a unit.");
    if (annotation.cue?.trim())
      errors.push("An uncalibrated uncertain observation must not emit a corrective cue.");
    if (!annotation.protocolSha256?.trim() || !annotation.protocolExecutionSha256?.trim())
      errors.push("An uncertain numeric observation must bind the protocol and executor hashes.");
  }
  if (
    (finite(annotation.observedValue) || finite(annotation.targetValue) || finite(annotation.delta)) &&
    annotation.evidenceClock !== "session-pose-callback-offset-ms"
  ) {
    errors.push("Numeric annotation evidence must name the session pose-callback clock.");
  }
  if (
    finite(annotation.observedValue) &&
    finite(annotation.targetValue) &&
    finite(annotation.delta) &&
    Math.abs(annotation.targetValue - annotation.observedValue - annotation.delta) > 1e-9
  ) {
    errors.push("Annotation delta must equal target minus observed on the retained estimator output.");
  }
  if (annotation.assessment === "unobservable" && !annotation.abstainReason?.trim()) {
    errors.push("An abstention reason is required when the target is unobservable.");
  }
  return errors;
}

export function validateResearchDraft(draft: ResearchEpisodeDraft): string[] {
  const errors: string[] = [];
  const metadata = draft.metadata;
  if (!metadata.participantCode.trim()) errors.push("Participant code is required.");
  if (!metadata.sessionCode.trim()) errors.push("Session code is required.");
  if (!metadata.episodeCode.trim()) errors.push("Episode code is required.");
  if (!metadata.movementFamily.trim()) errors.push("Movement family is required.");
  if (metadata.cameraView !== draft.protocol.protocol.cameraView)
    errors.push("Camera view must match the frozen protocol.");
  if (!draft.consent.adultResearcherSelfConfirmed || !draft.consent.collectorIsParticipant)
    errors.push("This pilot is limited to an adult researcher recording themself.");
  if (!draft.consent.recordingRightsConfirmed)
    errors.push("Recording and research-use rights must be confirmed.");
  if (!draft.consent.nonClinicalResearchAcknowledged)
    errors.push("The non-clinical research boundary must be acknowledged.");
  if (
    draft.consent.dataUse !== "internal-methods-validation-only" ||
    draft.consent.rawVideoIncluded !== true
  ) {
    errors.push("This pilot requires internal-only use and a continuous replayable video.");
  }
  if (!validIsoInstant(draft.consent.confirmedAt))
    errors.push("The pre-capture consent confirmation time must be a valid timestamp.");
  const session = draft.session;
  if (
    validIsoInstant(draft.consent.confirmedAt) &&
    validIsoInstant(session.startedAt) &&
    Date.parse(draft.consent.confirmedAt) > Date.parse(session.startedAt)
  ) {
    errors.push("Consent must be confirmed before the continuous session starts.");
  }
  if (!Number.isFinite(session.durationMs) || session.durationMs <= 0)
    errors.push("A positive continuous-session duration is required.");
  if (session.frameCount !== session.poseFrames.length)
    errors.push("Declared pose-frame count must equal retained pose frames.");
  if (!session.poseFrames.some((frame) => frame.poseDetected))
    errors.push("A pose-bearing continuous session is required.");
  if (!session.video || session.video.bytes <= 0 || !session.video.sha256.trim())
    errors.push("A hash-bound continuous replay video is required.");
  const allowedCameraSettings = new Set(["width", "height", "frameRate", "facingMode"]);
  for (const key of Object.keys(session.cameraSettings)) {
    if (!allowedCameraSettings.has(key))
      errors.push(`Camera setting ${key} is not admitted; device identifiers must not be retained.`);
  }
  const lastVideoCallback = session.videoFrameTimings.at(-1)?.callbackMonotonicTimeMs;
  if (
    lastVideoCallback !== undefined &&
    Math.abs(
      session.durationMs -
        (lastVideoCallback - session.clocks.sessionStartedMonotonicMs),
    ) > 1e-6
  ) {
    errors.push("Session duration must equal the final video callback offset from session start.");
  }
  for (const frame of session.poseFrames) {
    if (frame.sourceTimestampMs !== frame.callbackMonotonicTimeMs) {
      errors.push("Pose source timestamps must equal this provider's callback monotonic timestamps.");
      break;
    }
    if (
      Math.abs(
        frame.tMs -
          (frame.callbackMonotonicTimeMs -
            session.clocks.sessionStartedMonotonicMs),
      ) > 1e-6
    ) {
      errors.push("Pose session time must equal callback time minus the session clock origin.");
      break;
    }
  }
  const timing = validateCaptureTimingSession({
    videoFrames: session.videoFrameTimings,
    poses: session.poseFrames.map((frame) => ({
      sequence: frame.sequenceNumber,
      callbackMonotonicTimeMs: frame.callbackMonotonicTimeMs,
    })),
    intervals: session.intervals,
  });
  errors.push(...timing.issues.map((entry) => entry.message));
  const replayAttachments = buildResearchReplayAttachments(session.poseFrames);
  const replayTiming = validateReplayVideoTimingTrace(session.videoFrameTimings);
  const replayPoses = validateReplayPoseAttachments(
    replayAttachments,
    session.videoFrameTimings,
  );
  const replayBoundaries = auditReplayIntervalBoundaryCoverage(
    session.intervals,
    session.videoFrameTimings,
    replayAttachments,
  );
  errors.push(
    ...replayTiming.issues.map((entry) => entry.message),
    ...replayPoses.issues.map((entry) => entry.message),
    ...replayBoundaries.issues.map((entry) => entry.message),
  );
  if (session.replayAudit.status !== "PASS")
    errors.push("Replay timing and human confirmation must pass before export.");
  if (!session.replayAudit.integrityAudit.valid)
    errors.push("The retained replay-integrity audit must be confirmed.");
  if (session.replayAudit.videoAssetSha256 !== session.video.sha256)
    errors.push("Replay audit must bind the exact continuous video asset.");
  if (session.replayAudit.poseFrameCount !== session.poseFrames.length)
    errors.push("Replay audit pose count must equal the retained pose trace.");
  if (session.replayAudit.videoTimingCount !== session.videoFrameTimings.length)
    errors.push("Replay audit video-timing count must equal the retained source trace.");
  if (
    session.replayAudit.encodedReplayTimelineMapping !==
      RECORDED_BLOB_TIMELINE_MAPPING ||
    session.replayAudit.encodedReplayTimelineCalibrated !== false ||
    session.replayAudit.renderedOverlayAlignment !== "APPROXIMATE_UNCALIBRATED"
  ) {
    errors.push("The current MediaRecorder source-to-blob timeline must remain explicitly uncalibrated.");
  }
  const recomputedDecode = validateDecodedContinuousVideoEvidence(
    session.replayAudit.integrityAudit.decodedVideo.observed,
    session.videoFrameTimings,
  );
  if (
    canonicalJson(recomputedDecode) !==
    canonicalJson(session.replayAudit.integrityAudit.decodedVideo)
  ) {
    errors.push("Decoded-video validation must exactly match deterministic recomputation.");
  }
  const recomputedHuman = validateHumanReplayConfirmation({
    ...session.replayAudit.humanConfirmation,
    evidenceScope: session.replayAudit.humanConfirmation.evidenceScope,
  });
  if (
    canonicalJson(recomputedHuman) !==
    canonicalJson(session.replayAudit.integrityAudit.humanConfirmation)
  ) {
    errors.push("Human replay validation must exactly match deterministic recomputation.");
  }
  if (
    canonicalJson(replayBoundaries) !==
    canonicalJson(session.replayAudit.integrityAudit.boundaryCoverage)
  ) {
    errors.push("Replay boundary validation must exactly match deterministic source-clock recomputation.");
  }
  if (
    session.replayAudit.intervalBoundaryCoverage !== replayBoundaries.valid ||
    session.replayAudit.causalOverlaySelection !== replayBoundaries.valid
  ) {
    errors.push("Replay summary booleans must match the deterministic source-clock boundary audit.");
  }
  if (draft.annotations.length === 0)
    errors.push("At least one structured collector annotation is required.");
  for (const annotation of draft.annotations) errors.push(...validateAnnotation(annotation));
  if (draft.environment.collectorBuildId !== RESEARCH_COLLECTOR_BUILD_ID)
    errors.push("Collector build identity does not match this schema.");
  if (session.protocolExecution.protocolSha256 !== draft.protocol.sha256)
    errors.push("Protocol execution must bind the exact frozen protocol hash.");
  return Array.from(new Set(errors));
}

export async function storeResearchAids(
  poseFrames: readonly ResearchPoseFrame[],
): Promise<StoredResearchAids> {
  const raw = canonicalJson(poseFrames);
  const payload = deriveResearchCaptureAids(poseFrames);
  const aids = canonicalJson(payload);
  return {
    payload,
    derivedFromRawPoseSha256: await sha256Text(raw),
    canonicalBytes: new TextEncoder().encode(aids).byteLength,
    canonicalSha256: await sha256Text(aids),
  };
}

export async function buildResearchExport(
  draft: ResearchEpisodeDraft,
): Promise<ResearchExport> {
  const errors = validateResearchDraft(draft);
  const frozenProtocolJson = canonicalJson(draft.protocol.protocol);
  const protocolSha = await sha256Text(frozenProtocolJson);
  if (protocolSha !== draft.protocol.sha256 || frozenProtocolJson !== draft.protocol.canonicalJson) {
    errors.push("Frozen protocol bytes or hash do not match the retained protocol.");
  }
  const recomputedAids = await storeResearchAids(draft.session.poseFrames);
  if (canonicalJson(recomputedAids) !== canonicalJson(draft.session.derivedAids)) {
    errors.push("Stored RAW+AIDS do not exactly match deterministic recomputation from RAW.");
  }
  const recomputedExecution = executeObservableProtocol(
    draft.protocol,
    draft.session.poseFrames,
    draft.session.intervals,
  );
  if (
    canonicalJson(recomputedExecution) !==
    canonicalJson(draft.session.protocolExecution)
  ) {
    errors.push(
      "Stored protocol execution does not exactly match deterministic recomputation from the frozen protocol, RAW pose, and intervals.",
    );
  }
  const executionJson = canonicalJson(recomputedExecution);
  const executionSha = await sha256Text(executionJson);
  if (draft.annotations.some((annotation) => annotation.protocolExecutionSha256 !== executionSha)) {
    errors.push("Every collector annotation must bind the exact protocol execution hash.");
  }
  if (errors.length > 0) throw new Error(Array.from(new Set(errors)).join(" "));

  const members = [
    ["protocol.json", `${JSON.stringify(draft.protocol.protocol, null, 2)}\n`],
    ["continuous.pose.json", `${JSON.stringify(draft.session.poseFrames, null, 2)}\n`],
    ["continuous.video-timing.json", `${JSON.stringify(draft.session.videoFrameTimings, null, 2)}\n`],
    ["segments.json", `${JSON.stringify(draft.session.intervals, null, 2)}\n`],
    ["derived/aids.json", `${JSON.stringify(draft.session.derivedAids, null, 2)}\n`],
    ["protocol-execution.json", `${JSON.stringify(recomputedExecution, null, 2)}\n`],
    ["replay-audit.json", `${JSON.stringify(draft.session.replayAudit, null, 2)}\n`],
    ["environment.json", `${JSON.stringify(draft.environment, null, 2)}\n`],
  ] as const;
  const memberDigests = await Promise.all(
    members.map(async ([path, content]) => ({
      path,
      bytes: new TextEncoder().encode(content).byteLength,
      sha256: await sha256Text(content),
    })),
  );
  const bundle: ResearchBundle = {
    schemaVersion: "relay.body-native-episode/2",
    evidenceClass: "FIRST_PARTY_SELF_CAPTURE_MECHANICS_PILOT",
    admissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW",
    createdAt: new Date().toISOString(),
    ...draft,
    lineage: {
      episodeAncestorId: [
        draft.metadata.participantCode,
        draft.metadata.sessionCode,
        draft.metadata.episodeCode,
      ].join("/"),
      syntheticDescendantsCountAsIndependent: false,
    },
    observationContract: {
      nativeSignals: [
        "one continuous timestamped MediaPipe image-landmark stream",
        "one continuous timestamped MediaPipe world-landmark stream",
        "per-landmark visibility and presence",
        "requestVideoFrameCallback timing attached causally to pose",
        "one locally exported continuous RGB video",
      ],
      complementarySignals: [
        "frozen same-person captured-reference protocol",
      ],
      derivedSignals: [
        "causal analytic observation coverage",
        "bilateral elbow/knee/hip angles",
        "backward-difference angular rates",
        "left-minus-right angle symmetry deltas",
        "reference-relative left-shoulder peak-angle comparison",
      ],
      absentSignals: [
        "force",
        "contact",
        "load",
        "pain",
        "intent",
        "clinical safety truth",
        "anatomical ground-truth joint angles",
      ],
    },
    contentDigests: [
      ...memberDigests,
      {
        path: `continuous/${draft.session.video.fileName}`,
        bytes: draft.session.video.bytes,
        sha256: draft.session.video.sha256,
      },
    ],
    claimBoundary: [
      "This is an adult researcher self-capture for internal methods validation only.",
      "The episode remains quarantined until two blinded independent reviews and any required adjudication pass.",
      "The captured reference is a comparator, not an ideal, safe, or clinical target.",
      "The annotation and analytic aids describe observable estimator output only and are not medical advice.",
      "Any later window, paraphrase, augmentation, or synthetic descendant inherits this episode ancestor.",
      "The encoded MediaRecorder replay timeline is uncalibrated against the source rVFC time origin; rendered markers and overlays are approximate and cannot support exact video-boundary claims.",
    ],
  };
  const json = `${JSON.stringify(bundle, null, 2)}\n`;
  const sha256 = await sha256Text(json);
  return {
    bundle,
    json,
    sha256,
    sha256Sidecar: `${sha256}  episode.json\n`,
  };
}

export function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
