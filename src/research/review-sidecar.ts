import {
  compareResearchAnnotations,
  deriveResearchAnnotationReview,
  type BlindedIndependentReviewDeclaration,
  type ResearchAnnotationDisagreement,
  type ResearchAnnotationReviewEntry,
  type ResearchAnnotationReviewResult,
} from "./annotation-review";
import {
  sha256Bytes,
  sha256Text,
  type ResearchAnnotation,
  type ResearchBundle,
  type ResearchPoseFrame,
} from "./capture-bundle";
import type { CaptureInterval, VideoFrameTimingSample } from "./capture-timing";
import { validateCaptureTimingSession } from "./capture-timing";
import {
  auditReplayIntervalBoundaryCoverage,
  validateDecodedContinuousVideoEvidence,
  type DecodedContinuousVideoEvidence,
  type ReplayPoseAttachment,
} from "./capture-replay";

export const INDEPENDENT_REVIEW_SIDECAR_SCHEMA =
  "relay.body-native-independent-review-sidecar/1" as const;

export const RESEARCH_REVIEW_JOINTS = [
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
] as const satisfies readonly ResearchAnnotation["evidenceJoints"][number][];

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const ARCHIVE_SUFFIX = ".relay-episode-v2.zip";
const CHECKSUM_PATH = "SHA256SUMS.txt";
const EPISODE_PATH = "episode.json";
const REVIEW_JOINT_SET = new Set<string>(RESEARCH_REVIEW_JOINTS);

type UnknownRecord = Record<string, unknown>;

export class ResearchReviewValidationError extends Error {
  readonly issues: readonly string[];

  constructor(issues: readonly string[]) {
    const unique = Array.from(new Set(issues));
    super(unique.join(" "));
    this.name = "ResearchReviewValidationError";
    this.issues = unique;
  }
}

export interface ResearchReviewChecksum {
  readonly path: string;
  readonly sha256: string;
  readonly bytes: number;
}

export interface VerifiedResearchReviewArchive {
  readonly archiveName: string;
  readonly episode: ResearchBundle;
  readonly episodeJsonSha256: string;
  readonly episodeJsonBytes: number;
  readonly videoPath: string;
  readonly videoBytes: Uint8Array;
  readonly checksums: readonly ResearchReviewChecksum[];
}

/**
 * This is the only episode surface the review UI may render before commitment.
 * Protocol, execution, metadata, and all prior annotations are deliberately absent.
 */
export interface BlindedResearchReviewView {
  readonly schemaVersion: "relay.body-native-episode/2";
  readonly admissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW";
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly episodeJsonSha256: string;
  readonly session: {
    readonly durationMs: number;
    readonly frameCount: number;
    readonly cameraFacing: "user" | "environment";
    readonly poseFrames: readonly ResearchPoseFrame[];
    readonly videoFrameTimings: readonly VideoFrameTimingSample[];
    readonly intervals: readonly CaptureInterval[];
    readonly video: {
      readonly path: string;
      readonly fileName: string;
      readonly mimeType: string;
      readonly bytes: number;
      readonly sha256: string;
    };
  };
  readonly evidenceBoundary: "independent-observable-motion-review-not-clinical-truth";
}

export interface IndependentReviewSidecar {
  readonly schemaVersion: typeof INDEPENDENT_REVIEW_SIDECAR_SCHEMA;
  readonly evidenceClass: "HUMAN_INDEPENDENT_OBSERVABLE_MOTION_REVIEW";
  readonly admissionEffect: "NONE_REMAINS_QUARANTINED";
  readonly blindingStatus:
    "PROCEDURAL_SELF_ATTESTED_UI_BLINDING_NOT_SEALED_PACKAGE";
  readonly createdAt: string;
  readonly sourceEpisode: {
    readonly episodeJsonPath: "episode.json";
    readonly episodeJsonSha256: string;
    readonly episodeId: string;
    readonly episodeAncestorId: string;
    readonly sourceAdmissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW";
  };
  readonly appendOnly: true;
  readonly replayDeclaration: IndependentReviewerReplayDeclaration;
  readonly reviewEntry: ResearchAnnotationReviewEntry;
  readonly derivedReviewState: ResearchAnnotationReviewResult;
  readonly clinicalTruthStatus: "not-established-by-review";
  readonly claimBoundary: readonly string[];
}

export interface IndependentReviewSidecarExport {
  readonly sidecar: IndependentReviewSidecar;
  readonly json: string;
  readonly sha256: string;
  readonly sha256Sidecar: string;
  readonly fileName: string;
}

export interface IndependentReviewDeclarations {
  readonly blinded: boolean;
  readonly independentlyProduced: boolean;
  readonly replay: IndependentReviewerReplayDeclarationInput;
}

export interface IndependentReviewerReplayDeclarationInput {
  readonly watchedContinuousVideo: boolean;
  readonly overlaysVisible: boolean;
  readonly referenceMarkerInspected: boolean;
  readonly attemptMarkerInspected: boolean;
  readonly encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED";
}

export interface IndependentReviewerReplayDeclaration
  extends IndependentReviewerReplayDeclarationInput {
  readonly episodeJsonSha256: string;
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly raterCode: string;
  readonly declaredAt: string;
  readonly evidenceScope:
    "reviewer-replay-declaration-not-motion-label-or-clinical-truth";
}

export interface PostSubmissionCollectorComparison {
  readonly collectorAnnotation: ResearchAnnotation;
  readonly independentAnnotation: ResearchAnnotation;
  readonly disagreements: readonly ResearchAnnotationDisagreement[];
  readonly derivedReviewState: ResearchAnnotationReviewResult;
  readonly interpretation:
    "collector-comparison-only-not-consensus-adjudication-or-clinical-truth";
}

function record(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function safeArchiveMemberPath(path: string): boolean {
  if (!path || path !== path.trim() || path.includes("\\") || path.startsWith("/")) {
    return false;
  }
  const segments = path.split("/");
  return segments.every(
    (segment) =>
      segment.length > 0 &&
      segment !== "." &&
      segment !== ".." &&
      !/[\u0000-\u001f\u007f]/.test(segment),
  );
}

function safeVideoFileName(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value === value.trim() &&
    !value.includes("/") &&
    !value.includes("\\") &&
    value !== "." &&
    value !== ".." &&
    !/[\u0000-\u001f\u007f]/.test(value)
  );
}

function finiteNonnegative(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function positiveSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function decodeUtf8(bytes: Uint8Array, label: string): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new ResearchReviewValidationError([`${label} is not valid UTF-8.`]);
  }
}

function parseChecksumManifest(bytes: Uint8Array): Map<string, string> {
  const text = decodeUtf8(bytes, CHECKSUM_PATH);
  const entries = new Map<string, string>();
  const issues: string[] = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (line === "") continue;
    const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
    if (!match) {
      issues.push(`${CHECKSUM_PATH} line ${index + 1} is malformed.`);
      continue;
    }
    const [, sha256, path] = match;
    if (!safeArchiveMemberPath(path)) {
      issues.push(`${CHECKSUM_PATH} line ${index + 1} has an unsafe member path.`);
      continue;
    }
    if (path === CHECKSUM_PATH) {
      issues.push(`${CHECKSUM_PATH} cannot contain a circular checksum of itself.`);
      continue;
    }
    if (entries.has(path)) {
      issues.push(`${CHECKSUM_PATH} repeats ${path}.`);
      continue;
    }
    entries.set(path, sha256);
  }
  if (entries.size === 0) issues.push(`${CHECKSUM_PATH} has no member checksums.`);
  if (issues.length > 0) throw new ResearchReviewValidationError(issues);
  return entries;
}

function intervalLooksReplayable(value: unknown, expectedKind: string): boolean {
  const interval = record(value);
  const begin = record(interval?.begin);
  const end = record(interval?.end);
  return Boolean(
    interval?.kind === expectedKind &&
      finiteNonnegative(begin?.mediaTimeMs) &&
      finiteNonnegative(end?.mediaTimeMs) &&
      (end?.mediaTimeMs as number) > (begin?.mediaTimeMs as number),
  );
}

function validateEpisodeForBlindedReview(
  parsed: unknown,
  memberPaths: readonly string[],
  checksums: ReadonlyMap<string, string>,
): { episode: ResearchBundle; videoPath: string } {
  const issues: string[] = [];
  const episode = record(parsed);
  if (!episode) {
    throw new ResearchReviewValidationError(["episode.json must contain an object."]);
  }
  if (episode.schemaVersion !== "relay.body-native-episode/2") {
    issues.push("Only relay.body-native-episode/2 is accepted.");
  }
  if (episode.admissionStatus !== "QUARANTINED_PENDING_INDEPENDENT_REVIEW") {
    issues.push("The source episode must remain quarantined pending independent review.");
  }

  const metadata = record(episode.metadata);
  const lineage = record(episode.lineage);
  const session = record(episode.session);
  const video = record(session?.video);
  if (typeof metadata?.episodeCode !== "string" || !metadata.episodeCode.trim()) {
    issues.push("The episode ID is missing.");
  }
  if (
    typeof lineage?.episodeAncestorId !== "string" ||
    !lineage.episodeAncestorId.trim()
  ) {
    issues.push("The episode ancestor ID is missing.");
  }
  if (!safeVideoFileName(video?.fileName)) {
    issues.push("The continuous-video filename declaration is invalid.");
  }
  if (
    typeof video?.mimeType !== "string" ||
    !/^video\/[a-z0-9][a-z0-9!#$&^_.+\-]*(?:\s*;.+)?$/i.test(video.mimeType)
  ) {
    issues.push("The continuous-video MIME declaration is invalid.");
  }
  if (!positiveSafeInteger(video?.bytes)) {
    issues.push("The continuous-video byte declaration is invalid.");
  }
  if (typeof video?.sha256 !== "string" || !SHA256_PATTERN.test(video.sha256)) {
    issues.push("The continuous-video SHA-256 declaration is invalid.");
  }
  if (video?.continuousEpisode !== true) {
    issues.push("The video must be declared as the one continuous episode.");
  }

  const videoPath = safeVideoFileName(video?.fileName)
    ? `continuous/${video.fileName}`
    : "continuous/(invalid)";
  const continuousMembers = memberPaths.filter((path) =>
    path.startsWith("continuous/"),
  );
  if (continuousMembers.length !== 1 || continuousMembers[0] !== videoPath) {
    issues.push("The archive must contain exactly the declared continuous video.");
  }
  const expectedMembers = [
    "CLAIM_BOUNDARY.txt",
    "annotations/collector.json",
    "continuous.pose.json",
    "continuous.video-timing.json",
    "derived/aids.json",
    "environment.json",
    EPISODE_PATH,
    "protocol-execution.json",
    "protocol.json",
    "replay-audit.json",
    "segments.json",
    videoPath,
  ].sort();
  const observedMembers = [...memberPaths].sort();
  for (const path of expectedMembers) {
    if (!observedMembers.includes(path)) {
      issues.push(`Required v2 archive member ${path} is missing.`);
    }
  }
  for (const path of observedMembers) {
    if (!expectedMembers.includes(path)) {
      issues.push(`Unexpected v2 archive member ${path} is prohibited.`);
    }
  }
  if (
    typeof video?.sha256 === "string" &&
    checksums.get(videoPath) !== video.sha256.toLowerCase()
  ) {
    issues.push("The continuous-video declaration does not match its member checksum.");
  }

  const digests = Array.isArray(episode.contentDigests)
    ? episode.contentDigests
    : [];
  const videoDigests = digests.filter((value) => record(value)?.path === videoPath);
  if (videoDigests.length !== 1) {
    issues.push("The continuous video must have exactly one episode content-digest declaration.");
  } else {
    const digest = record(videoDigests[0]);
    if (
      digest?.bytes !== video?.bytes ||
      digest?.sha256 !== video?.sha256
    ) {
      issues.push("The episode content digest does not exactly match the video declaration.");
    }
  }

  if (!Array.isArray(session?.poseFrames)) {
    issues.push("The continuous pose trace is missing.");
  }
  if (!Array.isArray(session?.videoFrameTimings)) {
    issues.push("The continuous video timing trace is missing.");
  }
  const intervals = Array.isArray(session?.intervals) ? session.intervals : [];
  if (
    intervals.length < 2 ||
    !intervalLooksReplayable(intervals[0], "reference") ||
    !intervalLooksReplayable(intervals[1], "attempt")
  ) {
    issues.push("Ordered, positive reference and attempt intervals are required.");
  }
  if (!finiteNonnegative(session?.durationMs) || session?.durationMs === 0) {
    issues.push("The continuous session duration is invalid.");
  }
  if (!Number.isSafeInteger(session?.frameCount) || (session?.frameCount as number) < 0) {
    issues.push("The pose-frame count declaration is invalid.");
  }
  if (
    Array.isArray(session?.poseFrames) &&
    session?.frameCount !== session.poseFrames.length
  ) {
    issues.push("The pose-frame count does not match the continuous pose trace.");
  }

  const replayAudit = record(session?.replayAudit);
  if (
    replayAudit?.status !== "PASS" ||
    replayAudit?.videoAssetSha256 !== video?.sha256
  ) {
    issues.push("The episode replay audit must pass and bind the declared video.");
  }
  const integrityAudit = record(replayAudit?.integrityAudit);
  if (
    integrityAudit?.valid !== true ||
    integrityAudit?.status !== "REPLAY_INTEGRITY_CONFIRMED"
  ) {
    issues.push(
      "The retained replay-integrity audit must be present and confirmed; a top-level PASS string is insufficient.",
    );
  }
  const decodedValidation = record(integrityAudit?.decodedVideo);
  const decodedObserved = record(decodedValidation?.observed);
  if (decodedValidation?.valid !== true) {
    issues.push(
      "The retained replay audit must contain a passing decoded-video validation result.",
    );
  }
  if (Array.isArray(session?.videoFrameTimings)) {
    const recomputedDecode = validateDecodedContinuousVideoEvidence(
      decodedObserved as unknown as DecodedContinuousVideoEvidence,
      session.videoFrameTimings as unknown as VideoFrameTimingSample[],
    );
    if (!recomputedDecode.valid) {
      issues.push(
        ...recomputedDecode.issues.map(
          (entry) => `Decoded-video validation failed: ${entry.message}`,
        ),
      );
    }
  }
  if (!record(session?.protocolExecution)) {
    issues.push("The retained protocol execution is missing.");
  }
  if (!record(episode.protocol)) {
    issues.push("The frozen protocol is missing.");
  }

  if (
    Array.isArray(session?.poseFrames) &&
    Array.isArray(session?.videoFrameTimings) &&
    Array.isArray(session?.intervals)
  ) {
    try {
      const typed = episode as unknown as ResearchBundle;
      const causalTiming = validateCaptureTimingSession({
        videoFrames: typed.session.videoFrameTimings,
        poses: typed.session.poseFrames.map((frame) => ({
          sequence: frame.sequenceNumber,
          callbackMonotonicTimeMs: frame.callbackMonotonicTimeMs,
        })),
        intervals: typed.session.intervals,
      });
      if (!causalTiming.valid) {
        issues.push(
          ...causalTiming.issues.map(
            (entry) => `Continuous timing validation failed: ${entry.message}`,
          ),
        );
      }

      const storedAttachments: ReplayPoseAttachment<{
        sequence: number;
        callbackMonotonicTimeMs: number;
      }>[] = [];
      for (const [index, frame] of typed.session.poseFrames.entries()) {
        const stored = frame.videoTiming;
        const recomputed = causalTiming.poseTiming.attachments[index];
        if (!stored || !recomputed) {
          issues.push(
            `Pose frame ${index} is missing a complete causal video attachment.`,
          );
          continue;
        }
        if (
          stored.strategy !== recomputed.strategy ||
          stored.sequence !== recomputed.videoFrame.sequence ||
          stored.callbackMonotonicTimeMs !==
            recomputed.videoFrame.callbackMonotonicTimeMs ||
          stored.mediaTimeMs !== recomputed.videoFrame.mediaTimeMs ||
          stored.presentedFrames !== recomputed.videoFrame.presentedFrames
        ) {
          issues.push(
            `Pose frame ${index} retains a stale or non-causal video attachment.`,
          );
        }
        storedAttachments.push({
          strategy: stored.strategy,
          pose: {
            sequence: frame.sequenceNumber,
            callbackMonotonicTimeMs: frame.callbackMonotonicTimeMs,
          },
          videoFrame: {
            sequence: stored.sequence,
            callbackMonotonicTimeMs: stored.callbackMonotonicTimeMs,
            mediaTimeMs: stored.mediaTimeMs,
            presentedFrames: stored.presentedFrames,
          },
        });
      }
      if (storedAttachments.length !== typed.session.poseFrames.length) {
        issues.push("Every retained pose frame must have one causal video attachment.");
      }
      const replayCoverage = auditReplayIntervalBoundaryCoverage(
        typed.session.intervals,
        typed.session.videoFrameTimings,
        storedAttachments,
      );
      if (!replayCoverage.valid) {
        issues.push(
          ...replayCoverage.issues.map(
            (entry) => `Replay coverage validation failed: ${entry.message}`,
          ),
        );
      }
    } catch {
      issues.push("The retained continuous timing or causal replay trace is malformed.");
    }
  }

  const annotations = Array.isArray(episode.annotations) ? episode.annotations : [];
  if (annotations.length === 0) {
    issues.push("At least one collector annotation is required.");
  }
  for (const [index, value] of annotations.entries()) {
    const annotation = record(value);
    if (
      annotation?.raterRole !== "collector" ||
      annotation?.reviewStatus !== "draft-unreviewed" ||
      typeof annotation?.raterCode !== "string" ||
      !annotation.raterCode.trim()
    ) {
      issues.push(`annotations[${index}] is not a draft collector annotation.`);
    }
  }

  if (issues.length > 0) throw new ResearchReviewValidationError(issues);
  return { episode: episode as unknown as ResearchBundle, videoPath };
}

/**
 * Verify the exact uncompressed member set before parsing or displaying any
 * episode content. Every member except the necessarily external checksum file
 * must appear once in SHA256SUMS and match a fresh browser SHA-256.
 */
export async function verifyResearchReviewArchive(
  archiveName: string,
  members: Readonly<Record<string, Uint8Array>>,
): Promise<VerifiedResearchReviewArchive> {
  const issues: string[] = [];
  if (!archiveName.endsWith(ARCHIVE_SUFFIX)) {
    issues.push(`Choose one ${ARCHIVE_SUFFIX} archive.`);
  }
  const memberPaths = Object.keys(members);
  for (const path of memberPaths) {
    if (!safeArchiveMemberPath(path)) issues.push(`Unsafe archive member path: ${path}.`);
  }
  if (!members[EPISODE_PATH]) issues.push(`${EPISODE_PATH} is required.`);
  if (!members[CHECKSUM_PATH]) issues.push(`${CHECKSUM_PATH} is required.`);
  if (issues.length > 0) throw new ResearchReviewValidationError(issues);

  const declared = parseChecksumManifest(members[CHECKSUM_PATH]);
  const actualPaths = memberPaths.filter((path) => path !== CHECKSUM_PATH).sort();
  const declaredPaths = Array.from(declared.keys()).sort();
  for (const path of actualPaths) {
    if (!declared.has(path)) issues.push(`Archive member ${path} is not checksum-bound.`);
  }
  for (const path of declaredPaths) {
    if (!members[path]) issues.push(`Checksum-bound member ${path} is missing.`);
  }
  if (issues.length > 0) throw new ResearchReviewValidationError(issues);

  const verifiedChecksums: ResearchReviewChecksum[] = [];
  for (const path of actualPaths) {
    const observed = await sha256Bytes(members[path]);
    const expected = declared.get(path);
    if (observed !== expected) {
      issues.push(`SHA-256 mismatch for ${path}.`);
    }
    verifiedChecksums.push({ path, sha256: observed, bytes: members[path].byteLength });
  }
  if (issues.length > 0) throw new ResearchReviewValidationError(issues);

  let parsed: unknown;
  try {
    parsed = JSON.parse(decodeUtf8(members[EPISODE_PATH], EPISODE_PATH));
  } catch (error) {
    if (error instanceof ResearchReviewValidationError) throw error;
    throw new ResearchReviewValidationError([`${EPISODE_PATH} is not valid JSON.`]);
  }
  const { episode, videoPath } = validateEpisodeForBlindedReview(
    parsed,
    actualPaths,
    declared,
  );
  const videoBytes = members[videoPath];
  const declaredVideo = episode.session.video;
  if (
    !videoBytes ||
    videoBytes.byteLength !== declaredVideo.bytes ||
    (await sha256Bytes(videoBytes)) !== declaredVideo.sha256
  ) {
    throw new ResearchReviewValidationError([
      "The continuous-video bytes do not exactly match the episode declaration.",
    ]);
  }

  const episodeChecksum = verifiedChecksums.find(
    (entry) => entry.path === EPISODE_PATH,
  );
  if (!episodeChecksum) {
    throw new ResearchReviewValidationError(["The episode JSON checksum is missing."]);
  }
  return {
    archiveName,
    episode,
    episodeJsonSha256: episodeChecksum.sha256,
    episodeJsonBytes: episodeChecksum.bytes,
    videoPath,
    videoBytes: Uint8Array.from(videoBytes),
    checksums: verifiedChecksums,
  };
}

export function buildBlindedResearchReviewView(
  verified: VerifiedResearchReviewArchive,
): BlindedResearchReviewView {
  const { episode } = verified;
  return {
    schemaVersion: "relay.body-native-episode/2",
    admissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW",
    episodeId: episode.metadata.episodeCode,
    episodeAncestorId: episode.lineage.episodeAncestorId,
    episodeJsonSha256: verified.episodeJsonSha256,
    session: {
      durationMs: episode.session.durationMs,
      frameCount: episode.session.frameCount,
      cameraFacing: episode.session.cameraFacing,
      poseFrames: episode.session.poseFrames,
      videoFrameTimings: episode.session.videoFrameTimings,
      intervals: episode.session.intervals,
      video: {
        path: verified.videoPath,
        fileName: episode.session.video.fileName,
        mimeType: episode.session.video.mimeType,
        bytes: episode.session.video.bytes,
        sha256: episode.session.video.sha256,
      },
    },
    evidenceBoundary: "independent-observable-motion-review-not-clinical-truth",
  };
}

function identifier(value: string): string {
  return value.trim().toLocaleLowerCase("en-US");
}

function annotationIssues(
  annotation: ResearchAnnotation,
  sessionDurationMs: number,
): string[] {
  const issues: string[] = [];
  if (!annotation.annotationId.trim()) issues.push("Annotation ID is required.");
  if (!annotation.raterCode.trim()) issues.push("Reviewer code is required.");
  if (annotation.raterRole !== "independent-reviewer") {
    issues.push("The sidecar annotation must have the independent-reviewer role.");
  }
  if (annotation.reviewStatus !== "draft-unreviewed") {
    issues.push("Review state is derived; the annotation must remain draft-unreviewed.");
  }
  if (!annotation.movementIdentity.trim()) issues.push("Movement identity is required.");
  if (
    !Number.isFinite(annotation.confidence) ||
    annotation.confidence < 0 ||
    annotation.confidence > 1
  ) {
    issues.push("Confidence must be finite and between 0 and 1.");
  }
  if (!Number.isFinite(Date.parse(annotation.createdAt))) {
    issues.push("Annotation creation time must be a valid timestamp.");
  }
  if (
    annotation.assessment !== "uncertain" &&
    annotation.assessment !== "unobservable"
  ) {
    issues.push(
      "This pilot sidecar accepts only a qualitative uncertain observation or an unobservable abstention.",
    );
  }
  if (
    annotation.phaseStartMs !== undefined ||
    annotation.phaseEndMs !== undefined ||
    annotation.metric !== undefined ||
    annotation.observedValue !== undefined ||
    annotation.targetValue !== undefined ||
    annotation.delta !== undefined ||
    annotation.unit !== undefined ||
    annotation.cue !== undefined ||
    annotation.protocolSha256 !== undefined ||
    annotation.protocolExecutionSha256 !== undefined ||
    annotation.referenceEvidenceSequenceNumber !== undefined ||
    annotation.attemptEvidenceSequenceNumber !== undefined
  ) {
    issues.push(
      "This qualitative pilot sidecar prohibits phase claims, metric/value/delta fields, correction cues, and protocol-derived evidence bindings.",
    );
  }
  const numericFields = [
    "phaseStartMs",
    "phaseEndMs",
    "evidenceStartMs",
    "evidenceEndMs",
    "observedValue",
    "targetValue",
    "delta",
  ] as const;
  for (const field of numericFields) {
    const value = annotation[field];
    if (value !== undefined && !Number.isFinite(value)) {
      issues.push(`${field} must be finite when supplied.`);
    }
  }
  for (const field of [
    "phaseStartMs",
    "phaseEndMs",
    "evidenceStartMs",
    "evidenceEndMs",
  ] as const) {
    const value = annotation[field];
    if (value !== undefined && (value < 0 || value > sessionDurationMs)) {
      issues.push(`${field} must fall within the continuous episode duration.`);
    }
  }
  if (
    (annotation.phaseStartMs === undefined) !==
    (annotation.phaseEndMs === undefined)
  ) {
    issues.push("Phase start and end must be supplied together.");
  }
  if (
    annotation.phaseStartMs !== undefined &&
    annotation.phaseEndMs !== undefined &&
    annotation.phaseEndMs < annotation.phaseStartMs
  ) {
    issues.push("Phase end cannot precede phase start.");
  }
  if (
    (annotation.evidenceStartMs === undefined) !==
    (annotation.evidenceEndMs === undefined)
  ) {
    issues.push("Evidence start and end must be supplied together.");
  }
  if (
    annotation.evidenceStartMs !== undefined &&
    annotation.evidenceEndMs !== undefined &&
    annotation.evidenceEndMs < annotation.evidenceStartMs
  ) {
    issues.push("Evidence end cannot precede evidence start.");
  }
  if (new Set(annotation.evidenceJoints).size !== annotation.evidenceJoints.length) {
    issues.push("Evidence joints cannot contain duplicates.");
  }
  if (annotation.evidenceJoints.some((joint) => !REVIEW_JOINT_SET.has(joint))) {
    issues.push("The annotation contains an unknown evidence joint.");
  }

  if (annotation.assessment === "uncertain") {
    if (!annotation.realizedErrorType?.trim()) {
      issues.push(
        "A qualitative reference-relative observation is required for an observable uncertain review.",
      );
    }
    if (annotation.evidenceJoints.length === 0) {
      issues.push("At least one evidence joint is required for an observable review.");
    }
    if (
      annotation.evidenceStartMs === undefined ||
      annotation.evidenceEndMs === undefined
    ) {
      issues.push("An evidence interval is required for an observable review.");
    }
    if (annotation.evidenceClock !== "encoded-replay-approximate-ms") {
      issues.push(
        "Qualitative review timing must be tagged as approximate encoded-replay milliseconds until source-to-Blob calibration exists.",
      );
    }
    if (annotation.desiredTargetSource !== "captured-reference") {
      issues.push(
        "The qualitative observable review must name only the captured reference as its comparator source.",
      );
    }
  }

  if (
    annotation.observedValue !== undefined &&
    annotation.targetValue !== undefined &&
    annotation.delta !== undefined
  ) {
    const expectedDelta = annotation.targetValue - annotation.observedValue;
    const scale = Math.max(
      1,
      Math.abs(annotation.observedValue),
      Math.abs(annotation.targetValue),
      Math.abs(annotation.delta),
    );
    const floatingPointTolerance = Number.EPSILON * 16 * scale;
    if (Math.abs(annotation.delta - expectedDelta) > floatingPointTolerance) {
      issues.push("Delta must equal target value minus observed value.");
    }
  }

  if (annotation.assessment === "unobservable") {
    if (!annotation.abstainReason?.trim()) {
      issues.push("An abstention reason is required for an unobservable review.");
    }
    if (
      annotation.metric !== undefined ||
      annotation.observedValue !== undefined ||
      annotation.targetValue !== undefined ||
      annotation.delta !== undefined ||
      annotation.unit !== undefined ||
      annotation.cue !== undefined
    ) {
      issues.push("An unobservable review must omit metric, values, unit, and cue.");
    }
    if (annotation.desiredTargetSource !== "unknown") {
      issues.push("An unobservable review must declare the desired target source unknown.");
    }
  }
  return issues;
}

function immutableAnnotation(annotation: ResearchAnnotation): ResearchAnnotation {
  const copy: ResearchAnnotation = {
    ...annotation,
    annotationId: annotation.annotationId.trim(),
    raterCode: annotation.raterCode.trim(),
    movementIdentity: annotation.movementIdentity.trim(),
    evidenceJoints: [...annotation.evidenceJoints],
    realizedErrorType: annotation.realizedErrorType?.trim() || undefined,
    metric: annotation.metric?.trim() || undefined,
    unit: annotation.unit?.trim() || undefined,
    cue: annotation.cue?.trim() || undefined,
    abstainReason: annotation.abstainReason?.trim() || undefined,
    notes: annotation.notes?.trim() || undefined,
  };
  Object.freeze(copy.evidenceJoints);
  return Object.freeze(copy);
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value as UnknownRecord)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

function safeName(value: string): string {
  return value.trim().replace(/[^a-zA-Z0-9_-]+/g, "-") || "review";
}

export async function buildIndependentReviewSidecarExport(
  verified: VerifiedResearchReviewArchive,
  submittedAnnotation: ResearchAnnotation,
  declarations: IndependentReviewDeclarations,
): Promise<IndependentReviewSidecarExport> {
  const issues = annotationIssues(
    submittedAnnotation,
    verified.episode.session.durationMs,
  );
  if (!declarations.blinded || !declarations.independentlyProduced) {
    issues.push(
      "Both procedural UI-blinding and independently produced self-attestations are required before submission.",
    );
  }
  if (
    !declarations.replay?.watchedContinuousVideo ||
    !declarations.replay?.overlaysVisible ||
    !declarations.replay?.referenceMarkerInspected ||
    !declarations.replay?.attemptMarkerInspected
  ) {
    issues.push(
      "The reviewer must declare continuous replay, visible overlays, and inspection of both interval markers.",
    );
  }
  if (
    declarations.replay?.encodedOverlayAlignment !==
    "APPROXIMATE_UNCALIBRATED"
  ) {
    issues.push(
      "The reviewer must retain the approximate, uncalibrated encoded-overlay boundary.",
    );
  }
  const collectorAnnotations = verified.episode.annotations.filter(
    (annotation) => annotation.raterRole === "collector",
  );
  if (
    collectorAnnotations.some(
      (annotation) =>
        identifier(annotation.raterCode) === identifier(submittedAnnotation.raterCode),
    )
  ) {
    issues.push("Reviewer code must be distinct from every collector rater code.");
  }
  if (
    collectorAnnotations.some(
      (annotation) =>
        identifier(annotation.annotationId) === identifier(submittedAnnotation.annotationId),
    )
  ) {
    issues.push("Independent annotation ID must be distinct from collector annotations.");
  }
  if (issues.length > 0) throw new ResearchReviewValidationError(issues);

  const annotation = immutableAnnotation(submittedAnnotation);
  const declaredAt = annotation.createdAt;
  const declaration: BlindedIndependentReviewDeclaration = Object.freeze({
    annotationId: annotation.annotationId,
    raterCode: annotation.raterCode,
    episodeId: verified.episode.metadata.episodeCode,
    episodeAncestorId: verified.episode.lineage.episodeAncestorId,
    blinded: true,
    independentlyProduced: true,
    declaredAt,
  });
  const reviewEntry: ResearchAnnotationReviewEntry = Object.freeze({
    episodeId: verified.episode.metadata.episodeCode,
    episodeAncestorId: verified.episode.lineage.episodeAncestorId,
    annotation,
    independentReviewDeclaration: declaration,
  });
  const replayDeclaration: IndependentReviewerReplayDeclaration = Object.freeze({
    episodeJsonSha256: verified.episodeJsonSha256,
    episodeId: verified.episode.metadata.episodeCode,
    episodeAncestorId: verified.episode.lineage.episodeAncestorId,
    raterCode: annotation.raterCode,
    declaredAt,
    watchedContinuousVideo: true,
    overlaysVisible: true,
    referenceMarkerInspected: true,
    attemptMarkerInspected: true,
    encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
    evidenceScope:
      "reviewer-replay-declaration-not-motion-label-or-clinical-truth",
  });
  const derivedReviewState = deriveResearchAnnotationReview([reviewEntry]);
  const sidecar = deepFreeze<IndependentReviewSidecar>({
    schemaVersion: INDEPENDENT_REVIEW_SIDECAR_SCHEMA,
    evidenceClass: "HUMAN_INDEPENDENT_OBSERVABLE_MOTION_REVIEW",
    admissionEffect: "NONE_REMAINS_QUARANTINED",
    blindingStatus:
      "PROCEDURAL_SELF_ATTESTED_UI_BLINDING_NOT_SEALED_PACKAGE",
    createdAt: declaredAt,
    sourceEpisode: {
      episodeJsonPath: "episode.json",
      episodeJsonSha256: verified.episodeJsonSha256,
      episodeId: verified.episode.metadata.episodeCode,
      episodeAncestorId: verified.episode.lineage.episodeAncestorId,
      sourceAdmissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW",
    },
    appendOnly: true,
    replayDeclaration,
    reviewEntry,
    derivedReviewState,
    clinicalTruthStatus: "not-established-by-review",
    claimBoundary: [
      "This sidecar records one independent observable-motion judgment; it does not mutate the source episode.",
      "Blinding is procedural and self-attested at the review UI: the source archive contains collector/protocol material and is not a cryptographically sealed blind package.",
      "One independent review still requires a second procedurally blinded review and, if they disagree, third-rater adjudication.",
      "The encoded-video marker and pose-overlay alignment is approximate because the source-to-Blob timeline has not been calibrated.",
      "Agreement or adjudication does not establish clinical, safety, diagnostic, pain, force, load, contact, or intent truth.",
      "The source episode remains quarantined; admission is a separate fail-closed decision.",
    ],
  });
  const json = `${JSON.stringify(sidecar, null, 2)}\n`;
  const sha256 = await sha256Text(json);
  const fileName = `${safeName(sidecar.sourceEpisode.episodeId)}-${safeName(
    annotation.raterCode,
  )}.independent-review.json`;
  return {
    sidecar,
    json,
    sha256,
    sha256Sidecar: `${sha256}  ${fileName}\n`,
    fileName,
  };
}

/**
 * This function is intentionally separate from sidecar creation. Calling it is
 * the post-commit reveal: comparison never contributes to the submitted label.
 */
export function revealPostSubmissionCollectorComparison(
  verified: VerifiedResearchReviewArchive,
  sidecar: IndependentReviewSidecar,
): PostSubmissionCollectorComparison {
  const source = sidecar.sourceEpisode;
  const issues: string[] = [];
  if (source.episodeJsonSha256 !== verified.episodeJsonSha256) {
    issues.push("The review sidecar references a different episode JSON hash.");
  }
  if (source.episodeId !== verified.episode.metadata.episodeCode) {
    issues.push("The review sidecar references a different episode ID.");
  }
  if (source.episodeAncestorId !== verified.episode.lineage.episodeAncestorId) {
    issues.push("The review sidecar references a different episode ancestor.");
  }
  if (
    sidecar.reviewEntry.episodeId !== source.episodeId ||
    sidecar.reviewEntry.episodeAncestorId !== source.episodeAncestorId
  ) {
    issues.push("The append-only review entry does not match its sidecar lineage.");
  }
  const collector = verified.episode.annotations.find(
    (annotation) => annotation.raterRole === "collector",
  );
  if (!collector) issues.push("A collector annotation is required for comparison.");
  if (issues.length > 0 || !collector) {
    throw new ResearchReviewValidationError(issues);
  }
  const independent = sidecar.reviewEntry.annotation as ResearchAnnotation;
  return {
    collectorAnnotation: collector,
    independentAnnotation: independent,
    disagreements: compareResearchAnnotations(collector, independent),
    derivedReviewState: deriveResearchAnnotationReview([sidecar.reviewEntry]),
    interpretation:
      "collector-comparison-only-not-consensus-adjudication-or-clinical-truth",
  };
}
