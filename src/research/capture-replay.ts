/**
 * Replay integrity proves only that a local continuous-video asset, its causal
 * pose overlays, and its declared intervals were replayed consistently. It is
 * provenance evidence, not evidence that a movement label or clinical claim is
 * true.
 */
export const REPLAY_INTEGRITY_EVIDENCE_SCOPE =
  "capture-replay-integrity-only-not-movement-or-annotation-truth" as const;

export const REPLAY_INTEGRITY_CLAIM_BOUNDARY =
  "Replay integrity confirms local asset identity, source-clock causal pose attachment, approximate interval visibility on the encoded replay, and a human declaration; the MediaRecorder source-to-blob time offset is uncalibrated, and it does not establish movement, annotation, clinical, or safety truth.";

export const RECORDED_BLOB_TIMELINE_MAPPING =
  "UNCALIBRATED_MEDIARECORDER_START_OFFSET" as const;

export interface DeclaredContinuousVideoAsset {
  readonly fileName: string;
  readonly mimeType: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface LocalContinuousVideoBlob {
  readonly fileName: string;
  readonly blob: Blob;
}

export interface ObservedContinuousVideoAsset {
  readonly fileName: string;
  readonly mimeType: string;
  readonly bytes: number;
  readonly sha256: string;
}

/**
 * Browser-observed decode evidence. Hashing bytes is not proof that a media
 * container decodes, and a MIME string is not a decoder result.
 */
export interface DecodedContinuousVideoEvidence {
  readonly status: "DECODED_LOCAL_VIDEO";
  readonly durationMs: number;
  readonly videoWidth: number;
  readonly videoHeight: number;
  readonly seekableStartMs: number;
  readonly seekableEndMs: number;
  /** The decoded frame visible when the inspection ran; not necessarily frame 0. */
  readonly observedDecodedFrameMediaTimeMs: number;
  /** Source rVFC time zero has not yet been calibrated to encoded Blob time zero. */
  readonly sourceTimelineMapping: typeof RECORDED_BLOB_TIMELINE_MAPPING;
  readonly inspectedAt: string;
}

export interface ReplayVideoTimingSample {
  readonly sequence: number;
  readonly callbackMonotonicTimeMs: number;
  readonly mediaTimeMs: number;
  readonly presentedFrames?: number;
}

export interface ReplayPoseTimingSample {
  readonly sequence: number;
  readonly callbackMonotonicTimeMs: number;
}

/** Structurally compatible with capture-timing's causal attachment. */
export interface ReplayPoseAttachment<
  TPose extends ReplayPoseTimingSample = ReplayPoseTimingSample,
> {
  readonly strategy: "nearest-latest-causal-video-frame";
  readonly pose: TPose;
  readonly videoFrame: ReplayVideoTimingSample;
}

export type ReplayCaptureIntervalKind = "reference" | "attempt" | "revision";

export interface ReplayCaptureInterval {
  readonly kind: ReplayCaptureIntervalKind;
  readonly begin: ReplayVideoTimingSample;
  readonly end: ReplayVideoTimingSample;
}

export interface HumanReplayConfirmation {
  readonly reviewerCode: string;
  readonly watchedContinuousVideo: boolean;
  readonly overlaysVisible: boolean;
  readonly referenceIntervalConfirmed: boolean;
  readonly attemptIntervalConfirmed: boolean;
  readonly declaredAt: string;
  readonly evidenceScope: typeof REPLAY_INTEGRITY_EVIDENCE_SCOPE;
}

/**
 * Observe actual browser media decode before accepting a replay declaration.
 * This helper intentionally lives beside the validator so callers cannot
 * synthesize the evidence from Blob metadata alone.
 */
export async function observeDecodedContinuousVideo(
  video: HTMLVideoElement,
): Promise<DecodedContinuousVideoEvidence> {
  const ready = async () => {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) return;
    await new Promise<void>((resolve, reject) => {
      const onReady = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error("The retained continuous video did not decode locally."));
      };
      const cleanup = () => {
        video.removeEventListener("loadedmetadata", onReady);
        video.removeEventListener("error", onError);
      };
      video.addEventListener("loadedmetadata", onReady, { once: true });
      video.addEventListener("error", onError, { once: true });
      video.load();
    });
  };
  await ready();
  if (!Number.isFinite(video.duration) || video.duration <= 0) {
    throw new Error("The decoded continuous video has no finite positive duration.");
  }
  const seekableStartMs = video.seekable.length > 0 ? video.seekable.start(0) * 1000 : 0;
  const seekableEndMs =
    video.seekable.length > 0
      ? video.seekable.end(video.seekable.length - 1) * 1000
      : video.duration * 1000;
  let observedDecodedFrameMediaTimeMs = video.currentTime * 1000;
  // `loadeddata`/HAVE_CURRENT_DATA already proves that the current frame was
  // decoded. Requesting a *future* callback while an ended or paused replay is
  // being confirmed can never resolve, so only wait when no current frame is
  // available yet.
  if (
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA &&
    typeof video.requestVideoFrameCallback === "function"
  ) {
    observedDecodedFrameMediaTimeMs = await new Promise<number>((resolve, reject) => {
      const timeout = window.setTimeout(
        () => reject(new Error("The retained video produced no decoded frame callback.")),
        2000,
      );
      video.requestVideoFrameCallback((_now, metadata) => {
        window.clearTimeout(timeout);
        resolve(metadata.mediaTime * 1000);
      });
    });
  }
  return {
    status: "DECODED_LOCAL_VIDEO",
    durationMs: video.duration * 1000,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    seekableStartMs,
    seekableEndMs,
    observedDecodedFrameMediaTimeMs,
    sourceTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
    inspectedAt: new Date().toISOString(),
  };
}

export type ReplayIntegrityIssueCode =
  | "missing-video-declaration"
  | "missing-video-blob"
  | "invalid-file-name"
  | "file-name-mismatch"
  | "invalid-mime-type"
  | "mime-type-mismatch"
  | "invalid-byte-length"
  | "byte-length-mismatch"
  | "invalid-sha256"
  | "sha256-mismatch"
  | "blob-read-failed"
  | "blob-read-size-mismatch"
  | "hash-unavailable"
  | "missing-decoded-video-evidence"
  | "invalid-decoded-video-evidence"
  | "decoded-video-does-not-cover-timing"
  | "missing-video-timing"
  | "invalid-video-timing"
  | "video-sequence-gap"
  | "backward-video-timing"
  | "incomplete-presented-frame-timing"
  | "missing-pose-attachments"
  | "invalid-pose-attachment"
  | "pose-sequence-gap"
  | "backward-pose-attachment"
  | "noncausal-pose-attachment"
  | "stale-pose-attachment"
  | "pose-video-frame-not-in-trace"
  | "missing-intervals"
  | "invalid-interval-count"
  | "invalid-interval"
  | "invalid-interval-order"
  | "interval-boundary-not-in-trace"
  | "interval-overlap"
  | "missing-boundary-pose"
  | "missing-human-confirmation"
  | "missing-reviewer-code"
  | "continuous-video-not-confirmed"
  | "overlays-not-confirmed"
  | "reference-interval-not-confirmed"
  | "attempt-interval-not-confirmed"
  | "invalid-declared-timestamp"
  | "invalid-evidence-scope";

export interface ReplayIntegrityIssue {
  readonly code: ReplayIntegrityIssueCode;
  readonly path: string;
  readonly message: string;
}

export interface ContinuousVideoAssetValidation {
  readonly valid: boolean;
  readonly issues: readonly ReplayIntegrityIssue[];
  readonly observed: ObservedContinuousVideoAsset | null;
}

export interface ReplayTraceValidation {
  readonly valid: boolean;
  readonly issues: readonly ReplayIntegrityIssue[];
}

export interface ReplayBoundaryCoverage<
  TPose extends ReplayPoseTimingSample = ReplayPoseTimingSample,
> {
  readonly intervalIndex: number;
  readonly intervalKind: ReplayCaptureIntervalKind;
  readonly boundary: "begin" | "end";
  readonly replayMediaTimeMs: number;
  readonly videoFrame: ReplayVideoTimingSample;
  readonly selectedPose: ReplayPoseAttachment<TPose>;
  /** Reported, not thresholded: no uncalibrated maximum lag is invented here. */
  readonly poseMediaLagMs: number;
}

export interface ReplayBoundaryCoverageAudit<
  TPose extends ReplayPoseTimingSample = ReplayPoseTimingSample,
> {
  readonly valid: boolean;
  readonly issues: readonly ReplayIntegrityIssue[];
  readonly expectedBoundaryCount: number;
  readonly coveredBoundaryCount: number;
  readonly coverage: readonly ReplayBoundaryCoverage<TPose>[];
}

export interface HumanReplayConfirmationValidation {
  readonly valid: boolean;
  readonly issues: readonly ReplayIntegrityIssue[];
}

export interface LocalReplayIntegrityInput<
  TPose extends ReplayPoseTimingSample = ReplayPoseTimingSample,
> {
  readonly declaredVideo: DeclaredContinuousVideoAsset | null | undefined;
  readonly observedVideo: LocalContinuousVideoBlob | null | undefined;
  readonly decodedVideo:
    | DecodedContinuousVideoEvidence
    | null
    | undefined;
  readonly videoFrames: readonly ReplayVideoTimingSample[] | null | undefined;
  readonly poseAttachments:
    | readonly ReplayPoseAttachment<TPose>[]
    | null
    | undefined;
  readonly intervals: readonly ReplayCaptureInterval[] | null | undefined;
  readonly confirmation: HumanReplayConfirmation | null | undefined;
}

export interface LocalReplayIntegrityAudit<
  TPose extends ReplayPoseTimingSample = ReplayPoseTimingSample,
> {
  readonly valid: boolean;
  readonly status: "REPLAY_INTEGRITY_CONFIRMED" | "REPLAY_INTEGRITY_REJECTED";
  readonly evidenceScope: typeof REPLAY_INTEGRITY_EVIDENCE_SCOPE;
  readonly claimBoundary: typeof REPLAY_INTEGRITY_CLAIM_BOUNDARY;
  readonly issues: readonly ReplayIntegrityIssue[];
  readonly videoAsset: ContinuousVideoAssetValidation;
  readonly decodedVideo: DecodedContinuousVideoEvidenceValidation;
  readonly boundaryCoverage: ReplayBoundaryCoverageAudit<TPose>;
  readonly humanConfirmation: HumanReplayConfirmationValidation;
}

export interface DecodedContinuousVideoEvidenceValidation {
  readonly valid: boolean;
  readonly issues: readonly ReplayIntegrityIssue[];
  readonly observed: DecodedContinuousVideoEvidence | null;
}

const INTERVAL_ORDER = ["reference", "attempt", "revision"] as const;
const SHA256_PATTERN = /^[a-f0-9]{64}$/i;

function issue(
  code: ReplayIntegrityIssueCode,
  path: string,
  message: string,
): ReplayIntegrityIssue {
  return { code, path, message };
}

function uniqueIssues(
  issues: readonly ReplayIntegrityIssue[],
): ReplayIntegrityIssue[] {
  const seen = new Set<string>();
  return issues.filter((entry) => {
    const key = `${entry.code}:${entry.path}:${entry.message}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function validNonnegativeTime(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function validSequence(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function validPositiveByteLength(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function validFileName(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value === value.trim() &&
    value !== "." &&
    value !== ".." &&
    !/[\\/\u0000-\u001f\u007f]/.test(value)
  );
}

function normalizedVideoMimeType(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  if (!/^video\/[a-z0-9][a-z0-9!#$&^_.+\-]*(?:\s*;.+)?$/i.test(normalized)) {
    return null;
  }
  return normalized;
}

function readableBlob(value: unknown): value is Blob {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<Blob>;
  return (
    typeof candidate.size === "number" &&
    typeof candidate.type === "string" &&
    typeof candidate.arrayBuffer === "function"
  );
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/** Compute the identity digest locally with browser WebCrypto. */
export async function sha256Blob(blob: Blob): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Browser WebCrypto SHA-256 is unavailable.");
  }
  const buffer = await blob.arrayBuffer();
  const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  return bytesToHex(new Uint8Array(digest));
}

/**
 * Re-read and hash the actual blob. The declaration's digest is never trusted
 * as the observation, and every declared Blob-visible metadata field is bound.
 */
export async function validateRecordedContinuousVideo(
  declaration: DeclaredContinuousVideoAsset | null | undefined,
  observedVideo: LocalContinuousVideoBlob | null | undefined,
): Promise<ContinuousVideoAssetValidation> {
  const issues: ReplayIntegrityIssue[] = [];

  if (!declaration) {
    issues.push(
      issue(
        "missing-video-declaration",
        "declaredVideo",
        "A declared continuous-video asset is required.",
      ),
    );
  }
  if (!observedVideo) {
    issues.push(
      issue(
        "missing-video-blob",
        "observedVideo",
        "The local continuous-video blob is required for replay integrity.",
      ),
    );
  }
  if (!declaration || !observedVideo) {
    return { valid: false, issues, observed: null };
  }

  const declaredFileNameValid = validFileName(declaration.fileName);
  const observedFileNameValid = validFileName(observedVideo.fileName);
  if (!declaredFileNameValid) {
    issues.push(
      issue(
        "invalid-file-name",
        "declaredVideo.fileName",
        "The declared video filename must be a non-empty local filename.",
      ),
    );
  }
  if (!observedFileNameValid) {
    issues.push(
      issue(
        "invalid-file-name",
        "observedVideo.fileName",
        "The observed video filename must be a non-empty local filename.",
      ),
    );
  }
  if (
    declaredFileNameValid &&
    observedFileNameValid &&
    declaration.fileName !== observedVideo.fileName
  ) {
    issues.push(
      issue(
        "file-name-mismatch",
        "observedVideo.fileName",
        "The observed filename does not match the declared continuous-video asset.",
      ),
    );
  }

  const declaredMimeType = normalizedVideoMimeType(declaration.mimeType);
  if (!declaredMimeType) {
    issues.push(
      issue(
        "invalid-mime-type",
        "declaredVideo.mimeType",
        "The declared MIME type must be a valid video media type.",
      ),
    );
  }
  if (!validPositiveByteLength(declaration.bytes)) {
    issues.push(
      issue(
        "invalid-byte-length",
        "declaredVideo.bytes",
        "The declared byte length must be a positive safe integer.",
      ),
    );
  }
  if (
    typeof declaration.sha256 !== "string" ||
    !SHA256_PATTERN.test(declaration.sha256)
  ) {
    issues.push(
      issue(
        "invalid-sha256",
        "declaredVideo.sha256",
        "The declared digest must contain exactly 64 hexadecimal SHA-256 characters.",
      ),
    );
  }

  if (!readableBlob(observedVideo.blob)) {
    issues.push(
      issue(
        "missing-video-blob",
        "observedVideo.blob",
        "The observed asset is not a readable browser Blob.",
      ),
    );
    return { valid: false, issues, observed: null };
  }

  const blob = observedVideo.blob;
  const observedMimeType = normalizedVideoMimeType(blob.type);
  if (!observedMimeType) {
    issues.push(
      issue(
        "invalid-mime-type",
        "observedVideo.blob.type",
        "The observed Blob must declare a valid video MIME type.",
      ),
    );
  } else if (declaredMimeType && observedMimeType !== declaredMimeType) {
    issues.push(
      issue(
        "mime-type-mismatch",
        "observedVideo.blob.type",
        "The observed Blob MIME type does not match the declaration.",
      ),
    );
  }

  if (!validPositiveByteLength(blob.size)) {
    issues.push(
      issue(
        "invalid-byte-length",
        "observedVideo.blob.size",
        "The observed video Blob must contain at least one byte.",
      ),
    );
  } else if (
    validPositiveByteLength(declaration.bytes) &&
    blob.size !== declaration.bytes
  ) {
    issues.push(
      issue(
        "byte-length-mismatch",
        "observedVideo.blob.size",
        "The observed Blob byte length does not match the declaration.",
      ),
    );
  }

  let buffer: ArrayBuffer;
  try {
    buffer = await blob.arrayBuffer();
  } catch {
    issues.push(
      issue(
        "blob-read-failed",
        "observedVideo.blob",
        "The continuous-video Blob could not be read locally.",
      ),
    );
    return { valid: false, issues, observed: null };
  }

  if (buffer.byteLength !== blob.size) {
    issues.push(
      issue(
        "blob-read-size-mismatch",
        "observedVideo.blob",
        "The bytes read from the Blob do not match its reported size.",
      ),
    );
  }

  let computedSha256: string;
  try {
    if (!globalThis.crypto?.subtle) throw new Error("WebCrypto unavailable");
    const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
    computedSha256 = bytesToHex(new Uint8Array(digest));
  } catch {
    issues.push(
      issue(
        "hash-unavailable",
        "observedVideo.blob",
        "The observed video could not be hashed with browser WebCrypto.",
      ),
    );
    return { valid: false, issues, observed: null };
  }

  if (
    SHA256_PATTERN.test(declaration.sha256) &&
    computedSha256 !== declaration.sha256.toLowerCase()
  ) {
    issues.push(
      issue(
        "sha256-mismatch",
        "observedVideo.blob",
        "The observed video SHA-256 does not match the declaration.",
      ),
    );
  }

  const observed: ObservedContinuousVideoAsset = {
    fileName: observedVideo.fileName,
    mimeType: blob.type,
    bytes: buffer.byteLength,
    sha256: computedSha256,
  };
  return { valid: issues.length === 0, issues, observed };
}

/** Validate browser-observed media decode and coverage of the retained clock. */
export function validateDecodedContinuousVideoEvidence(
  decodedVideo: DecodedContinuousVideoEvidence | null | undefined,
  videoFrames: readonly ReplayVideoTimingSample[] | null | undefined,
): DecodedContinuousVideoEvidenceValidation {
  const issues: ReplayIntegrityIssue[] = [];
  if (!decodedVideo) {
    issues.push(
      issue(
        "missing-decoded-video-evidence",
        "decodedVideo",
        "Browser-observed decoded-video metadata is required; byte identity alone does not prove replayability.",
      ),
    );
    return { valid: false, issues, observed: null };
  }
  const finitePositive = (value: unknown): value is number =>
    typeof value === "number" && Number.isFinite(value) && value > 0;
  const finiteNonnegative = (value: unknown): value is number =>
    typeof value === "number" && Number.isFinite(value) && value >= 0;
  if (
    decodedVideo.status !== "DECODED_LOCAL_VIDEO" ||
    !finitePositive(decodedVideo.durationMs) ||
    !Number.isSafeInteger(decodedVideo.videoWidth) ||
    decodedVideo.videoWidth <= 0 ||
    !Number.isSafeInteger(decodedVideo.videoHeight) ||
    decodedVideo.videoHeight <= 0 ||
    !finiteNonnegative(decodedVideo.seekableStartMs) ||
    !finitePositive(decodedVideo.seekableEndMs) ||
    decodedVideo.seekableEndMs <= decodedVideo.seekableStartMs ||
    !finiteNonnegative(decodedVideo.observedDecodedFrameMediaTimeMs) ||
    decodedVideo.sourceTimelineMapping !== RECORDED_BLOB_TIMELINE_MAPPING ||
    !validIsoInstant(decodedVideo.inspectedAt)
  ) {
    issues.push(
      issue(
        "invalid-decoded-video-evidence",
        "decodedVideo",
        "Decoded-video evidence must contain finite duration/seek range, nonzero dimensions, a decoded frame time, the uncalibrated source-to-blob mapping declaration, and an ISO timestamp.",
      ),
    );
  }
  const timingValidation = validateReplayVideoTimingTrace(videoFrames);
  issues.push(...timingValidation.issues);
  // Source rVFC time zero and encoded Blob time zero are deliberately declared
  // uncalibrated. Comparing their absolute endpoints here would assume the very
  // mapping this artifact says is UNKNOWN. Exact coverage becomes an admission
  // gate only after a measured source-to-blob calibration is retained.
  return {
    valid: issues.length === 0,
    issues: uniqueIssues(issues),
    observed: decodedVideo,
  };
}

function videoTimingIssues(
  sample: ReplayVideoTimingSample | null | undefined,
  path: string,
): ReplayIntegrityIssue[] {
  if (!sample || typeof sample !== "object") {
    return [
      issue(
        "invalid-video-timing",
        path,
        "A complete video-frame timing sample is required.",
      ),
    ];
  }
  const issues: ReplayIntegrityIssue[] = [];
  if (!validSequence(sample.sequence)) {
    issues.push(
      issue(
        "invalid-video-timing",
        `${path}.sequence`,
        "Video-frame sequence must be a non-negative safe integer.",
      ),
    );
  }
  if (!validNonnegativeTime(sample.callbackMonotonicTimeMs)) {
    issues.push(
      issue(
        "invalid-video-timing",
        `${path}.callbackMonotonicTimeMs`,
        "Video callback time must be finite and non-negative.",
      ),
    );
  }
  if (!validNonnegativeTime(sample.mediaTimeMs)) {
    issues.push(
      issue(
        "invalid-video-timing",
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
        "invalid-video-timing",
        `${path}.presentedFrames`,
        "Presented-frame count must be a non-negative safe integer when retained.",
      ),
    );
  }
  return issues;
}

function sameVideoTimingSample(
  left: ReplayVideoTimingSample,
  right: ReplayVideoTimingSample,
): boolean {
  return (
    left.sequence === right.sequence &&
    left.callbackMonotonicTimeMs === right.callbackMonotonicTimeMs &&
    left.mediaTimeMs === right.mediaTimeMs &&
    left.presentedFrames === right.presentedFrames
  );
}

/** Validate the retained replay video clock without sorting or repairing it. */
export function validateReplayVideoTimingTrace(
  videoFrames: readonly ReplayVideoTimingSample[] | null | undefined,
): ReplayTraceValidation {
  const issues: ReplayIntegrityIssue[] = [];
  if (!videoFrames || videoFrames.length === 0) {
    issues.push(
      issue(
        "missing-video-timing",
        "videoFrames",
        "At least one retained video-frame timing sample is required.",
      ),
    );
    return { valid: false, issues };
  }

  videoFrames.forEach((sample, index) => {
    issues.push(...videoTimingIssues(sample, `videoFrames[${index}]`));
  });
  for (let index = 1; index < videoFrames.length; index += 1) {
    const previous = videoFrames[index - 1];
    const current = videoFrames[index];
    if (
      validSequence(previous?.sequence) &&
      validSequence(current?.sequence) &&
      current.sequence !== previous.sequence + 1
    ) {
      issues.push(
        issue(
          "video-sequence-gap",
          `videoFrames[${index}].sequence`,
          "The retained video timing trace must be strictly ordered and contiguous.",
        ),
      );
    }
    if (
      validNonnegativeTime(previous?.callbackMonotonicTimeMs) &&
      validNonnegativeTime(current?.callbackMonotonicTimeMs) &&
      current.callbackMonotonicTimeMs < previous.callbackMonotonicTimeMs
    ) {
      issues.push(
        issue(
          "backward-video-timing",
          `videoFrames[${index}].callbackMonotonicTimeMs`,
          "Video callback time moved backward.",
        ),
      );
    }
    if (
      validNonnegativeTime(previous?.mediaTimeMs) &&
      validNonnegativeTime(current?.mediaTimeMs) &&
      current.mediaTimeMs < previous.mediaTimeMs
    ) {
      issues.push(
        issue(
          "backward-video-timing",
          `videoFrames[${index}].mediaTimeMs`,
          "Video media time moved backward.",
        ),
      );
    }
  }

  const presentedCount = videoFrames.filter(
    (sample) => sample.presentedFrames !== undefined,
  ).length;
  if (presentedCount > 0 && presentedCount !== videoFrames.length) {
    issues.push(
      issue(
        "incomplete-presented-frame-timing",
        "videoFrames",
        "Presented-frame counts must be retained for every timing sample or for none.",
      ),
    );
  }
  return { valid: issues.length === 0, issues };
}

/** Validate that every retained pose attachment is ordered and causal. */
export function validateReplayPoseAttachments<
  TPose extends ReplayPoseTimingSample,
>(
  attachments: readonly ReplayPoseAttachment<TPose>[] | null | undefined,
  videoFrames?: readonly ReplayVideoTimingSample[] | null,
): ReplayTraceValidation {
  const issues: ReplayIntegrityIssue[] = [];
  if (!attachments || attachments.length === 0) {
    issues.push(
      issue(
        "missing-pose-attachments",
        "poseAttachments",
        "At least one causally attached pose is required for overlay replay.",
      ),
    );
    return { valid: false, issues };
  }

  attachments.forEach((attachment, index) => {
    const path = `poseAttachments[${index}]`;
    if (!attachment || typeof attachment !== "object") {
      issues.push(
        issue(
          "invalid-pose-attachment",
          path,
          "A complete causal pose attachment is required.",
        ),
      );
      return;
    }
    if (attachment.strategy !== "nearest-latest-causal-video-frame") {
      issues.push(
        issue(
          "invalid-pose-attachment",
          `${path}.strategy`,
          "The replay pose must retain the declared latest-causal attachment strategy.",
        ),
      );
    }
    if (!attachment.pose || typeof attachment.pose !== "object") {
      issues.push(
        issue(
          "invalid-pose-attachment",
          `${path}.pose`,
          "A pose timing record is required.",
        ),
      );
    } else {
      if (!validSequence(attachment.pose.sequence)) {
        issues.push(
          issue(
            "invalid-pose-attachment",
            `${path}.pose.sequence`,
            "Pose sequence must be a non-negative safe integer.",
          ),
        );
      }
      if (!validNonnegativeTime(attachment.pose.callbackMonotonicTimeMs)) {
        issues.push(
          issue(
            "invalid-pose-attachment",
            `${path}.pose.callbackMonotonicTimeMs`,
            "Pose callback time must be finite and non-negative.",
          ),
        );
      }
    }
    issues.push(...videoTimingIssues(attachment.videoFrame, `${path}.videoFrame`));
    if (
      validNonnegativeTime(attachment.pose?.callbackMonotonicTimeMs) &&
      validNonnegativeTime(attachment.videoFrame?.callbackMonotonicTimeMs) &&
      attachment.videoFrame.callbackMonotonicTimeMs >
        attachment.pose.callbackMonotonicTimeMs
    ) {
      issues.push(
        issue(
          "noncausal-pose-attachment",
          path,
          "A pose cannot be attached to a video callback from its future.",
        ),
      );
    }
  });

  for (let index = 1; index < attachments.length; index += 1) {
    const previous = attachments[index - 1];
    const current = attachments[index];
    if (!previous?.pose || !current?.pose || !previous.videoFrame || !current.videoFrame) {
      continue;
    }
    if (
      validSequence(previous.pose.sequence) &&
      validSequence(current.pose.sequence) &&
      current.pose.sequence !== previous.pose.sequence + 1
    ) {
      issues.push(
        issue(
          "pose-sequence-gap",
          `poseAttachments[${index}].pose.sequence`,
          "The retained pose attachment trace must be strictly ordered and contiguous.",
        ),
      );
    }
    if (
      validNonnegativeTime(previous.pose.callbackMonotonicTimeMs) &&
      validNonnegativeTime(current.pose.callbackMonotonicTimeMs) &&
      current.pose.callbackMonotonicTimeMs <
        previous.pose.callbackMonotonicTimeMs
    ) {
      issues.push(
        issue(
          "backward-pose-attachment",
          `poseAttachments[${index}].pose.callbackMonotonicTimeMs`,
          "Pose callback time moved backward.",
        ),
      );
    }
    if (
      validSequence(previous.videoFrame.sequence) &&
      validSequence(current.videoFrame.sequence) &&
      current.videoFrame.sequence < previous.videoFrame.sequence
    ) {
      issues.push(
        issue(
          "backward-pose-attachment",
          `poseAttachments[${index}].videoFrame.sequence`,
          "Attached video-frame sequence moved backward.",
        ),
      );
    }
    if (
      validNonnegativeTime(previous.videoFrame.mediaTimeMs) &&
      validNonnegativeTime(current.videoFrame.mediaTimeMs) &&
      current.videoFrame.mediaTimeMs < previous.videoFrame.mediaTimeMs
    ) {
      issues.push(
        issue(
          "backward-pose-attachment",
          `poseAttachments[${index}].videoFrame.mediaTimeMs`,
          "Attached video media time moved backward.",
        ),
      );
    }
    if (
      current.videoFrame.sequence === previous.videoFrame.sequence &&
      !sameVideoTimingSample(current.videoFrame, previous.videoFrame)
    ) {
      issues.push(
        issue(
          "invalid-pose-attachment",
          `poseAttachments[${index}].videoFrame`,
          "A repeated attached video-frame sequence must retain identical timing metadata.",
        ),
      );
    }
  }

  if (videoFrames) {
    const videoValidation = validateReplayVideoTimingTrace(videoFrames);
    issues.push(...videoValidation.issues);
    if (videoValidation.valid) {
      attachments.forEach((attachment, index) => {
        const expected = latestVideoFrameAtOrBeforeMonotonicTime(
          attachment.pose.callbackMonotonicTimeMs,
          videoFrames,
        );
        if (!expected || !sameVideoTimingSample(expected, attachment.videoFrame)) {
          issues.push(
            issue(
              "stale-pose-attachment",
              `poseAttachments[${index}].videoFrame`,
              "Each pose must bind the exact latest retained video callback at or before the pose callback; an older causal frame is not sufficient.",
            ),
          );
        }
      });
    }
  }
  return { valid: issues.length === 0, issues };
}

function latestVideoFrameAtOrBeforeMonotonicTime(
  callbackMonotonicTimeMs: number,
  videoFrames: readonly ReplayVideoTimingSample[],
): ReplayVideoTimingSample | null {
  let lower = 0;
  let upper = videoFrames.length - 1;
  let match: ReplayVideoTimingSample | null = null;
  while (lower <= upper) {
    const middle = lower + Math.floor((upper - lower) / 2);
    const candidate = videoFrames[middle];
    if (candidate.callbackMonotonicTimeMs <= callbackMonotonicTimeMs) {
      match = candidate;
      lower = middle + 1;
    } else {
      upper = middle - 1;
    }
  }
  return match;
}

/**
 * Select the final attached pose whose source video media time is no later than
 * the observed replay time. Invalid traces return null and are never repaired,
 * sorted, interpolated, or allowed to look at a future video frame.
 */
export function selectLatestCausalPoseForReplayTime<
  TPose extends ReplayPoseTimingSample,
>(
  observedReplayMediaTimeMs: number,
  attachments: readonly ReplayPoseAttachment<TPose>[] | null | undefined,
  videoFrames: readonly ReplayVideoTimingSample[] | null | undefined,
): ReplayPoseAttachment<TPose> | null {
  if (!validNonnegativeTime(observedReplayMediaTimeMs)) return null;
  if (
    !videoFrames ||
    !validateReplayVideoTimingTrace(videoFrames).valid ||
    !validateReplayPoseAttachments(attachments, videoFrames).valid ||
    !attachments
  ) {
    return null;
  }

  let match: ReplayPoseAttachment<TPose> | null = null;
  for (const candidate of attachments) {
    if (candidate.videoFrame.mediaTimeMs <= observedReplayMediaTimeMs) {
      match = candidate;
    } else break;
  }
  return match;
}

function videoTimingIndex(
  boundary: ReplayVideoTimingSample,
  videoFrames: readonly ReplayVideoTimingSample[],
): number {
  const index = videoFrames.findIndex(
    (sample) => sample.sequence === boundary.sequence,
  );
  if (index < 0 || !sameVideoTimingSample(boundary, videoFrames[index])) {
    return -1;
  }
  return index;
}

/**
 * Select source evidence available by the exact boundary callback. A pose that
 * arrived after the boundary is not retroactively admitted merely because it
 * attached to the boundary's latest video frame.
 */
function selectLatestPoseAtSourceBoundary<
  TPose extends ReplayPoseTimingSample,
>(
  boundary: ReplayVideoTimingSample,
  attachments: readonly ReplayPoseAttachment<TPose>[],
): ReplayPoseAttachment<TPose> | null {
  let match: ReplayPoseAttachment<TPose> | null = null;
  for (const candidate of attachments) {
    if (candidate.pose.callbackMonotonicTimeMs > boundary.callbackMonotonicTimeMs) {
      break;
    }
    if (candidate.videoFrame.mediaTimeMs <= boundary.mediaTimeMs) {
      match = candidate;
    }
  }
  return match;
}

/** Audit exact video and causal pose coverage at every retained interval edge. */
export function auditReplayIntervalBoundaryCoverage<
  TPose extends ReplayPoseTimingSample,
>(
  intervals: readonly ReplayCaptureInterval[] | null | undefined,
  videoFrames: readonly ReplayVideoTimingSample[] | null | undefined,
  attachments: readonly ReplayPoseAttachment<TPose>[] | null | undefined,
): ReplayBoundaryCoverageAudit<TPose> {
  const issues: ReplayIntegrityIssue[] = [];
  const coverage: ReplayBoundaryCoverage<TPose>[] = [];
  const expectedBoundaryCount = intervals ? intervals.length * 2 : 0;
  const videoValidation = validateReplayVideoTimingTrace(videoFrames);
  const poseValidation = validateReplayPoseAttachments(attachments, videoFrames);
  issues.push(...videoValidation.issues, ...poseValidation.issues);

  if (!intervals || intervals.length === 0) {
    issues.push(
      issue(
        "missing-intervals",
        "intervals",
        "Reference and attempt intervals are required for replay audit.",
      ),
    );
    return {
      valid: false,
      issues: uniqueIssues(issues),
      expectedBoundaryCount,
      coveredBoundaryCount: 0,
      coverage,
    };
  }
  if (intervals.length !== 2 && intervals.length !== 3) {
    issues.push(
      issue(
        "invalid-interval-count",
        "intervals",
        "Replay requires reference and attempt intervals, plus at most one revision.",
      ),
    );
  }

  if (videoValidation.valid && videoFrames && attachments) {
    attachments.forEach((attachment, index) => {
      if (videoTimingIndex(attachment.videoFrame, videoFrames) < 0) {
        issues.push(
          issue(
            "pose-video-frame-not-in-trace",
            `poseAttachments[${index}].videoFrame`,
            "An attached pose references a video frame absent from this replay trace.",
          ),
        );
      }
    });
  }

  const intervalIndexes: Array<{ begin: number; end: number } | null> = [];
  intervals.forEach((interval, intervalIndex) => {
    const path = `intervals[${intervalIndex}]`;
    const expectedKind = INTERVAL_ORDER[intervalIndex];
    if (!interval || typeof interval !== "object") {
      issues.push(
        issue("invalid-interval", path, "A complete replay interval is required."),
      );
      intervalIndexes.push(null);
      return;
    }
    if (interval.kind !== expectedKind) {
      issues.push(
        issue(
          "invalid-interval-order",
          `${path}.kind`,
          `Expected ${expectedKind ?? "no additional"} at this interval position.`,
        ),
      );
    }
    const beginIssues = videoTimingIssues(interval.begin, `${path}.begin`);
    const endIssues = videoTimingIssues(interval.end, `${path}.end`);
    issues.push(...beginIssues, ...endIssues);

    let beginIndex = -1;
    let endIndex = -1;
    if (videoValidation.valid && videoFrames && beginIssues.length === 0) {
      beginIndex = videoTimingIndex(interval.begin, videoFrames);
    }
    if (videoValidation.valid && videoFrames && endIssues.length === 0) {
      endIndex = videoTimingIndex(interval.end, videoFrames);
    }
    if (beginIndex < 0) {
      issues.push(
        issue(
          "interval-boundary-not-in-trace",
          `${path}.begin`,
          "The interval begin is not an exact sample from this replay video trace.",
        ),
      );
    }
    if (endIndex < 0) {
      issues.push(
        issue(
          "interval-boundary-not-in-trace",
          `${path}.end`,
          "The interval end is not an exact sample from this replay video trace.",
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
          "invalid-interval",
          path,
          "The interval end must follow its begin on every retained clock.",
        ),
      );
    }
    intervalIndexes.push(
      beginIndex >= 0 && endIndex >= 0 ? { begin: beginIndex, end: endIndex } : null,
    );

    if (
      videoValidation.valid &&
      poseValidation.valid &&
      videoFrames &&
      attachments &&
      beginIndex >= 0 &&
      endIndex >= 0
    ) {
      (["begin", "end"] as const).forEach((boundaryName) => {
        const boundary = interval[boundaryName];
        const selectedPose = selectLatestPoseAtSourceBoundary(
          boundary,
          attachments,
        );
        if (!selectedPose) {
          issues.push(
            issue(
              "missing-boundary-pose",
              `${path}.${boundaryName}`,
              "No causally attached pose is available at this replay boundary.",
            ),
          );
          return;
        }
        if (videoTimingIndex(selectedPose.videoFrame, videoFrames) < 0) {
          issues.push(
            issue(
              "pose-video-frame-not-in-trace",
              `${path}.${boundaryName}`,
              "The selected boundary pose is not attached to this replay trace.",
            ),
          );
          return;
        }
        coverage.push({
          intervalIndex,
          intervalKind: interval.kind,
          boundary: boundaryName,
          replayMediaTimeMs: boundary.mediaTimeMs,
          videoFrame: boundary,
          selectedPose,
          poseMediaLagMs:
            boundary.mediaTimeMs - selectedPose.videoFrame.mediaTimeMs,
        });
      });
    }
  });

  for (let index = 1; index < intervalIndexes.length; index += 1) {
    const previous = intervalIndexes[index - 1];
    const current = intervalIndexes[index];
    if (previous && current && previous.end > current.begin) {
      issues.push(
        issue(
          "interval-overlap",
          `intervals[${index}]`,
          "Replay intervals must be chronological and non-overlapping.",
        ),
      );
    }
  }

  const deduplicated = uniqueIssues(issues);
  return {
    valid:
      deduplicated.length === 0 && coverage.length === expectedBoundaryCount,
    issues: deduplicated,
    expectedBoundaryCount,
    coveredBoundaryCount: coverage.length,
    coverage,
  };
}

function validIsoInstant(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const match = value.match(
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$/,
  );
  if (!match || !Number.isFinite(Date.parse(value))) return false;
  const [, yearText, monthText, dayText, hourText, minuteText, secondText] =
    match;
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const hour = Number(hourText);
  const minute = Number(minuteText);
  const second = Number(secondText);
  if (
    month < 1 ||
    month > 12 ||
    hour > 23 ||
    minute > 59 ||
    second > 59
  ) {
    return false;
  }
  const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate();
  return day >= 1 && day <= daysInMonth;
}

/** Require an explicit human declaration after continuous local replay. */
export function validateHumanReplayConfirmation(
  confirmation: HumanReplayConfirmation | null | undefined,
): HumanReplayConfirmationValidation {
  const issues: ReplayIntegrityIssue[] = [];
  if (!confirmation) {
    issues.push(
      issue(
        "missing-human-confirmation",
        "confirmation",
        "A human replay confirmation record is required.",
      ),
    );
    return { valid: false, issues };
  }
  if (
    typeof confirmation.reviewerCode !== "string" ||
    !confirmation.reviewerCode.trim()
  ) {
    issues.push(
      issue(
        "missing-reviewer-code",
        "confirmation.reviewerCode",
        "A non-empty reviewer code is required.",
      ),
    );
  }
  if (confirmation.watchedContinuousVideo !== true) {
    issues.push(
      issue(
        "continuous-video-not-confirmed",
        "confirmation.watchedContinuousVideo",
        "The reviewer must declare that the complete continuous video was watched.",
      ),
    );
  }
  if (confirmation.overlaysVisible !== true) {
    issues.push(
      issue(
        "overlays-not-confirmed",
        "confirmation.overlaysVisible",
        "The reviewer must declare that pose overlays were visible during replay.",
      ),
    );
  }
  if (confirmation.referenceIntervalConfirmed !== true) {
    issues.push(
      issue(
        "reference-interval-not-confirmed",
        "confirmation.referenceIntervalConfirmed",
        "The reviewer must confirm the visible reference interval.",
      ),
    );
  }
  if (confirmation.attemptIntervalConfirmed !== true) {
    issues.push(
      issue(
        "attempt-interval-not-confirmed",
        "confirmation.attemptIntervalConfirmed",
        "The reviewer must confirm the visible attempt interval.",
      ),
    );
  }
  if (!validIsoInstant(confirmation.declaredAt)) {
    issues.push(
      issue(
        "invalid-declared-timestamp",
        "confirmation.declaredAt",
        "The replay declaration requires a finite ISO 8601 timestamp with a timezone.",
      ),
    );
  }
  if (confirmation.evidenceScope !== REPLAY_INTEGRITY_EVIDENCE_SCOPE) {
    issues.push(
      issue(
        "invalid-evidence-scope",
        "confirmation.evidenceScope",
        "Replay confirmation must retain the integrity-only evidence boundary.",
      ),
    );
  }
  return { valid: issues.length === 0, issues };
}

/** Run every local replay-integrity gate and fail closed if any gate is absent. */
export async function auditLocalReplayIntegrity<
  TPose extends ReplayPoseTimingSample,
>(
  input: LocalReplayIntegrityInput<TPose>,
): Promise<LocalReplayIntegrityAudit<TPose>> {
  const [videoAsset, decodedVideo, boundaryCoverage, humanConfirmation] = await Promise.all([
    validateRecordedContinuousVideo(input.declaredVideo, input.observedVideo),
    Promise.resolve(
      validateDecodedContinuousVideoEvidence(input.decodedVideo, input.videoFrames),
    ),
    Promise.resolve(
      auditReplayIntervalBoundaryCoverage(
        input.intervals,
        input.videoFrames,
        input.poseAttachments,
      ),
    ),
    Promise.resolve(validateHumanReplayConfirmation(input.confirmation)),
  ]);
  const issues = uniqueIssues([
    ...videoAsset.issues,
    ...decodedVideo.issues,
    ...boundaryCoverage.issues,
    ...humanConfirmation.issues,
  ]);
  const valid =
    videoAsset.valid &&
    decodedVideo.valid &&
    boundaryCoverage.valid &&
    humanConfirmation.valid;
  return {
    valid,
    status: valid
      ? "REPLAY_INTEGRITY_CONFIRMED"
      : "REPLAY_INTEGRITY_REJECTED",
    evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
    claimBoundary: REPLAY_INTEGRITY_CLAIM_BOUNDARY,
    issues,
    videoAsset,
    decodedVideo,
    boundaryCoverage,
    humanConfirmation,
  };
}
