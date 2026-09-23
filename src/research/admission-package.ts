import { unzipSync } from "fflate";
import { canonicalJson, sha256Bytes } from "./capture-bundle";

export const ADMISSION_PACKAGE_PREFLIGHT_SCHEMA =
  "relay.body-native-admission-package-preflight/1" as const;

export interface RawChecksumSidecar {
  readonly fileName: string;
  readonly bytes: Uint8Array;
}

export interface RawChecksummedArtifact {
  readonly fileName: string;
  readonly bytes: Uint8Array;
  readonly checksumSidecar: RawChecksumSidecar;
}

export interface AdmissionPackageV1Input {
  readonly archive: RawChecksummedArtifact;
  readonly reviews: readonly RawChecksummedArtifact[];
  readonly calibration: RawChecksummedArtifact;
}

export interface ByteArtifactLineage {
  readonly fileName: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly checksumSidecar: {
    readonly fileName: string;
    readonly bytes: number;
    readonly sha256: string;
    readonly declaredArtifactFileName: string;
    readonly declaredArtifactSha256: string;
  };
}

export interface ArchiveMemberLineage {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface ArchiveArtifactLineage extends ByteArtifactLineage {
  readonly schemaVersion: "relay.body-native-episode/2";
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly episodeJsonSha256: string;
  readonly continuousVideoSha256: string;
  readonly continuousVideoPath: string;
  readonly videoTimingSha256: string;
  readonly members: readonly ArchiveMemberLineage[];
}

export interface ReviewArtifactLineage extends ByteArtifactLineage {
  readonly schemaVersion: "relay.body-native-independent-review-sidecar/1";
  readonly sourceEpisodeJsonSha256: string;
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly annotationId: string;
  readonly raterCode: string;
  readonly encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED";
}

export interface CalibrationArtifactLineage extends ByteArtifactLineage {
  readonly schemaVersion: "relay.media-clock-calibration/1";
  readonly fitStatus: "FIT_REPORTED_NOT_ADMISSION_CALIBRATED" | "REJECTED";
  readonly episodeBindingStatus: "ABSENT_FROM_CALIBRATION_SCHEMA_V1";
}

export type AdmissionPackagePreflightGateId =
  | "byte-and-checksum-integrity"
  | "archive-member-lineage"
  | "append-only-review-lineage"
  | "runtime-and-app-build-identity"
  | "episode-video-timing-calibration-binding"
  | "review-timeline-calibration"
  | "v1-admission-contract";

export interface AdmissionPackagePreflightGate {
  readonly id: AdmissionPackagePreflightGateId;
  readonly passed: boolean;
  readonly issues: readonly string[];
  readonly facts: Readonly<Record<string, string | number | boolean | null>>;
}

export interface AdmissionPackagePreflightReport {
  readonly schemaVersion: typeof ADMISSION_PACKAGE_PREFLIGHT_SCHEMA;
  readonly status: "QUARANTINED";
  readonly admitted: false;
  readonly evidenceClass: "QUARANTINED_UNADMITTED";
  readonly permittedUse: "integrity-and-lineage-audit-only";
  readonly lineage: {
    readonly archive: ArchiveArtifactLineage;
    readonly reviews: readonly ReviewArtifactLineage[];
    readonly calibration: CalibrationArtifactLineage;
  };
  readonly packageRoot: {
    readonly algorithm: "SHA-256";
    readonly canonicalManifest: string;
    readonly sha256: string;
  };
  readonly gates: readonly AdmissionPackagePreflightGate[];
  readonly blockers: readonly [
    "The relay.body-native-episode/2 archive does not retain a trusted pose-runtime version and immutable application build SHA-256.",
    "relay.media-clock-calibration/1 is not cryptographically bound to this episode JSON, continuous video, and video-timing artifact.",
    "relay.body-native-independent-review-sidecar/1 uses approximate, uncalibrated encoded-video time.",
    "No v1 package can be promoted to ALLOWED; a new reviewed admission contract is required.",
  ];
}

type UnknownRecord = Record<string, unknown>;

interface ParsedArchive {
  readonly lineage: ArchiveArtifactLineage;
  readonly runtimeVersionPresent: boolean;
  readonly appBuildSha256Present: boolean;
}

interface ParsedJsonArtifact {
  readonly parsed: UnknownRecord;
  readonly identity: ByteArtifactLineage;
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const EPISODE_PATH = "episode.json";
const VIDEO_TIMING_PATH = "continuous.video-timing.json";
const ARCHIVE_CHECKSUM_PATH = "SHA256SUMS.txt";

const BLOCKERS = [
  "The relay.body-native-episode/2 archive does not retain a trusted pose-runtime version and immutable application build SHA-256.",
  "relay.media-clock-calibration/1 is not cryptographically bound to this episode JSON, continuous video, and video-timing artifact.",
  "relay.body-native-independent-review-sidecar/1 uses approximate, uncalibrated encoded-video time.",
  "No v1 package can be promoted to ALLOWED; a new reviewed admission contract is required.",
] as const;

export class AdmissionPackageValidationError extends Error {
  readonly issues: readonly string[];

  constructor(issues: readonly string[]) {
    const unique = Array.from(new Set(issues));
    super(unique.join(" "));
    this.name = "AdmissionPackageValidationError";
    this.issues = unique;
  }
}

function record(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function nonemptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function safeFileName(value: string): boolean {
  return (
    value.length > 0 &&
    value === value.trim() &&
    !value.includes("/") &&
    !value.includes("\\") &&
    value !== "." &&
    value !== ".." &&
    !/[\u0000-\u001f\u007f]/.test(value)
  );
}

function safeArchivePath(value: string): boolean {
  if (!value || value !== value.trim() || value.startsWith("/") || value.includes("\\")) {
    return false;
  }
  return value.split("/").every(
    (segment) =>
      segment.length > 0 &&
      segment !== "." &&
      segment !== ".." &&
      !/[\u0000-\u001f\u007f]/.test(segment),
  );
}

function byteArray(value: unknown): value is Uint8Array {
  // Browser/test realms can own different Uint8Array constructors.
  return Object.prototype.toString.call(value) === "[object Uint8Array]";
}

function decodeUtf8(bytes: Uint8Array, label: string): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new AdmissionPackageValidationError([`${label} is not valid UTF-8.`]);
  }
}

function parseJsonObject(text: string, label: string): UnknownRecord {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new AdmissionPackageValidationError([`${label} is not valid JSON.`]);
  }
  const object = record(parsed);
  if (!object) {
    throw new AdmissionPackageValidationError([`${label} must contain one JSON object.`]);
  }
  return object;
}

async function verifyChecksummedArtifact(
  artifact: RawChecksummedArtifact,
  label: string,
): Promise<ByteArtifactLineage> {
  const issues: string[] = [];
  if (!safeFileName(artifact.fileName)) issues.push(`${label} filename is unsafe.`);
  if (!safeFileName(artifact.checksumSidecar.fileName)) {
    issues.push(`${label} checksum-sidecar filename is unsafe.`);
  }
  if (!byteArray(artifact.bytes) || artifact.bytes.byteLength === 0) {
    issues.push(`${label} bytes are empty or invalid.`);
  }
  if (
    !byteArray(artifact.checksumSidecar.bytes) ||
    artifact.checksumSidecar.bytes.byteLength === 0
  ) {
    issues.push(`${label} checksum-sidecar bytes are empty or invalid.`);
  }
  if (issues.length > 0) throw new AdmissionPackageValidationError(issues);

  const sidecarText = decodeUtf8(
    artifact.checksumSidecar.bytes,
    `${label} checksum sidecar`,
  );
  const match = /^([a-f0-9]{64})  ([^\r\n]+)\n$/.exec(sidecarText);
  if (!match) {
    throw new AdmissionPackageValidationError([
      `${label} checksum sidecar must be one canonical lowercase SHA-256 line ending in LF.`,
    ]);
  }
  const [, declaredSha256, declaredFileName] = match;
  if (declaredFileName !== artifact.fileName) {
    issues.push(`${label} checksum sidecar names a different artifact.`);
  }
  const observedSha256 = await sha256Bytes(artifact.bytes);
  if (observedSha256 !== declaredSha256) {
    issues.push(`${label} SHA-256 does not match its checksum sidecar.`);
  }
  if (issues.length > 0) throw new AdmissionPackageValidationError(issues);

  return {
    fileName: artifact.fileName,
    bytes: artifact.bytes.byteLength,
    sha256: observedSha256,
    checksumSidecar: {
      fileName: artifact.checksumSidecar.fileName,
      bytes: artifact.checksumSidecar.bytes.byteLength,
      sha256: await sha256Bytes(artifact.checksumSidecar.bytes),
      declaredArtifactFileName: declaredFileName,
      declaredArtifactSha256: declaredSha256,
    },
  };
}

async function verifyJsonArtifact(
  artifact: RawChecksummedArtifact,
  label: string,
): Promise<ParsedJsonArtifact> {
  const identity = await verifyChecksummedArtifact(artifact, label);
  const text = decodeUtf8(artifact.bytes, label);
  const parsed = parseJsonObject(text, label);
  const producerCanonicalJson = `${JSON.stringify(parsed, null, 2)}\n`;
  if (text !== producerCanonicalJson) {
    throw new AdmissionPackageValidationError([
      `${label} is valid JSON but not the producer-canonical two-space serialization with one trailing LF.`,
    ]);
  }
  return { parsed, identity };
}

function parseArchiveChecksumManifest(
  bytes: Uint8Array,
): ReadonlyMap<string, string> {
  const text = decodeUtf8(bytes, ARCHIVE_CHECKSUM_PATH);
  const lines = text.split("\n");
  if (lines.at(-1) !== "") {
    throw new AdmissionPackageValidationError([
      `${ARCHIVE_CHECKSUM_PATH} must end in LF.`,
    ]);
  }
  lines.pop();
  const checksums = new Map<string, string>();
  const issues: string[] = [];
  for (const [index, line] of lines.entries()) {
    const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
    if (!match) {
      issues.push(`${ARCHIVE_CHECKSUM_PATH} line ${index + 1} is malformed.`);
      continue;
    }
    const [, sha256, path] = match;
    if (!safeArchivePath(path)) {
      issues.push(`${ARCHIVE_CHECKSUM_PATH} line ${index + 1} has an unsafe path.`);
    } else if (path === ARCHIVE_CHECKSUM_PATH) {
      issues.push(`${ARCHIVE_CHECKSUM_PATH} cannot checksum itself.`);
    } else if (checksums.has(path)) {
      issues.push(`${ARCHIVE_CHECKSUM_PATH} repeats ${path}.`);
    } else {
      checksums.set(path, sha256);
    }
  }
  if (checksums.size === 0) issues.push(`${ARCHIVE_CHECKSUM_PATH} is empty.`);
  if (issues.length > 0) throw new AdmissionPackageValidationError(issues);
  return checksums;
}

async function verifyArchive(
  artifact: RawChecksummedArtifact,
): Promise<ParsedArchive> {
  const identity = await verifyChecksummedArtifact(artifact, "archive");
  if (!artifact.fileName.endsWith(".relay-episode-v2.zip")) {
    throw new AdmissionPackageValidationError([
      "Archive filename must end with .relay-episode-v2.zip.",
    ]);
  }

  let members: Record<string, Uint8Array>;
  try {
    members = unzipSync(Uint8Array.from(artifact.bytes));
  } catch {
    throw new AdmissionPackageValidationError(["Archive bytes are not a valid ZIP archive."]);
  }
  const paths = Object.keys(members);
  const issues: string[] = [];
  for (const path of paths) {
    if (!safeArchivePath(path)) issues.push(`Archive member path ${path} is unsafe.`);
  }
  if (!members[ARCHIVE_CHECKSUM_PATH]) issues.push(`${ARCHIVE_CHECKSUM_PATH} is missing.`);
  if (!members[EPISODE_PATH]) issues.push(`${EPISODE_PATH} is missing.`);
  if (!members[VIDEO_TIMING_PATH]) issues.push(`${VIDEO_TIMING_PATH} is missing.`);
  if (issues.length > 0) throw new AdmissionPackageValidationError(issues);

  const declared = parseArchiveChecksumManifest(members[ARCHIVE_CHECKSUM_PATH]);
  const actualPaths = paths.filter((path) => path !== ARCHIVE_CHECKSUM_PATH).sort();
  for (const path of actualPaths) {
    if (!declared.has(path)) issues.push(`Archive member ${path} is not checksum-bound.`);
  }
  for (const path of declared.keys()) {
    if (!members[path]) issues.push(`Checksum-bound archive member ${path} is missing.`);
  }
  if (issues.length > 0) throw new AdmissionPackageValidationError(issues);

  const memberLineage: ArchiveMemberLineage[] = [];
  for (const path of actualPaths) {
    const observed = await sha256Bytes(members[path]);
    if (observed !== declared.get(path)) {
      issues.push(`Archive member ${path} fails SHA-256 verification.`);
    }
    memberLineage.push({
      path,
      bytes: members[path].byteLength,
      sha256: observed,
    });
  }
  if (issues.length > 0) throw new AdmissionPackageValidationError(issues);

  const episodeText = decodeUtf8(members[EPISODE_PATH], EPISODE_PATH);
  const episode = parseJsonObject(episodeText, EPISODE_PATH);
  if (episodeText !== canonicalJson(episode)) {
    throw new AdmissionPackageValidationError([
      `${EPISODE_PATH} is not the compact, recursively key-sorted canonical serialization.`,
    ]);
  }
  if (episode.schemaVersion !== "relay.body-native-episode/2") {
    issues.push("Archive episode schema must be relay.body-native-episode/2.");
  }
  if (episode.admissionStatus !== "QUARANTINED_PENDING_INDEPENDENT_REVIEW") {
    issues.push("Archive episode must retain its quarantined source status.");
  }
  const metadata = record(episode.metadata);
  const lineage = record(episode.lineage);
  const session = record(episode.session);
  const video = record(session?.video);
  const poseModel = record(session?.poseModel);
  const environment = record(episode.environment);
  if (!nonemptyString(metadata?.episodeCode)) issues.push("Archive episode ID is missing.");
  if (!nonemptyString(lineage?.episodeAncestorId)) {
    issues.push("Archive episode ancestor ID is missing.");
  }
  if (!nonemptyString(video?.fileName) || !safeFileName(video.fileName)) {
    issues.push("Archive continuous-video filename is invalid.");
  }
  if (!nonemptyString(video?.sha256) || !SHA256_PATTERN.test(video.sha256)) {
    issues.push("Archive continuous-video SHA-256 is invalid.");
  }
  if (!Number.isSafeInteger(video?.bytes) || (video?.bytes as number) <= 0) {
    issues.push("Archive continuous-video byte count is invalid.");
  }
  const videoPath = nonemptyString(video?.fileName)
    ? `continuous/${video.fileName}`
    : "continuous/(invalid)";
  const videoMember = memberLineage.find((member) => member.path === videoPath);
  if (!videoMember) {
    issues.push("Archive does not contain its declared continuous-video member.");
  } else if (
    videoMember.sha256 !== video?.sha256 ||
    videoMember.bytes !== video?.bytes
  ) {
    issues.push("Archive continuous-video bytes do not match the episode declaration.");
  }
  const timingMember = memberLineage.find(
    (member) => member.path === VIDEO_TIMING_PATH,
  );
  if (!timingMember) issues.push("Archive video-timing lineage is missing.");
  if (issues.length > 0 || !videoMember || !timingMember) {
    throw new AdmissionPackageValidationError(issues);
  }

  const runtimeVersionPresent =
    nonemptyString(poseModel?.runtimeLibrary) &&
    nonemptyString(poseModel?.runtimeVersion);
  const appBuildSha256Present =
    nonemptyString(environment?.appBuildSha256) &&
    SHA256_PATTERN.test(environment.appBuildSha256);

  return {
    runtimeVersionPresent,
    appBuildSha256Present,
    lineage: {
      ...identity,
      schemaVersion: "relay.body-native-episode/2",
      episodeId: metadata!.episodeCode as string,
      episodeAncestorId: lineage!.episodeAncestorId as string,
      episodeJsonSha256: declared.get(EPISODE_PATH)!,
      continuousVideoSha256: videoMember.sha256,
      continuousVideoPath: videoMember.path,
      videoTimingSha256: timingMember.sha256,
      members: memberLineage,
    },
  };
}

function requireMatchingString(
  issues: string[],
  observed: unknown,
  expected: string,
  label: string,
): void {
  if (observed !== expected) issues.push(`${label} does not match archive lineage.`);
}

async function verifyReview(
  artifact: RawChecksummedArtifact,
  archive: ArchiveArtifactLineage,
): Promise<ReviewArtifactLineage> {
  const { parsed, identity } = await verifyJsonArtifact(artifact, "review artifact");
  const issues: string[] = [];
  if (
    parsed.schemaVersion !==
    "relay.body-native-independent-review-sidecar/1"
  ) {
    issues.push("Review schema must be relay.body-native-independent-review-sidecar/1.");
  }
  if (parsed.admissionEffect !== "NONE_REMAINS_QUARANTINED") {
    issues.push("Review sidecar must not claim an admission effect.");
  }
  if (parsed.appendOnly !== true) issues.push("Review sidecar must be append-only.");

  const source = record(parsed.sourceEpisode);
  const replay = record(parsed.replayDeclaration);
  const entry = record(parsed.reviewEntry);
  const annotation = record(entry?.annotation);
  if (!source || !replay || !entry || !annotation) {
    issues.push("Review sidecar lineage, replay declaration, or annotation is missing.");
  } else {
    requireMatchingString(
      issues,
      source.episodeJsonSha256,
      archive.episodeJsonSha256,
      "Review source episode JSON SHA-256",
    );
    requireMatchingString(issues, source.episodeId, archive.episodeId, "Review source episode ID");
    requireMatchingString(
      issues,
      source.episodeAncestorId,
      archive.episodeAncestorId,
      "Review source episode ancestor",
    );
    requireMatchingString(
      issues,
      replay.episodeJsonSha256,
      archive.episodeJsonSha256,
      "Review replay episode JSON SHA-256",
    );
    requireMatchingString(issues, replay.episodeId, archive.episodeId, "Review replay episode ID");
    requireMatchingString(
      issues,
      replay.episodeAncestorId,
      archive.episodeAncestorId,
      "Review replay episode ancestor",
    );
    requireMatchingString(issues, entry.episodeId, archive.episodeId, "Review entry episode ID");
    requireMatchingString(
      issues,
      entry.episodeAncestorId,
      archive.episodeAncestorId,
      "Review entry episode ancestor",
    );
    if (replay.encodedOverlayAlignment !== "APPROXIMATE_UNCALIBRATED") {
      issues.push("Review replay must retain approximate, uncalibrated encoded-video time.");
    }
    if (!nonemptyString(annotation.annotationId)) {
      issues.push("Review annotation ID is missing.");
    }
    if (!nonemptyString(annotation.raterCode)) issues.push("Review rater code is missing.");
    if (replay.raterCode !== annotation.raterCode) {
      issues.push("Review replay declaration rater does not match its annotation.");
    }
  }
  if (issues.length > 0 || !source || !replay || !entry || !annotation) {
    throw new AdmissionPackageValidationError(issues);
  }

  return {
    ...identity,
    schemaVersion: "relay.body-native-independent-review-sidecar/1",
    sourceEpisodeJsonSha256: source.episodeJsonSha256 as string,
    episodeId: source.episodeId as string,
    episodeAncestorId: source.episodeAncestorId as string,
    annotationId: (annotation.annotationId as string).trim(),
    raterCode: (annotation.raterCode as string).trim(),
    encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
  };
}

async function verifyCalibration(
  artifact: RawChecksummedArtifact,
): Promise<CalibrationArtifactLineage> {
  const { parsed, identity } = await verifyJsonArtifact(
    artifact,
    "calibration artifact",
  );
  const issues: string[] = [];
  if (parsed.schemaVersion !== "relay.media-clock-calibration/1") {
    issues.push("Calibration schema must be relay.media-clock-calibration/1.");
  }
  if (
    parsed.status !== "FIT_REPORTED_NOT_ADMISSION_CALIBRATED" &&
    parsed.status !== "REJECTED"
  ) {
    issues.push("Calibration fit status is invalid.");
  }
  if (issues.length > 0) throw new AdmissionPackageValidationError(issues);
  return {
    ...identity,
    schemaVersion: "relay.media-clock-calibration/1",
    fitStatus: parsed.status as CalibrationArtifactLineage["fitStatus"],
    episodeBindingStatus: "ABSENT_FROM_CALIBRATION_SCHEMA_V1",
  };
}

function duplicateReviewIssues(
  reviews: readonly ReviewArtifactLineage[],
): string[] {
  const issues: string[] = [];
  const seenFileNames = new Set<string>();
  const seenDigests = new Set<string>();
  const seenAnnotations = new Set<string>();
  const seenRaters = new Set<string>();
  for (const review of reviews) {
    const fileName = review.fileName.toLowerCase();
    const annotation = [
      review.sourceEpisodeJsonSha256,
      review.annotationId.toLowerCase(),
    ].join(":");
    const rater = [
      review.sourceEpisodeJsonSha256,
      review.raterCode.toLowerCase(),
    ].join(":");
    if (seenFileNames.has(fileName)) issues.push(`Duplicate review filename ${review.fileName}.`);
    if (seenDigests.has(review.sha256)) issues.push(`Duplicate review bytes ${review.sha256}.`);
    if (seenAnnotations.has(annotation)) {
      issues.push(`Duplicate review annotation ID ${review.annotationId}.`);
    }
    if (seenRaters.has(rater)) issues.push(`Duplicate review rater ${review.raterCode}.`);
    seenFileNames.add(fileName);
    seenDigests.add(review.sha256);
    seenAnnotations.add(annotation);
    seenRaters.add(rater);
  }
  return issues;
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value as UnknownRecord)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

/**
 * Assemble cryptographic lineage for current v1 artifacts. Integrity-valid input
 * is still never admitted: v1 lacks the identities and time bindings required by
 * the downstream admission episode, and this module intentionally has no mapper.
 */
export async function preflightAdmissionPackageV1(
  input: AdmissionPackageV1Input,
): Promise<AdmissionPackagePreflightReport> {
  if (input.reviews.length === 0) {
    throw new AdmissionPackageValidationError([
      "At least one independent-review artifact is required for package preflight.",
    ]);
  }
  const parsedArchive = await verifyArchive(input.archive);
  const reviews = await Promise.all(
    input.reviews.map((review) => verifyReview(review, parsedArchive.lineage)),
  );
  const duplicates = duplicateReviewIssues(reviews);
  if (duplicates.length > 0) throw new AdmissionPackageValidationError(duplicates);
  const sortedReviews = [...reviews].sort(
    (left, right) =>
      left.fileName.localeCompare(right.fileName) ||
      left.sha256.localeCompare(right.sha256),
  );
  const calibration = await verifyCalibration(input.calibration);
  const lineage = {
    archive: parsedArchive.lineage,
    reviews: sortedReviews,
    calibration,
  } as const;
  const canonicalManifest = canonicalJson({
    schemaVersion: ADMISSION_PACKAGE_PREFLIGHT_SCHEMA,
    lineage,
  });
  const packageRootSha256 = await sha256Bytes(
    new TextEncoder().encode(canonicalManifest),
  );

  const gates: readonly AdmissionPackagePreflightGate[] = [
    {
      id: "byte-and-checksum-integrity",
      passed: true,
      issues: [],
      facts: {
        artifactCount: 2 + sortedReviews.length,
        allOuterChecksumSidecarsVerified: true,
      },
    },
    {
      id: "archive-member-lineage",
      passed: true,
      issues: [],
      facts: {
        archiveMemberCount: parsedArchive.lineage.members.length,
        allArchiveMembersChecksumBound: true,
        episodeJsonSha256: parsedArchive.lineage.episodeJsonSha256,
      },
    },
    {
      id: "append-only-review-lineage",
      passed: true,
      issues: [],
      facts: {
        reviewArtifactCount: sortedReviews.length,
        duplicateReviewArtifacts: false,
        allReviewsEpisodeBound: true,
      },
    },
    {
      id: "runtime-and-app-build-identity",
      passed: false,
      issues: [BLOCKERS[0]],
      facts: {
        untrustedExtensionRuntimeVersionPresent:
          parsedArchive.runtimeVersionPresent,
        untrustedExtensionAppBuildSha256Present:
          parsedArchive.appBuildSha256Present,
        trustedByEpisodeV2Schema: false,
      },
    },
    {
      id: "episode-video-timing-calibration-binding",
      passed: false,
      issues: [BLOCKERS[1]],
      facts: {
        episodeJsonBound: false,
        continuousVideoBound: false,
        videoTimingArtifactBound: false,
        calibrationFitStatus: calibration.fitStatus,
      },
    },
    {
      id: "review-timeline-calibration",
      passed: false,
      issues: [BLOCKERS[2]],
      facts: {
        approximateUncalibratedReviewCount: sortedReviews.length,
        calibratedReviewCount: 0,
      },
    },
    {
      id: "v1-admission-contract",
      passed: false,
      issues: [BLOCKERS[3]],
      facts: {
        admissionImplementationPresent: false,
        mapsToContinuousCaptureAdmissionEpisode: false,
      },
    },
  ];

  return deepFreeze({
    schemaVersion: ADMISSION_PACKAGE_PREFLIGHT_SCHEMA,
    status: "QUARANTINED",
    admitted: false,
    evidenceClass: "QUARANTINED_UNADMITTED",
    permittedUse: "integrity-and-lineage-audit-only",
    lineage,
    packageRoot: {
      algorithm: "SHA-256",
      canonicalManifest,
      sha256: packageRootSha256,
    },
    gates,
    blockers: BLOCKERS,
  });
}
