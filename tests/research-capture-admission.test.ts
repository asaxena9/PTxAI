import { describe, expect, it } from "vitest";
import type {
  BlindedIndependentReviewDeclaration,
  ResearchAnnotationAdjudication,
  ResearchAnnotationReviewEntry,
} from "../src/research/annotation-review";
import {
  identifyCanonicalCaptureArtifact,
  verifyContinuousCaptureAdmission,
  type CaptureAdmissionGateId,
  type CaptureAdmissionPoseFrame,
  type CaptureAdmissionReport,
  type ContinuousCaptureAdmissionEpisode,
} from "../src/research/capture-admission";
import { deriveResearchCaptureAids } from "../src/research/capture-aids";
import type {
  CaptureInterval,
  VideoFrameTimingSample,
} from "../src/research/capture-timing";
import type { ResearchAnnotation } from "../src/research/capture-bundle";
import type { LandmarkSet } from "../src/motion/types";

const EPISODE_ID = "episode-001";
const ANCESTOR_ID = "P001/S001/E001";
const VIDEO_SHA256 = "a".repeat(64);
const APP_SHA256 = "b".repeat(64);
const MODEL_SHA256 = "c".repeat(64);
const METRIC_ID = "ipsilateral-shoulder-abduction-included-angle";

type DeepMutable<T> = T extends (...args: never[]) => unknown
  ? T
  : T extends readonly (infer Item)[]
    ? DeepMutable<Item>[]
    : T extends object
      ? { -readonly [Key in keyof T]: DeepMutable<T[Key]> }
      : T;

function mutableClone<T>(value: T): DeepMutable<T> {
  return structuredClone(value) as DeepMutable<T>;
}

function verifyMutableEpisode(
  value: DeepMutable<ContinuousCaptureAdmissionEpisode>,
): Promise<CaptureAdmissionReport> {
  return verifyContinuousCaptureAdmission(
    value as unknown as ContinuousCaptureAdmissionEpisode,
  );
}

function videoFrame(
  sequence: number,
  callbackMonotonicTimeMs: number,
  mediaTimeMs: number,
): VideoFrameTimingSample {
  return {
    sequence,
    callbackMonotonicTimeMs,
    mediaTimeMs,
    presentedFrames: sequence + 100,
  };
}

function landmarks(offset: number): LandmarkSet {
  return {
    "left-hip": {
      x: 0.4 + offset,
      y: 0.7,
      z: -0.05,
      visibility: 0.95,
      presence: 0.96,
    },
    "left-shoulder": {
      x: 0.42 + offset,
      y: 0.4,
      z: -0.04,
      visibility: 0.93,
      presence: 0.94,
    },
    "left-elbow": {
      x: 0.6 + offset,
      y: 0.42,
      z: -0.02,
      visibility: 0.91,
      presence: 0.92,
    },
  };
}

function poseFrame(
  sequenceNumber: number,
  callbackMonotonicTimeMs: number,
  tMs: number,
): CaptureAdmissionPoseFrame {
  const image = landmarks(sequenceNumber * 0.01);
  const world = landmarks(sequenceNumber * 0.02);
  return {
    sequenceNumber,
    tMs,
    sourceTimestampMs: 1_000 + tMs,
    callbackMonotonicTimeMs,
    frameWidth: 1280,
    frameHeight: 720,
    poseDetected: true,
    imageJointCount: Object.keys(image).length,
    worldJointCount: Object.keys(world).length,
    image,
    world,
  };
}

function annotation(
  annotationId: string,
  raterCode: string,
  createdAt: string,
  overrides: Partial<ResearchAnnotation> = {},
): ResearchAnnotation {
  return {
    annotationId,
    raterCode,
    raterRole: "movement-professional",
    reviewStatus: "draft-unreviewed",
    movementIdentity: "lateral arm raise",
    assessment: "acceptable",
    phaseStartMs: 10,
    phaseEndMs: 50,
    evidenceJoints: ["left-hip", "left-shoulder", "left-elbow"],
    evidenceStartMs: 10,
    evidenceEndMs: 50,
    metric: METRIC_ID,
    observedValue: 87,
    targetValue: 87,
    delta: 0,
    unit: "degrees",
    desiredTargetSource: "captured-reference",
    cue: "Match the accepted reference angle.",
    confidence: 0.8,
    createdAt,
    ...overrides,
  };
}

function declaration(
  value: ResearchAnnotation,
  declaredAt: string,
): BlindedIndependentReviewDeclaration {
  return {
    annotationId: value.annotationId,
    raterCode: value.raterCode,
    episodeId: EPISODE_ID,
    episodeAncestorId: ANCESTOR_ID,
    blinded: true,
    independentlyProduced: true,
    declaredAt,
  };
}

function reviewEntry(
  value: ResearchAnnotation,
  declaredAt: string,
): ResearchAnnotationReviewEntry {
  return {
    episodeId: EPISODE_ID,
    episodeAncestorId: ANCESTOR_ID,
    annotation: value,
    independentReviewDeclaration: declaration(value, declaredAt),
  };
}

async function validEpisode(): Promise<ContinuousCaptureAdmissionEpisode> {
  const videoFrameTimings = [
    videoFrame(0, 100, 0),
    videoFrame(1, 120, 20),
    videoFrame(2, 140, 40),
    videoFrame(3, 160, 60),
    videoFrame(4, 180, 80),
  ];
  const intervals: CaptureInterval[] = [
    {
      kind: "reference",
      begin: videoFrameTimings[0],
      end: videoFrameTimings[1],
    },
    {
      kind: "attempt",
      begin: videoFrameTimings[2],
      end: videoFrameTimings[4],
    },
  ];
  const rawPoseFrames = [poseFrame(0, 110, 10), poseFrame(1, 150, 50)];
  const derivedAids = deriveResearchCaptureAids(rawPoseFrames);
  const rawIdentity = await identifyCanonicalCaptureArtifact(rawPoseFrames);
  const aidsIdentity = await identifyCanonicalCaptureArtifact(derivedAids);
  const protocol = {
    schemaVersion: "relay.observable-movement-protocol/1",
    protocolId: "relay.lateral-arm-raise.reference-relative",
    protocolVersion: "1.0.0",
    target: {
      metricId: "ipsilateral-shoulder-abduction-included-angle",
      source: "captured-reference",
    },
    prohibitedClaims: ["clinical correctness or safety"],
  };
  const protocolIdentity = await identifyCanonicalCaptureArtifact(protocol);
  const execution = {
    schemaVersion: "relay.observable-protocol-execution/1" as const,
    evidenceClass: "DERIVED" as const,
    interpretation:
      "reference-relative-observable-geometry-not-anatomical-or-clinical-truth" as const,
    protocolId: protocol.protocolId,
    protocolVersion: protocol.protocolVersion,
    protocolSha256: protocolIdentity.sha256,
    metricId: METRIC_ID,
    status: "observable" as const,
    primaryThreshold: 0.65 as const,
    sensitivity: [],
    sensitivityDirectionStable: true,
    referencePeakDegrees: 87,
    attemptPeakDegrees: 87,
    targetMinusObservedDegrees: 0,
    direction: "matched-reference-peak" as const,
    evidence: {
      referenceSequenceNumber: 0,
      referenceSessionTimeMs: 10,
      referenceMediaTimeMs: 0,
      attemptSequenceNumber: 1,
      attemptSessionTimeMs: 50,
      attemptMediaTimeMs: 40,
    },
    claimBoundary: ["Not clinical truth."],
  };
  const executionIdentity = await identifyCanonicalCaptureArtifact(execution);
  const first = annotation(
    "A1",
    "R1",
    "2026-08-12T03:00:00.000Z",
    {
      protocolSha256: protocolIdentity.sha256,
      protocolExecutionSha256: executionIdentity.sha256,
      referenceEvidenceSequenceNumber: 0,
      attemptEvidenceSequenceNumber: 1,
    },
  );
  const second = annotation(
    "A2",
    "R2",
    "2026-08-12T03:02:00.000Z",
    {
      protocolSha256: protocolIdentity.sha256,
      protocolExecutionSha256: executionIdentity.sha256,
      referenceEvidenceSequenceNumber: 0,
      attemptEvidenceSequenceNumber: 1,
    },
  );

  return {
    episodeId: EPISODE_ID,
    episodeAncestorId: ANCESTOR_ID,
    rawVideoEnabled: true,
    declaredPoseFrameCount: rawPoseFrames.length,
    declaredVideoFrameTimingCount: videoFrameTimings.length,
    declaredIntervalCount: intervals.length,
    rawPoseFrames,
    videoFrameTimings,
    intervals,
    rawPoseArtifact: {
      ...rawIdentity,
      frameCount: rawPoseFrames.length,
    },
    derivedAidsArtifact: {
      ...aidsIdentity,
      frameCount: rawPoseFrames.length,
      derivedFromRawSha256: rawIdentity.sha256,
      aids: derivedAids,
    },
    frozenProtocol: {
      protocol,
      canonicalJson: protocolIdentity.canonicalJson,
      sha256: protocolIdentity.sha256,
    },
    protocolExecution: {
      execution,
      ...executionIdentity,
    },
    runtimePoseModel: {
      modelName: "MediaPipe Pose Landmarker Full",
      runtimeLibrary: "@mediapipe/tasks-vision",
      runtimeVersion: "1.0.0",
      assetPath: "/models/pose_landmarker_full.task",
      assetBytes: 9_398_198,
      assetSha256: MODEL_SHA256,
      realizedDelegate: "gpu-requested-runtime-fallback-observed",
    },
    appEnvironment: {
      appName: "Relay",
      appVersion: "0.1.0",
      appBuildSha256: APP_SHA256,
      collectorSchemaVersion: "relay.body-native-episode/2",
      userAgent: "test-browser/1",
      platform: "test-platform",
      locale: "en-US",
      timeZone: "America/Los_Angeles",
      capturedAt: "2026-08-12T02:55:00.000Z",
    },
    video: {
      fileName: "episode-001.webm",
      mimeType: "video/webm",
      bytes: 2_048,
      sha256: VIDEO_SHA256,
      captureTopology: "one-continuous-session",
    },
    replayAudit: {
      encodedReplayTimelineCalibrated: true,
      status: "passed",
      auditedAt: "2026-08-12T03:10:00.000Z",
      auditorCode: "replay-reviewer-1",
      rawPoseSha256: rawIdentity.sha256,
      derivedAidsSha256: aidsIdentity.sha256,
      videoSha256: VIDEO_SHA256,
      replayedPoseFrameCount: rawPoseFrames.length,
      continuousVideoDecoded: true,
      poseOverlayRendered: true,
      referenceAttemptIntervalsReviewed: true,
      timingSessionRevalidated: true,
    },
    collectionScope: {
      participantScope: "adult-researcher-self",
      dataUseScope: "internal-methods-only",
      recordingRightsConfirmed: true,
      nonClinicalResearchAcknowledged: true,
      selfAttestedAt: "2026-08-12T02:50:00.000Z",
    },
    review: {
      storageMode: "append-only",
      claimedFinalState: "adjudicated",
      entries: [
        reviewEntry(first, "2026-08-12T03:01:00.000Z"),
        reviewEntry(second, "2026-08-12T03:03:00.000Z"),
      ],
    },
  };
}

function admissionGate(
  report: CaptureAdmissionReport,
  id: CaptureAdmissionGateId,
) {
  const result = report.gates.find((entry) => entry.id === id);
  if (!result) throw new Error(`Missing admission gate ${id}`);
  return result;
}

describe("continuous collector v2 admission", () => {
  it("admits only a complete hash-bound nonclinical mechanics pilot", async () => {
    const episode = await validEpisode();
    const before = structuredClone(episode);

    const report = await verifyContinuousCaptureAdmission(episode);

    expect(report).toMatchObject({
      status: "ALLOWED_MEASURED_PILOT_MECHANICS_ONLY",
      admitted: true,
      evidenceClass: "MEASURED_PILOT_MECHANICS_ONLY",
      clinicalTruthStatus: "NOT_ESTABLISHED",
      permittedUse: "bounded-nonclinical-mechanics-methods-pilot",
      issues: [],
    });
    expect(report.gates).toHaveLength(14);
    expect(report.gates.every((entry) => entry.passed)).toBe(true);
    expect(report.prohibitedClaims).toEqual(
      expect.arrayContaining([
        "clinical correctness",
        "safety",
        "general body intelligence",
      ]),
    );
    expect(episode).toEqual(before);
  });

  it("fails closed on nonfinite, backward, out-of-range, or count-forged pose fields", async () => {
    const episode = mutableClone(await validEpisode());
    episode.rawPoseFrames[1].sourceTimestampMs = 999;
    episode.rawPoseFrames[1].world["left-elbow"]!.x = Number.NaN;
    episode.rawPoseFrames[1].image["left-elbow"]!.visibility = 1.01;
    episode.rawPoseFrames[0].imageJointCount = 99;

    const report = await verifyMutableEpisode(episode);
    const pose = admissionGate(report, "pose-integrity");

    expect(report.status).toBe("QUARANTINED");
    expect(pose.passed).toBe(false);
    expect(pose.issues.join(" ")).toMatch(/sourceTimestampMs moved backward/);
    expect(pose.issues.join(" ")).toMatch(/x must be finite/);
    expect(pose.issues.join(" ")).toMatch(/visibility must be finite and within/);
    expect(pose.issues.join(" ")).toMatch(/imageJointCount does not equal/);
    expect(admissionGate(report, "derived-aids-artifact").passed).toBe(false);
  });

  it("requires exact counts and a fully valid shared continuous timing trace", async () => {
    const episode = mutableClone(await validEpisode());
    episode.declaredPoseFrameCount += 1;
    episode.derivedAidsArtifact.frameCount += 1;
    episode.videoFrameTimings[3].callbackMonotonicTimeMs = 130;

    const report = await verifyMutableEpisode(episode);

    expect(report.status).toBe("QUARANTINED");
    expect(admissionGate(report, "exact-frame-counts")).toMatchObject({
      passed: false,
      issues: expect.arrayContaining([
        expect.stringMatching(/declaredPoseFrameCount/),
        expect.stringMatching(/derivedAidsArtifact.frameCount/),
      ]),
    });
    expect(admissionGate(report, "continuous-timing")).toMatchObject({
      passed: false,
      issues: expect.arrayContaining([expect.stringMatching(/backward-time/)]),
    });
    expect(admissionGate(report, "replay-audit").passed).toBe(false);
  });

  it("binds exact canonical raw and derived bytes, digests, and derivation lineage", async () => {
    const episode = mutableClone(await validEpisode());
    episode.rawPoseArtifact.canonicalJson += "\n";
    episode.rawPoseArtifact.sha256 = "0".repeat(64);
    episode.derivedAidsArtifact.derivedFromRawSha256 = "1".repeat(64);
    episode.derivedAidsArtifact.aids.frames[0].observations.image.coverageFraction =
      0.123;

    const report = await verifyMutableEpisode(episode);
    const raw = admissionGate(report, "raw-pose-artifact");
    const aids = admissionGate(report, "derived-aids-artifact");

    expect(report.status).toBe("QUARANTINED");
    expect(raw.issues.join(" ")).toMatch(/exact canonical UTF-8 serialization/);
    expect(raw.issues.join(" ")).toMatch(/sha256 does not match/);
    expect(aids.issues.join(" ")).toMatch(/derivedFromRawSha256/);
    expect(aids.issues.join(" ")).toMatch(
      /do not exactly equal deriveResearchCaptureAids/,
    );
  });

  it("requires immutable protocol, model asset, app environment, video, and replay identity", async () => {
    const episode = mutableClone(await validEpisode());
    episode.frozenProtocol.protocol.protocolVersion = "mutated";
    episode.runtimePoseModel.assetBytes = 0;
    episode.runtimePoseModel.assetSha256 = MODEL_SHA256.toUpperCase();
    episode.appEnvironment.userAgent = "";
    episode.appEnvironment.appBuildSha256 = "not-a-sha";
    episode.video!.bytes = 0;
    episode.video!.captureTopology = "split-clips" as "one-continuous-session";
    episode.replayAudit.poseOverlayRendered = false;
    episode.replayAudit.videoSha256 = "f".repeat(64);

    const report = await verifyMutableEpisode(episode);

    expect(report.status).toBe("QUARANTINED");
    for (const id of [
      "immutable-protocol",
      "runtime-model-identity",
      "app-environment-identity",
      "continuous-video-identity",
      "replay-audit",
    ] as const) {
      expect(admissionGate(report, id).passed).toBe(false);
    }
  });

  it("admits no broader participant or data-use scope by self-attestation", async () => {
    const episode = mutableClone(await validEpisode());
    episode.collectionScope.participantScope =
      "patient" as "adult-researcher-self";
    episode.collectionScope.dataUseScope = "public" as "internal-methods-only";
    episode.collectionScope.recordingRightsConfirmed = false as true;

    const report = await verifyMutableEpisode(episode);
    const scope = admissionGate(report, "collection-scope");

    expect(report.status).toBe("QUARANTINED");
    expect(scope.passed).toBe(false);
    expect(scope.facts.legalOrIRBApprovalEstablished).toBe(false);
    expect(scope.issues.join(" ")).toMatch(/adult-researcher-self/);
    expect(scope.issues.join(" ")).toMatch(/internal-methods-only/);
  });

  it("requires append-only independent review to derive final adjudicated state", async () => {
    const episode = mutableClone(await validEpisode());
    episode.review.storageMode = "mutable" as "append-only";
    episode.review.entries = episode.review.entries.slice(0, 1);

    const report = await verifyMutableEpisode(episode);
    const review = admissionGate(report, "append-only-independent-review");

    expect(report.status).toBe("QUARANTINED");
    expect(review.passed).toBe(false);
    expect(review.issues.join(" ")).toMatch(/storageMode must be append-only/);
    expect(review.issues.join(" ")).toMatch(/needs-second-review/);
    expect(review.facts.clinicalTruthEstablished).toBe(false);
  });

  it("quarantines independent disagreement until a distinct third rater adjudicates", async () => {
    const episode = mutableClone(await validEpisode());
    episode.review.entries[1].annotation = {
      ...episode.review.entries[1].annotation,
      cue: "A different non-clinical description.",
    };

    const unadjudicated = await verifyMutableEpisode(episode);
    expect(unadjudicated.status).toBe("QUARANTINED");
    expect(
      admissionGate(unadjudicated, "append-only-independent-review").issues.join(
        " ",
      ),
    ).toMatch(/needs-adjudication/);

    const thirdAnnotation = annotation(
      "A3",
      "R3",
      "2026-08-12T03:04:00.000Z",
      {
        cue: "Third-rater non-clinical description.",
        protocolSha256: episode.frozenProtocol.sha256,
        protocolExecutionSha256: episode.protocolExecution.sha256,
        referenceEvidenceSequenceNumber: 0,
        attemptEvidenceSequenceNumber: 1,
      },
    );
    const adjudication: ResearchAnnotationAdjudication = {
      episodeId: EPISODE_ID,
      episodeAncestorId: ANCESTOR_ID,
      annotation: thirdAnnotation,
      reviewedAnnotationIds: ["A1", "A2"],
      declaredAt: "2026-08-12T03:05:00.000Z",
    };
    episode.review.adjudication = mutableClone(adjudication);

    const adjudicated = await verifyMutableEpisode(episode);
    expect(adjudicated.status).toBe(
      "ALLOWED_MEASURED_PILOT_MECHANICS_ONLY",
    );
    expect(
      admissionGate(adjudicated, "append-only-independent-review"),
    ).toMatchObject({
      passed: true,
      facts: {
        derivedState: "adjudicated",
        resolution: "third-rater-adjudication",
        clinicalTruthEstablished: false,
      },
    });
  });

  it("quarantines a pose-only fixture from the continuous-video collector-v2 admission class", async () => {
    const episode = mutableClone(await validEpisode());
    episode.rawVideoEnabled = false;
    delete episode.video;
    delete episode.replayAudit.videoSha256;
    episode.replayAudit.continuousVideoDecoded = false;

    const report = await verifyMutableEpisode(episode);

    expect(report.status).toBe("QUARANTINED");
    expect(admissionGate(report, "continuous-video-identity")).toMatchObject({
      passed: false,
      facts: { rawVideoEnabled: false, retainedVideoAssetCount: 0 },
    });
  });

  it("returns quarantine rather than throwing on invalid review structure", async () => {
    const episode = mutableClone(await validEpisode());
    episode.review.entries[1].independentReviewDeclaration = undefined;

    const report = await verifyMutableEpisode(episode);

    expect(report.status).toBe("QUARANTINED");
    expect(
      admissionGate(report, "append-only-independent-review").issues.join(" "),
    ).toMatch(/requires a blinded independent review declaration/);
    expect(report.clinicalTruthStatus).toBe("NOT_ESTABLISHED");
  });

  it("quarantines two matching evidence-free claim reviews", async () => {
    const episode = mutableClone(await validEpisode());
    for (const entry of episode.review.entries) {
      entry.annotation = {
        ...entry.annotation,
        movementIdentity: "",
        evidenceJoints: [],
        evidenceStartMs: undefined,
        evidenceEndMs: undefined,
        metric: undefined,
        observedValue: undefined,
        targetValue: undefined,
        delta: undefined,
        unit: undefined,
        protocolSha256: undefined,
        protocolExecutionSha256: undefined,
        referenceEvidenceSequenceNumber: undefined,
        attemptEvidenceSequenceNumber: undefined,
      };
    }

    const report = await verifyMutableEpisode(episode);
    const review = admissionGate(report, "append-only-independent-review");

    expect(report.status).toBe("QUARANTINED");
    expect(review.passed).toBe(false);
    expect(review.issues.join(" ")).toMatch(
      /movement identity is required[\s\S]*evidence interval is required/,
    );
    expect(review.facts.clinicalTruthEstablished).toBe(false);
    expect(report.clinicalTruthStatus).toBe("NOT_ESTABLISHED");
  });

  it("quarantines an observable review whose source hashes or executor evidence do not bind", async () => {
    const episode = mutableClone(await validEpisode());
    for (const entry of episode.review.entries) {
      entry.annotation.protocolExecutionSha256 = "d".repeat(64);
      entry.annotation.attemptEvidenceSequenceNumber = 99;
    }

    const report = await verifyMutableEpisode(episode);
    const review = admissionGate(report, "append-only-independent-review");

    expect(report.status).toBe("QUARANTINED");
    expect(review.issues.join(" ")).toMatch(/protocolExecutionSha256/);
    expect(review.issues.join(" ")).toMatch(/attemptEvidenceSequenceNumber/);
    expect(report.clinicalTruthStatus).toBe("NOT_ESTABLISHED");
  });

  it("quarantines replay until source-to-encoded timeline calibration is measured", async () => {
    const episode = mutableClone(await validEpisode());
    episode.replayAudit.encodedReplayTimelineCalibrated = false as true;

    const report = await verifyMutableEpisode(episode);
    const replay = admissionGate(report, "replay-audit");

    expect(report.status).toBe("QUARANTINED");
    expect(replay.passed).toBe(false);
    expect(replay.issues.join(" ")).toMatch(/source timeline.*encoded MediaRecorder/i);
    expect(report.clinicalTruthStatus).toBe("NOT_ESTABLISHED");
  });
});
