import { describe, expect, it } from "vitest";
import type { LandmarkSet } from "../src/motion/types";
import {
  buildResearchReplayAttachments,
  buildResearchExport,
  canonicalJson,
  RESEARCH_COLLECTOR_BUILD_ID,
  sha256Bytes,
  sha256Text,
  storeResearchAids,
  type ResearchEpisodeDraft,
  type ResearchPoseFrame,
  type ResearchReplayAudit,
  validateResearchDraft,
} from "../src/research/capture-bundle";
import {
  auditLocalReplayIntegrity,
  RECORDED_BLOB_TIMELINE_MAPPING,
  REPLAY_INTEGRITY_EVIDENCE_SCOPE,
  type ReplayPoseAttachment,
} from "../src/research/capture-replay";
import type {
  CaptureInterval,
  VideoFrameTimingSample,
} from "../src/research/capture-timing";
import { executeObservableProtocol } from "../src/research/protocol-executor";
import {
  freezeObservableProtocol,
  LATERAL_ARM_RAISE_REFERENCE_V1,
} from "../src/research/protocol";

function timing(sequence: number): VideoFrameTimingSample {
  return {
    sequence,
    callbackMonotonicTimeMs: 1000 + sequence * 100,
    mediaTimeMs: sequence * 100,
    presentedFrames: 10 + sequence,
  };
}

function poseForAngle(degrees: number): LandmarkSet {
  const radians = degrees * (Math.PI / 180);
  return {
    "left-shoulder": { x: 0.5, y: 0.5, z: 0, visibility: 1, presence: 1 },
    "left-hip": { x: 0.5, y: 0.8, z: 0, visibility: 1, presence: 1 },
    "left-elbow": {
      x: 0.5 + Math.sin(radians) * 0.2,
      y: 0.5 + Math.cos(radians) * 0.2,
      z: 0,
      visibility: 1,
      presence: 1,
    },
  };
}

function frame(sequenceNumber: number): ResearchPoseFrame {
  const sample = timing(sequenceNumber);
  const angle = sequenceNumber >= 2 && sequenceNumber <= 6 ? 90 : 50;
  const image = poseForAngle(angle);
  return {
    sequenceNumber,
    tMs: sequenceNumber * 100,
    sourceTimestampMs: sample.callbackMonotonicTimeMs,
    callbackMonotonicTimeMs: sample.callbackMonotonicTimeMs,
    frameWidth: 100,
    frameHeight: 100,
    poseDetected: true,
    imageJointCount: 3,
    worldJointCount: 0,
    image,
    world: {},
    videoTiming: {
      strategy: "nearest-latest-causal-video-frame",
      ...sample,
    },
  };
}

async function draft(): Promise<ResearchEpisodeDraft> {
  const protocol = await freezeObservableProtocol(
    LATERAL_ARM_RAISE_REFERENCE_V1,
  );
  const videoFrames = Array.from({ length: 15 }, (_, index) => timing(index));
  const poseFrames = Array.from({ length: 15 }, (_, index) => frame(index));
  const intervals: CaptureInterval[] = [
    { kind: "reference", begin: timing(2), end: timing(6) },
    { kind: "attempt", begin: timing(8), end: timing(12) },
  ];
  const videoBlob = new Blob([new Uint8Array([1, 2, 3])], {
    type: "video/webm",
  });
  const videoSha = await sha256Bytes(
    new Uint8Array(await videoBlob.arrayBuffer()),
  );
  const attachments: ReplayPoseAttachment[] = poseFrames.map((pose) => ({
    strategy: "nearest-latest-causal-video-frame",
    pose: {
      sequence: pose.sequenceNumber,
      callbackMonotonicTimeMs: pose.callbackMonotonicTimeMs,
    },
    videoFrame: pose.videoTiming as VideoFrameTimingSample,
  }));
  const integrityAudit = await auditLocalReplayIntegrity({
    declaredVideo: {
      fileName: "continuous.webm",
      mimeType: "video/webm",
      bytes: videoBlob.size,
      sha256: videoSha,
    },
    observedVideo: { fileName: "continuous.webm", blob: videoBlob },
    decodedVideo: {
      status: "DECODED_LOCAL_VIDEO",
      durationMs: 1400,
      videoWidth: 100,
      videoHeight: 100,
      seekableStartMs: 0,
      seekableEndMs: 1400,
      observedDecodedFrameMediaTimeMs: 0,
      sourceTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
      inspectedAt: "2026-08-12T00:00:00.000Z",
    },
    videoFrames,
    poseAttachments: attachments,
    intervals,
    confirmation: {
      reviewerCode: "RR1",
      watchedContinuousVideo: true,
      overlaysVisible: true,
      referenceIntervalConfirmed: true,
      attemptIntervalConfirmed: true,
      declaredAt: "2026-08-12T00:00:00.000Z",
      evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
    },
  });
  expect(integrityAudit.valid).toBe(true);
  const replayAudit: ResearchReplayAudit = {
    status: "PASS",
    videoAssetSha256: videoSha,
    poseFrameCount: poseFrames.length,
    videoTimingCount: videoFrames.length,
    intervalBoundaryCoverage: true,
    causalOverlaySelection: true,
    humanConfirmation: {
      reviewerCode: "RR1",
      watchedContinuousVideo: true,
      overlaysVisible: true,
      referenceIntervalConfirmed: true,
      attemptIntervalConfirmed: true,
      declaredAt: "2026-08-12T00:00:00.000Z",
      evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
    },
    encodedReplayTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
    encodedReplayTimelineCalibrated: false,
    renderedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
    issues: [],
    integrityAudit,
  };
  const execution = executeObservableProtocol(protocol, poseFrames, intervals);
  const executionSha = await sha256Text(canonicalJson(execution));
  return {
    metadata: {
      participantCode: "SELF-P001",
      sessionCode: "S001",
      episodeCode: "E001",
      movementFamily: protocol.protocol.movementFamily,
      cameraView: "front",
      deviceNotes: "test",
      calibrationNotes: "fixed front view",
    },
    consent: {
      adultResearcherSelfConfirmed: true,
      collectorIsParticipant: true,
      recordingRightsConfirmed: true,
      nonClinicalResearchAcknowledged: true,
      dataUse: "internal-methods-validation-only",
      rawVideoIncluded: true,
      authorizationStatus: "self-attested-limited-pilot",
      formVersion: "relay-research-self-capture-v2",
      confirmedAt: "2026-08-12T00:00:00.000Z",
    },
    protocol,
    session: {
      sessionCaptureId: "continuous-1",
      startedAt: "2026-08-12T00:00:00.000Z",
      endedAt: "2026-08-12T00:00:02.000Z",
      durationMs: 1400,
      cameraFacing: "user",
      cameraSettings: {},
      clocks: {
        performanceTimeOriginMs: 0,
        sessionStartedMonotonicMs: 1000,
        videoRecorderStartedMonotonicMs: 1000,
        videoFrameClock: "requestVideoFrameCallback.mediaTime",
        poseClock: "performance.now",
        poseVideoSynchronization: "latest-past-causal-video-frame-callback",
      },
      poseModel: {
        name: "MediaPipe Pose Landmarker Full",
        asset: "/models/pose_landmarker_full.task",
        assetBytes: 9398198,
        assetSha256:
          "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
        inferenceCadenceMs: 66,
        requestedDelegate: "GPU",
        realizedDelegate: "unobserved-gpu-with-cpu-fallback",
      },
      frameCount: poseFrames.length,
      poseFrames,
      videoFrameTimings: videoFrames,
      intervals,
      derivedAids: await storeResearchAids(poseFrames),
      protocolExecution: execution,
      video: {
        fileName: "continuous.webm",
        mimeType: "video/webm",
        bytes: videoBlob.size,
        sha256: videoSha,
        continuousEpisode: true,
      },
      replayAudit,
    },
    annotations: [
      {
        annotationId: "A1",
        raterCode: "R1",
        raterRole: "collector",
        reviewStatus: "draft-unreviewed",
        movementIdentity: protocol.protocol.movementFamily,
        assessment: "uncertain",
        realizedErrorType:
          "derived reference-relative estimator difference; practical significance uncalibrated",
        evidenceJoints: [...protocol.protocol.target.evidenceJoints],
        evidenceStartMs: execution.evidence?.attemptSessionTimeMs,
        evidenceEndMs: execution.evidence?.attemptSessionTimeMs,
        evidenceClock: "session-pose-callback-offset-ms",
        metric: protocol.protocol.target.metricId,
        observedValue: execution.attemptPeakDegrees,
        targetValue: execution.referencePeakDegrees,
        delta: execution.targetMinusObservedDegrees,
        unit: "degrees",
        desiredTargetSource: "captured-reference",
        confidence: 0.8,
        protocolSha256: protocol.sha256,
        protocolExecutionSha256: executionSha,
        createdAt: "2026-08-12T00:00:01.000Z",
      },
    ],
    environment: {
      collectorBuildId: RESEARCH_COLLECTOR_BUILD_ID,
      userAgent: "test",
      platform: "test",
      language: "en",
      timezone: "UTC",
      secureContext: true,
      origin: "https://example.test",
      recordedAt: "2026-08-12T00:00:00.000Z",
    },
  };
}

describe("research capture bundle v2", () => {
  it("uses one minimal replay-attachment shape across capture and export", () => {
    const poseFrames = [frame(0), frame(1)];
    const attachments = buildResearchReplayAttachments(poseFrames);

    expect(attachments).toHaveLength(2);
    expect(attachments[0].pose).toEqual({
      sequence: 0,
      callbackMonotonicTimeMs: 1000,
    });
    expect(attachments[0].pose).not.toHaveProperty("frame");
  });
  it("canonicalizes object keys without changing array order", () => {
    expect(canonicalJson({ z: 1, a: [{ y: 2, b: 3 }] })).toBe(
      '{"a":[{"b":3,"y":2}],"z":1}',
    );
  });

  it("accepts and closes a continuous, replay-confirmed self-capture", async () => {
    const value = await draft();
    expect(validateResearchDraft(value)).toEqual([]);
    const exported = await buildResearchExport(value);
    expect(exported.bundle.schemaVersion).toBe("relay.body-native-episode/2");
    expect(exported.bundle.session.intervals.map(({ kind }) => kind)).toEqual([
      "reference",
      "attempt",
    ]);
    expect(exported.bundle.contentDigests.map(({ path }) => path)).toContain(
      "derived/aids.json",
    );
    expect(exported.sha256Sidecar).toContain(exported.sha256);
  });

  it("rejects noncontinuous scope and correction labels without computed fields", async () => {
    const value = await draft();
    value.annotations[0].observedValue = undefined;
    value.annotations[0].delta = undefined;
    expect(validateResearchDraft(value)).toEqual(
      expect.arrayContaining([
        "An uncertain numeric observation requires finite observed, reference, and difference values.",
      ]),
    );
  });

  it("rejects any stored aid mutation at final export", async () => {
    const value = await draft();
    value.session.derivedAids.payload.frameCount += 1;
    await expect(buildResearchExport(value)).rejects.toThrow(/RAW\+AIDS/);
  });
});
