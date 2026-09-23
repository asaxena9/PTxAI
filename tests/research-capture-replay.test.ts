import { describe, expect, it } from "vitest";
import {
  RECORDED_BLOB_TIMELINE_MAPPING,
  REPLAY_INTEGRITY_CLAIM_BOUNDARY,
  REPLAY_INTEGRITY_EVIDENCE_SCOPE,
  auditLocalReplayIntegrity,
  auditReplayIntervalBoundaryCoverage,
  selectLatestCausalPoseForReplayTime,
  sha256Blob,
  validateHumanReplayConfirmation,
  validateRecordedContinuousVideo,
  validateReplayPoseAttachments,
  validateReplayVideoTimingTrace,
  type DeclaredContinuousVideoAsset,
  type HumanReplayConfirmation,
  type ReplayCaptureInterval,
  type ReplayPoseAttachment,
  type ReplayPoseTimingSample,
  type ReplayVideoTimingSample,
} from "../src/research/capture-replay";

const VIDEO_SHA256 =
  "d3290da62ef8ce25e82444898f60719b966fa2a5cd19f89f676920930267cbae";

function videoBlob(): Blob {
  return new Blob(["continuous-video"], { type: "video/webm" });
}

function declaration(
  overrides: Partial<DeclaredContinuousVideoAsset> = {},
): DeclaredContinuousVideoAsset {
  return {
    fileName: "episode-continuous.webm",
    mimeType: "video/webm",
    bytes: 16,
    sha256: VIDEO_SHA256,
    ...overrides,
  };
}

function videoFrame(
  sequence: number,
  callbackMonotonicTimeMs = 100 + sequence * 10,
  mediaTimeMs = sequence * 10,
  presentedFrames: number | undefined = sequence + 40,
): ReplayVideoTimingSample {
  return {
    sequence,
    callbackMonotonicTimeMs,
    mediaTimeMs,
    ...(presentedFrames === undefined ? {} : { presentedFrames }),
  };
}

function videoFrames(): ReplayVideoTimingSample[] {
  return Array.from({ length: 8 }, (_, sequence) => videoFrame(sequence));
}

interface TestPose extends ReplayPoseTimingSample {
  readonly poseLabel: string;
}

function poseAttachments(
  frames = videoFrames(),
): ReplayPoseAttachment<TestPose>[] {
  return [0, 2, 4, 6].map((videoSequence, poseSequence) => ({
    strategy: "nearest-latest-causal-video-frame" as const,
    pose: {
      sequence: poseSequence,
      callbackMonotonicTimeMs:
        frames[videoSequence].callbackMonotonicTimeMs + 5,
      poseLabel: `pose-${poseSequence}`,
    },
    videoFrame: frames[videoSequence],
  }));
}

function intervals(frames = videoFrames()): ReplayCaptureInterval[] {
  return [
    { kind: "reference", begin: frames[1], end: frames[3] },
    { kind: "attempt", begin: frames[4], end: frames[7] },
  ];
}

function confirmation(
  overrides: Partial<HumanReplayConfirmation> = {},
): HumanReplayConfirmation {
  return {
    reviewerCode: "R-BLIND-02",
    watchedContinuousVideo: true,
    overlaysVisible: true,
    referenceIntervalConfirmed: true,
    attemptIntervalConfirmed: true,
    declaredAt: "2026-08-12T20:15:30.123Z",
    evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
    ...overrides,
  };
}

describe("continuous-video asset integrity", () => {
  it("re-reads the Blob and binds filename, MIME type, bytes, and SHA-256", async () => {
    const blob = videoBlob();
    expect(await sha256Blob(blob)).toBe(VIDEO_SHA256);

    const result = await validateRecordedContinuousVideo(declaration(), {
      fileName: "episode-continuous.webm",
      blob,
    });

    expect(result).toEqual({
      valid: true,
      issues: [],
      observed: {
        fileName: "episode-continuous.webm",
        mimeType: "video/webm",
        bytes: 16,
        sha256: VIDEO_SHA256,
      },
    });
  });

  it("fails closed on every declared-to-observed identity mismatch", async () => {
    const result = await validateRecordedContinuousVideo(
      declaration({
        fileName: "other.webm",
        mimeType: "video/mp4",
        bytes: 17,
        sha256: "0".repeat(64),
      }),
      { fileName: "episode-continuous.webm", blob: videoBlob() },
    );

    expect(result.valid).toBe(false);
    expect(result.observed?.sha256).toBe(VIDEO_SHA256);
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "file-name-mismatch" }),
        expect.objectContaining({ code: "mime-type-mismatch" }),
        expect.objectContaining({ code: "byte-length-mismatch" }),
        expect.objectContaining({ code: "sha256-mismatch" }),
      ]),
    );
  });

  it("rejects missing, malformed, empty, and inconsistently readable assets", async () => {
    const missing = await validateRecordedContinuousVideo(null, null);
    expect(missing.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "missing-video-declaration" }),
        expect.objectContaining({ code: "missing-video-blob" }),
      ]),
    );

    const malformed = await validateRecordedContinuousVideo(
      {
        fileName: "../episode.webm",
        mimeType: "application/octet-stream",
        bytes: Number.NaN,
        sha256: "not-a-digest",
      },
      { fileName: "episode-continuous.webm", blob: new Blob([]) },
    );
    expect(malformed.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "invalid-file-name" }),
        expect.objectContaining({ code: "invalid-mime-type" }),
        expect.objectContaining({ code: "invalid-byte-length" }),
        expect.objectContaining({ code: "invalid-sha256" }),
      ]),
    );

    const inconsistentBlob = {
      size: 16,
      type: "video/webm",
      arrayBuffer: async () => new Uint8Array([1]).buffer,
    } as Blob;
    const inconsistent = await validateRecordedContinuousVideo(declaration(), {
      fileName: "episode-continuous.webm",
      blob: inconsistentBlob,
    });
    expect(inconsistent.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "blob-read-size-mismatch" }),
        expect.objectContaining({ code: "sha256-mismatch" }),
      ]),
    );
  });
});

describe("causal pose selection during replay", () => {
  it("chooses the latest attached pose at or before observed media time", () => {
    const attachments = poseAttachments();

    expect(selectLatestCausalPoseForReplayTime(0, attachments, videoFrames())?.pose.poseLabel).toBe(
      "pose-0",
    );
    expect(
      selectLatestCausalPoseForReplayTime(39.999, attachments, videoFrames())?.pose.poseLabel,
    ).toBe("pose-1");
    expect(
      selectLatestCausalPoseForReplayTime(40, attachments, videoFrames())?.pose.poseLabel,
    ).toBe("pose-2");
    expect(selectLatestCausalPoseForReplayTime(-1, attachments, videoFrames())).toBeNull();
    expect(
      selectLatestCausalPoseForReplayTime(Number.POSITIVE_INFINITY, attachments, videoFrames()),
    ).toBeNull();
  });

  it("is prefix-stable and never selects an attached future media frame", () => {
    const attachments = poseAttachments();
    const beforeFutureAttachments = selectLatestCausalPoseForReplayTime(
      25,
      attachments.slice(0, 2),
      videoFrames(),
    );
    const afterFutureAttachments = selectLatestCausalPoseForReplayTime(
      25,
      attachments,
      videoFrames(),
    );

    expect(afterFutureAttachments).toEqual(beforeFutureAttachments);
    expect(afterFutureAttachments?.videoFrame.mediaTimeMs).toBe(20);
    expect(
      afterFutureAttachments!.videoFrame.mediaTimeMs <= 25,
    ).toBe(true);
  });

  it("fails closed on missing, nonfinite, gapped, backward, or noncausal traces", () => {
    expect(selectLatestCausalPoseForReplayTime(10, [], videoFrames())).toBeNull();

    const gapped = poseAttachments();
    gapped[1] = {
      ...gapped[1],
      pose: { ...gapped[1].pose, sequence: 3 },
    };
    expect(validateReplayPoseAttachments(gapped)).toMatchObject({
      valid: false,
      issues: expect.arrayContaining([
        expect.objectContaining({ code: "pose-sequence-gap" }),
      ]),
    });
    expect(selectLatestCausalPoseForReplayTime(30, gapped, videoFrames())).toBeNull();

    const noncausal = poseAttachments();
    noncausal[0] = {
      ...noncausal[0],
      pose: { ...noncausal[0].pose, callbackMonotonicTimeMs: 99 },
    };
    expect(validateReplayPoseAttachments(noncausal).issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "noncausal-pose-attachment" }),
      ]),
    );

    const backward = poseAttachments();
    backward[2] = { ...backward[2], videoFrame: videoFrame(1) };
    expect(selectLatestCausalPoseForReplayTime(40, backward, videoFrames())).toBeNull();

    const nonfinite = poseAttachments();
    nonfinite[0] = {
      ...nonfinite[0],
      videoFrame: { ...nonfinite[0].videoFrame, mediaTimeMs: Number.NaN },
    };
    expect(selectLatestCausalPoseForReplayTime(0, nonfinite, videoFrames())).toBeNull();
  });

  it("rejects incomplete or corrupt replay video timing traces", () => {
    expect(validateReplayVideoTimingTrace([])).toMatchObject({
      valid: false,
      issues: [expect.objectContaining({ code: "missing-video-timing" })],
    });
    expect(
      validateReplayVideoTimingTrace([videoFrame(0), videoFrame(2)]).issues,
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "video-sequence-gap" }),
      ]),
    );
    expect(
      validateReplayVideoTimingTrace([
        videoFrame(0, 100, 0, 40),
        {
          sequence: 1,
          callbackMonotonicTimeMs: 110,
          mediaTimeMs: 10,
        },
      ]).issues,
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "incomplete-presented-frame-timing",
        }),
      ]),
    );
    expect(
      validateReplayVideoTimingTrace([
        videoFrame(0),
        videoFrame(1, 99, Number.NaN),
      ]).valid,
    ).toBe(false);
  });
});

describe("interval-boundary replay coverage", () => {
  it("binds every reference/attempt edge to an exact video frame and past pose", () => {
    const frames = videoFrames();
    const attachments = poseAttachments(frames);
    const intervalTrace = intervals(frames);
    const snapshot = structuredClone({ frames, attachments, intervalTrace });

    const result = auditReplayIntervalBoundaryCoverage(
      intervalTrace,
      frames,
      attachments,
    );

    expect(result.valid).toBe(true);
    expect(result.expectedBoundaryCount).toBe(4);
    expect(result.coveredBoundaryCount).toBe(4);
    expect(result.coverage.map((entry) => entry.selectedPose.pose.poseLabel)).toEqual(
      ["pose-0", "pose-1", "pose-1", "pose-3"],
    );
    expect(result.coverage.map((entry) => entry.poseMediaLagMs)).toEqual([
      10, 10, 20, 10,
    ]);
    expect(
      result.coverage.every(
        (entry) =>
          entry.selectedPose.videoFrame.mediaTimeMs <= entry.replayMediaTimeMs,
      ),
    ).toBe(true);
    expect(
      result.coverage.every(
        (entry) =>
          entry.selectedPose.pose.callbackMonotonicTimeMs <=
          entry.videoFrame.callbackMonotonicTimeMs,
      ),
    ).toBe(true);
    expect({ frames, attachments, intervalTrace }).toEqual(snapshot);
  });

  it("never admits a pose callback that arrived after an exact interval edge", () => {
    const frames = videoFrames();
    const attachments = poseAttachments(frames);
    const result = auditReplayIntervalBoundaryCoverage(
      [
        { kind: "reference", begin: frames[1], end: frames[2] },
        { kind: "attempt", begin: frames[3], end: frames[6] },
      ],
      frames,
      attachments,
    );

    expect(result.valid).toBe(true);
    const referenceEnd = result.coverage.find(
      (entry) =>
        entry.intervalKind === "reference" && entry.boundary === "end",
    );
    expect(referenceEnd?.selectedPose.pose.poseLabel).toBe("pose-0");
    expect(
      referenceEnd?.selectedPose.pose.callbackMonotonicTimeMs,
    ).toBeLessThanOrEqual(frames[2].callbackMonotonicTimeMs);
  });

  it("fails on forged/nonfinite boundaries, overlap, and absent boundary pose", () => {
    const frames = videoFrames();
    const forgedIntervals: ReplayCaptureInterval[] = [
      {
        kind: "reference",
        begin: { ...frames[0], mediaTimeMs: Number.NaN },
        end: { ...frames[2], mediaTimeMs: 20.5 },
      },
      { kind: "attempt", begin: frames[1], end: frames[6] },
    ];
    const result = auditReplayIntervalBoundaryCoverage(
      forgedIntervals,
      frames,
      poseAttachments(frames).slice(1),
    );

    expect(result.valid).toBe(false);
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "invalid-video-timing" }),
        expect.objectContaining({ code: "interval-boundary-not-in-trace" }),
      ]),
    );

    const overlapping = auditReplayIntervalBoundaryCoverage(
      [
        { kind: "reference", begin: frames[0], end: frames[4] },
        { kind: "attempt", begin: frames[3], end: frames[6] },
      ],
      frames,
      poseAttachments(frames),
    );
    expect(overlapping.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "interval-overlap" }),
      ]),
    );

    const noOpeningPose = poseAttachments(frames).map((attachment, index) => ({
      ...attachment,
      pose: { ...attachment.pose, sequence: index },
    })).slice(1).map((attachment, index) => ({
      ...attachment,
      pose: { ...attachment.pose, sequence: index },
    }));
    const noOpeningCoverage = auditReplayIntervalBoundaryCoverage(
      intervals(frames),
      frames,
      noOpeningPose,
    );
    expect(noOpeningCoverage.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "missing-boundary-pose" }),
      ]),
    );
  });

  it("rejects pose attachments to a different video trace", () => {
    const frames = videoFrames();
    const attachments = poseAttachments(frames);
    attachments[3] = {
      ...attachments[3],
      videoFrame: videoFrame(6, 165, 60, 46),
    };
    const result = auditReplayIntervalBoundaryCoverage(
      intervals(frames),
      frames,
      attachments,
    );

    expect(result.valid).toBe(false);
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "pose-video-frame-not-in-trace" }),
      ]),
    );
  });
});

describe("human replay declaration and combined audit", () => {
  it("requires a scoped, timestamped human confirmation of the full replay", () => {
    expect(validateHumanReplayConfirmation(confirmation())).toEqual({
      valid: true,
      issues: [],
    });

    const invalid = validateHumanReplayConfirmation({
      ...confirmation(),
      reviewerCode: " ",
      watchedContinuousVideo: false,
      overlaysVisible: false,
      referenceIntervalConfirmed: false,
      attemptIntervalConfirmed: false,
      declaredAt: "2026-02-30T20:15:30Z",
      evidenceScope: "movement-is-true",
    } as unknown as HumanReplayConfirmation);
    expect(invalid.valid).toBe(false);
    expect(invalid.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "missing-reviewer-code" }),
        expect.objectContaining({ code: "continuous-video-not-confirmed" }),
        expect.objectContaining({ code: "overlays-not-confirmed" }),
        expect.objectContaining({ code: "reference-interval-not-confirmed" }),
        expect.objectContaining({ code: "attempt-interval-not-confirmed" }),
        expect.objectContaining({ code: "invalid-declared-timestamp" }),
        expect.objectContaining({ code: "invalid-evidence-scope" }),
      ]),
    );
    expect(validateHumanReplayConfirmation(null).issues).toEqual([
      expect.objectContaining({ code: "missing-human-confirmation" }),
    ]);
  });

  it("confirms a complete local integrity audit without promoting truth", async () => {
    const frames = videoFrames();
    const result = await auditLocalReplayIntegrity({
      declaredVideo: declaration(),
      observedVideo: {
        fileName: "episode-continuous.webm",
        blob: videoBlob(),
      },
      decodedVideo: {
        status: "DECODED_LOCAL_VIDEO",
        durationMs: 70,
        videoWidth: 100,
        videoHeight: 100,
        seekableStartMs: 0,
        seekableEndMs: 70,
        observedDecodedFrameMediaTimeMs: 0,
        sourceTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
        inspectedAt: "2026-08-12T20:15:30.123Z",
      },
      videoFrames: frames,
      poseAttachments: poseAttachments(frames),
      intervals: intervals(frames),
      confirmation: confirmation(),
    });

    expect(result).toMatchObject({
      valid: true,
      status: "REPLAY_INTEGRITY_CONFIRMED",
      evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
      claimBoundary: REPLAY_INTEGRITY_CLAIM_BOUNDARY,
      issues: [],
      videoAsset: { valid: true },
      boundaryCoverage: { valid: true, coveredBoundaryCount: 4 },
      humanConfirmation: { valid: true },
    });
    expect(result.claimBoundary).toContain("does not establish movement");
    expect(result.claimBoundary).toContain("clinical");
  });

  it("rejects the whole audit when an asset or human gate fails", async () => {
    const frames = videoFrames();
    const result = await auditLocalReplayIntegrity({
      declaredVideo: declaration({ sha256: "f".repeat(64) }),
      observedVideo: {
        fileName: "episode-continuous.webm",
        blob: videoBlob(),
      },
      decodedVideo: {
        status: "DECODED_LOCAL_VIDEO",
        durationMs: 70,
        videoWidth: 100,
        videoHeight: 100,
        seekableStartMs: 0,
        seekableEndMs: 70,
        observedDecodedFrameMediaTimeMs: 0,
        sourceTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
        inspectedAt: "2026-08-12T20:15:30.123Z",
      },
      videoFrames: frames,
      poseAttachments: poseAttachments(frames),
      intervals: intervals(frames),
      confirmation: confirmation({ overlaysVisible: false }),
    });

    expect(result.valid).toBe(false);
    expect(result.status).toBe("REPLAY_INTEGRITY_REJECTED");
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "sha256-mismatch" }),
        expect.objectContaining({ code: "overlays-not-confirmed" }),
      ]),
    );
  });

  it("fails closed when every required replay record is absent", async () => {
    const result = await auditLocalReplayIntegrity({
      declaredVideo: null,
      observedVideo: null,
      decodedVideo: null,
      videoFrames: null,
      poseAttachments: null,
      intervals: null,
      confirmation: null,
    });

    expect(result.valid).toBe(false);
    expect(result.status).toBe("REPLAY_INTEGRITY_REJECTED");
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "missing-video-declaration" }),
        expect.objectContaining({ code: "missing-video-blob" }),
        expect.objectContaining({ code: "missing-video-timing" }),
        expect.objectContaining({ code: "missing-pose-attachments" }),
        expect.objectContaining({ code: "missing-intervals" }),
        expect.objectContaining({ code: "missing-human-confirmation" }),
      ]),
    );
  });
});
