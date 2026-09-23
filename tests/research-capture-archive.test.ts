import { unzipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { buildResearchArchive } from "../src/research/capture-archive";
import type { ResearchExport } from "../src/research/capture-bundle";
import {
  sha256Bytes,
  sha256Text,
  type ResearchPoseFrame,
} from "../src/research/capture-bundle";
import {
  auditLocalReplayIntegrity,
  RECORDED_BLOB_TIMELINE_MAPPING,
  REPLAY_INTEGRITY_EVIDENCE_SCOPE,
} from "../src/research/capture-replay";

describe("research episode archive", () => {
  it("closes the episode, continuous video, and member checksums in one zip", async () => {
    const video = new Blob([new Uint8Array([1, 2, 3, 4])], {
      type: "video/webm",
    });
    const videoSha = await sha256Bytes(
      new Uint8Array(await video.arrayBuffer()),
    );
    const videoFrames = Array.from({ length: 5 }, (_, sequence) => ({
      sequence,
      callbackMonotonicTimeMs: 1000 + sequence * 100,
      mediaTimeMs: sequence * 100,
      presentedFrames: sequence + 1,
    }));
    const poseFrames = videoFrames.map((videoTiming) => ({
      sequenceNumber: videoTiming.sequence,
      tMs: videoTiming.mediaTimeMs,
      sourceTimestampMs: videoTiming.callbackMonotonicTimeMs,
      callbackMonotonicTimeMs: videoTiming.callbackMonotonicTimeMs,
      frameWidth: 100,
      frameHeight: 100,
      poseDetected: true,
      imageJointCount: 0,
      worldJointCount: 0,
      image: {},
      world: {},
      videoTiming: {
        strategy: "nearest-latest-causal-video-frame" as const,
        ...videoTiming,
      },
    })) satisfies ResearchPoseFrame[];
    const intervals = [
      { kind: "reference" as const, begin: videoFrames[0], end: videoFrames[1] },
      { kind: "attempt" as const, begin: videoFrames[3], end: videoFrames[4] },
    ];
    const integrityAudit = await auditLocalReplayIntegrity({
      declaredVideo: {
        fileName: "continuous.webm",
        mimeType: "video/webm",
        bytes: 4,
        sha256: videoSha,
      },
      observedVideo: { fileName: "continuous.webm", blob: video },
      decodedVideo: {
        status: "DECODED_LOCAL_VIDEO",
        durationMs: 400,
        videoWidth: 100,
        videoHeight: 100,
        seekableStartMs: 0,
        seekableEndMs: 400,
        observedDecodedFrameMediaTimeMs: 0,
        sourceTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
        inspectedAt: "2026-08-12T00:00:00.000Z",
      },
      videoFrames,
      poseAttachments: poseFrames.map((pose) => ({
        strategy: "nearest-latest-causal-video-frame" as const,
        pose: {
          sequence: pose.sequenceNumber,
          callbackMonotonicTimeMs: pose.callbackMonotonicTimeMs,
        },
        videoFrame: pose.videoTiming!,
      })),
      intervals,
      confirmation: {
        reviewerCode: "R1",
        watchedContinuousVideo: true,
        overlaysVisible: true,
        referenceIntervalConfirmed: true,
        attemptIntervalConfirmed: true,
        declaredAt: "2026-08-12T00:00:00.000Z",
        evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
      },
    });
    expect(integrityAudit.valid).toBe(true);
    const humanConfirmation = {
      reviewerCode: "R1",
      watchedContinuousVideo: true as const,
      overlaysVisible: true as const,
      referenceIntervalConfirmed: true as const,
      attemptIntervalConfirmed: true as const,
      declaredAt: "2026-08-12T00:00:00.000Z",
      evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
    };
    const bundle = {
      metadata: {
        participantCode: "P1",
        sessionCode: "S1",
        episodeCode: "E1",
      },
      protocol: { protocol: { id: "protocol" } },
      session: {
        video: {
          fileName: "continuous.webm",
          mimeType: "video/webm",
          bytes: 4,
          sha256: videoSha,
        },
        poseFrames,
        videoFrameTimings: videoFrames,
        intervals,
        derivedAids: {},
        protocolExecution: {},
        replayAudit: {
          status: "PASS",
          videoAssetSha256: videoSha,
          poseFrameCount: poseFrames.length,
          videoTimingCount: videoFrames.length,
          intervalBoundaryCoverage: true,
          causalOverlaySelection: true,
          encodedReplayTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
          encodedReplayTimelineCalibrated: false,
          renderedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
          humanConfirmation,
          issues: [],
          integrityAudit,
        },
      },
      environment: {},
      annotations: [],
      claimBoundary: ["not clinical truth"],
    } as unknown as ResearchExport["bundle"];
    bundle.contentDigests = await Promise.all([
      ["protocol.json", `${JSON.stringify(bundle.protocol.protocol, null, 2)}\n`],
      ["continuous.pose.json", `${JSON.stringify(bundle.session.poseFrames, null, 2)}\n`],
      ["continuous.video-timing.json", `${JSON.stringify(bundle.session.videoFrameTimings, null, 2)}\n`],
      ["segments.json", `${JSON.stringify(bundle.session.intervals, null, 2)}\n`],
      ["derived/aids.json", `${JSON.stringify(bundle.session.derivedAids, null, 2)}\n`],
      ["protocol-execution.json", `${JSON.stringify(bundle.session.protocolExecution, null, 2)}\n`],
      ["replay-audit.json", `${JSON.stringify(bundle.session.replayAudit, null, 2)}\n`],
      ["environment.json", `${JSON.stringify(bundle.environment, null, 2)}\n`],
    ].map(async ([path, content]) => ({
      path,
      bytes: new TextEncoder().encode(content).byteLength,
      sha256: await sha256Text(content),
    })));
    bundle.contentDigests.push({
      path: "continuous/continuous.webm",
      bytes: 4,
      sha256: videoSha,
    });
    const json = `${JSON.stringify(bundle)}\n`;
    const researchExport = {
      bundle,
      json,
      sha256: await sha256Text(json),
      sha256Sidecar: "",
    } satisfies ResearchExport;

    const result = await buildResearchArchive(researchExport, video);
    const files = unzipSync(new Uint8Array(await result.blob.arrayBuffer()));

    expect(result.fileName).toBe("P1-S1-E1.relay-episode-v2.zip");
    expect(result.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(Object.keys(files)).toEqual(
      expect.arrayContaining([
        "episode.json",
        "continuous/continuous.webm",
        "SHA256SUMS.txt",
        "protocol.json",
        "replay-audit.json",
      ]),
    );
    expect(Array.from(files["continuous/continuous.webm"])).toEqual([1, 2, 3, 4]);
    expect(new TextDecoder().decode(files["SHA256SUMS.txt"])).toContain(
      "continuous/continuous.webm",
    );
  });

  it("rejects a mismatched video", async () => {
    const video = new Blob([new Uint8Array([1])], { type: "video/webm" });
    const bundle = {
      metadata: { participantCode: "P", sessionCode: "S", episodeCode: "E" },
      protocol: { protocol: {} },
      session: {
        video: {
          fileName: "continuous.webm",
          mimeType: "video/webm",
          bytes: 2,
          sha256: "0".repeat(64),
        },
        poseFrames: [],
        videoFrameTimings: [],
        intervals: [],
        derivedAids: {},
        protocolExecution: {},
        replayAudit: {},
      },
      environment: {},
      annotations: [],
      claimBoundary: [],
    } as unknown as ResearchExport["bundle"];
    const json = `${JSON.stringify(bundle)}\n`;
    await expect(
      buildResearchArchive(
        {
          bundle,
          json,
          sha256: await sha256Text(json),
          sha256Sidecar: "",
        },
        video,
      ),
    ).rejects.toThrow(/video bytes/);
  });
});
