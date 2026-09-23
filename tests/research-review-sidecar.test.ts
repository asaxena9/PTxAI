import { describe, expect, it } from "vitest";
import type { ResearchAnnotation, ResearchBundle } from "../src/research/capture-bundle";
import { sha256Bytes } from "../src/research/capture-bundle";
import {
  buildBlindedResearchReviewView,
  buildIndependentReviewSidecarExport,
  revealPostSubmissionCollectorComparison,
  verifyResearchReviewArchive,
  type VerifiedResearchReviewArchive,
} from "../src/research/review-sidecar";
import { RECORDED_BLOB_TIMELINE_MAPPING } from "../src/research/capture-replay";

const ARCHIVE_NAME = "P001-S001-E001.relay-episode-v2.zip";
const COLLECTOR_CUE_SENTINEL = "COLLECTOR-CUE-MUST-STAY-HIDDEN";
const PROTOCOL_TARGET_SENTINEL = "PROTOCOL-TARGET-MUST-STAY-HIDDEN";
const textEncoder = new TextEncoder();

function collectorAnnotation(): ResearchAnnotation {
  return {
    annotationId: "collector-annotation-001",
    raterCode: "collector-rater-001",
    raterRole: "collector",
    reviewStatus: "draft-unreviewed",
    movementIdentity: "lateral arm raise",
    assessment: "needs-correction",
    realizedErrorType: "reference-relative peak difference",
    evidenceJoints: ["left-hip", "left-shoulder", "left-elbow"],
    evidenceStartMs: 170,
    evidenceEndMs: 190,
    metric: "left shoulder included angle",
    observedValue: 71.25,
    targetValue: 88.765,
    delta: 17.515,
    unit: "degrees",
    desiredTargetSource: "captured-reference",
    cue: COLLECTOR_CUE_SENTINEL,
    confidence: 0.75,
    createdAt: "2026-08-12T20:00:00.000Z",
  };
}

function independentAnnotation(
  overrides: Partial<ResearchAnnotation> = {},
): ResearchAnnotation {
  return {
    annotationId: "independent-annotation-001",
    raterCode: "independent-rater-002",
    raterRole: "independent-reviewer",
    reviewStatus: "draft-unreviewed",
    movementIdentity: "lateral arm raise",
    assessment: "uncertain",
    realizedErrorType:
      "The attempt appears to stop below the captured reference peak.",
    evidenceJoints: ["left-hip", "left-shoulder", "left-elbow"],
    evidenceStartMs: 168,
    evidenceEndMs: 194,
    evidenceClock: "encoded-replay-approximate-ms",
    desiredTargetSource: "captured-reference",
    confidence: 0.72,
    createdAt: "2026-08-12T20:05:00.000Z",
    ...overrides,
  };
}

async function archiveMembers(): Promise<Record<string, Uint8Array>> {
  const videoBytes = new Uint8Array([11, 22, 33, 44, 55]);
  const videoSha256 = await sha256Bytes(videoBytes);
  const videoFrameTimings = [0, 1, 2, 3, 4, 5, 6, 7, 8].map((sequence) => ({
    sequence,
    callbackMonotonicTimeMs: 100 + sequence * 100,
    mediaTimeMs: 100 + sequence * 100,
    presentedFrames: 10 + sequence,
  }));
  const poseFrames = videoFrameTimings.slice(0, 8).map((timing, sequence) => ({
    sequenceNumber: sequence,
    tMs: sequence * 100,
    sourceTimestampMs: timing.callbackMonotonicTimeMs + 50,
    callbackMonotonicTimeMs: timing.callbackMonotonicTimeMs + 50,
    frameWidth: 640,
    frameHeight: 480,
    poseDetected: true,
    imageJointCount: 0,
    worldJointCount: 0,
    image: {},
    world: {},
    videoTiming: {
      strategy: "nearest-latest-causal-video-frame",
      sequence: timing.sequence,
      callbackMonotonicTimeMs: timing.callbackMonotonicTimeMs,
      mediaTimeMs: timing.mediaTimeMs,
      presentedFrames: timing.presentedFrames,
    },
  }));
  const bundle = {
    schemaVersion: "relay.body-native-episode/2",
    evidenceClass: "FIRST_PARTY_SELF_CAPTURE_MECHANICS_PILOT",
    admissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW",
    createdAt: "2026-08-12T20:01:00.000Z",
    metadata: {
      participantCode: "P001",
      sessionCode: "S001",
      episodeCode: "E001",
      movementFamily: "hidden-before-review",
      cameraView: "front",
      deviceNotes: "test",
      calibrationNotes: "test",
    },
    consent: {},
    protocol: {
      protocol: {
        target: {
          privateTarget: PROTOCOL_TARGET_SENTINEL,
          targetValue: 97.125,
        },
      },
    },
    session: {
      sessionCaptureId: "capture-001",
      startedAt: "2026-08-12T20:00:00.000Z",
      endedAt: "2026-08-12T20:00:01.000Z",
      durationMs: 1000,
      cameraFacing: "user",
      cameraSettings: {},
      clocks: {},
      poseModel: {},
      frameCount: poseFrames.length,
      poseFrames,
      videoFrameTimings,
      intervals: [
        {
          kind: "reference",
          begin: {
            sequence: 2,
            callbackMonotonicTimeMs: 300,
            mediaTimeMs: 300,
            presentedFrames: 12,
          },
          end: {
            sequence: 3,
            callbackMonotonicTimeMs: 400,
            mediaTimeMs: 400,
            presentedFrames: 13,
          },
        },
        {
          kind: "attempt",
          begin: {
            sequence: 5,
            callbackMonotonicTimeMs: 600,
            mediaTimeMs: 600,
            presentedFrames: 15,
          },
          end: {
            sequence: 7,
            callbackMonotonicTimeMs: 800,
            mediaTimeMs: 800,
            presentedFrames: 17,
          },
        },
      ],
      derivedAids: {},
      protocolExecution: {
        targetValue: 97.125,
        cue: COLLECTOR_CUE_SENTINEL,
      },
      video: {
        fileName: "continuous.webm",
        mimeType: "video/webm",
        bytes: videoBytes.byteLength,
        sha256: videoSha256,
        continuousEpisode: true,
      },
      replayAudit: {
        status: "PASS",
        videoAssetSha256: videoSha256,
        integrityAudit: {
          valid: true,
          status: "REPLAY_INTEGRITY_CONFIRMED",
          decodedVideo: {
            valid: true,
            issues: [],
            observed: {
              status: "DECODED_LOCAL_VIDEO",
              durationMs: 1000,
              videoWidth: 640,
              videoHeight: 480,
              seekableStartMs: 0,
              seekableEndMs: 1000,
              observedDecodedFrameMediaTimeMs: 0,
              sourceTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
              inspectedAt: "2026-08-12T20:00:30.000Z",
            },
          },
        },
      },
    },
    annotations: [collectorAnnotation()],
    environment: {},
    lineage: {
      episodeAncestorId: "P001/S001/E001",
      syntheticDescendantsCountAsIndependent: false,
    },
    observationContract: {},
    contentDigests: [
      {
        path: "continuous/continuous.webm",
        bytes: videoBytes.byteLength,
        sha256: videoSha256,
      },
    ],
    claimBoundary: ["not clinical truth"],
  } as unknown as ResearchBundle;
  const episodeBytes = textEncoder.encode(`${JSON.stringify(bundle, null, 2)}\n`);
  const members: Record<string, Uint8Array> = {
    "episode.json": episodeBytes,
    "protocol.json": textEncoder.encode("{}\n"),
    "continuous.pose.json": textEncoder.encode("[]\n"),
    "continuous.video-timing.json": textEncoder.encode("[]\n"),
    "segments.json": textEncoder.encode("[]\n"),
    "derived/aids.json": textEncoder.encode("{}\n"),
    "protocol-execution.json": textEncoder.encode("{}\n"),
    "replay-audit.json": textEncoder.encode("{}\n"),
    "environment.json": textEncoder.encode("{}\n"),
    "annotations/collector.json": textEncoder.encode("[]\n"),
    "CLAIM_BOUNDARY.txt": textEncoder.encode("not clinical truth\n"),
    "continuous/continuous.webm": videoBytes,
  };
  const sums = await Promise.all(
    Object.entries(members)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(async ([path, bytes]) => `${await sha256Bytes(bytes)}  ${path}`),
  );
  members["SHA256SUMS.txt"] = textEncoder.encode(`${sums.join("\n")}\n`);
  return members;
}

async function verifiedArchive(): Promise<VerifiedResearchReviewArchive> {
  return verifyResearchReviewArchive(ARCHIVE_NAME, await archiveMembers());
}

async function rewriteEpisodeAndResign(
  members: Record<string, Uint8Array>,
  mutate: (episode: Record<string, unknown>) => void,
): Promise<Record<string, Uint8Array>> {
  const episode = JSON.parse(
    new TextDecoder().decode(members["episode.json"]),
  ) as Record<string, unknown>;
  mutate(episode);
  members["episode.json"] = textEncoder.encode(
    `${JSON.stringify(episode, null, 2)}\n`,
  );
  const sums = await Promise.all(
    Object.entries(members)
      .filter(([path]) => path !== "SHA256SUMS.txt")
      .sort(([left], [right]) => left.localeCompare(right))
      .map(async ([path, bytes]) => `${await sha256Bytes(bytes)}  ${path}`),
  );
  members["SHA256SUMS.txt"] = textEncoder.encode(`${sums.join("\n")}\n`);
  return members;
}

const replayDeclaration = {
  watchedContinuousVideo: true,
  overlaysVisible: true,
  referenceMarkerInspected: true,
  attemptMarkerInspected: true,
  encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
} as const;

describe("independent research-review archive", () => {
  it("recomputes all member hashes and rejects changed or unbound bytes", async () => {
    const verified = await verifiedArchive();
    expect(verified.videoPath).toBe("continuous/continuous.webm");
    expect(verified.checksums.map(({ path }) => path)).toEqual([
      "CLAIM_BOUNDARY.txt",
      "annotations/collector.json",
      "continuous.pose.json",
      "continuous.video-timing.json",
      "continuous/continuous.webm",
      "derived/aids.json",
      "environment.json",
      "episode.json",
      "protocol-execution.json",
      "protocol.json",
      "replay-audit.json",
      "segments.json",
    ]);

    const changed = await archiveMembers();
    changed["continuous/continuous.webm"][0] = 99;
    await expect(verifyResearchReviewArchive(ARCHIVE_NAME, changed)).rejects.toThrow(
      /SHA-256 mismatch for continuous\/continuous\.webm/,
    );

    const unbound = await archiveMembers();
    unbound["surprise.txt"] = textEncoder.encode("not checksum-bound");
    await expect(verifyResearchReviewArchive(ARCHIVE_NAME, unbound)).rejects.toThrow(
      /not checksum-bound/,
    );

    const boundExtra = await archiveMembers();
    boundExtra["surprise.txt"] = textEncoder.encode("checksum-bound but prohibited");
    await expect(
      verifyResearchReviewArchive(
        ARCHIVE_NAME,
        await rewriteEpisodeAndResign(boundExtra, () => undefined),
      ),
    ).rejects.toThrow(/Unexpected v2 archive member surprise\.txt is prohibited/);
  });

  it("rejects self-checksummed empty or stale causal replay traces", async () => {
    const emptyTrace = await rewriteEpisodeAndResign(
      await archiveMembers(),
      (episode) => {
        const session = episode.session as Record<string, unknown>;
        session.frameCount = 0;
        session.poseFrames = [];
        session.videoFrameTimings = [];
      },
    );
    await expect(
      verifyResearchReviewArchive(ARCHIVE_NAME, emptyTrace),
    ).rejects.toThrow(/timing validation failed|continuous video timing trace is missing/i);

    const staleTrace = await rewriteEpisodeAndResign(
      await archiveMembers(),
      (episode) => {
        const session = episode.session as Record<string, unknown>;
        const frames = session.poseFrames as Array<Record<string, unknown>>;
        frames[3].videoTiming = frames[0].videoTiming;
      },
    );
    await expect(
      verifyResearchReviewArchive(ARCHIVE_NAME, staleTrace),
    ).rejects.toThrow(/stale or non-causal video attachment/);
  });

  it("rejects a signed top-level replay PASS without retained decode evidence", async () => {
    const passOnly = await rewriteEpisodeAndResign(
      await archiveMembers(),
      (episode) => {
        const session = episode.session as Record<string, unknown>;
        session.replayAudit = {
          status: "PASS",
          videoAssetSha256: (
            session.video as Record<string, unknown>
          ).sha256,
          integrityAudit: {
            valid: true,
            status: "REPLAY_INTEGRITY_CONFIRMED",
          },
        };
      },
    );
    await expect(
      verifyResearchReviewArchive(ARCHIVE_NAME, passOnly),
    ).rejects.toThrow(/decoded-video validation|Browser-observed decoded-video/);
  });

  it("exposes replay evidence while withholding prior judgments before commit", async () => {
    const view = buildBlindedResearchReviewView(await verifiedArchive());
    const serialized = JSON.stringify(view);

    expect(view).toMatchObject({
      schemaVersion: "relay.body-native-episode/2",
      admissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW",
      episodeId: "E001",
      episodeAncestorId: "P001/S001/E001",
      session: {
        video: { path: "continuous/continuous.webm" },
      },
    });
    expect(serialized).not.toContain(COLLECTOR_CUE_SENTINEL);
    expect(serialized).not.toContain(PROTOCOL_TARGET_SENTINEL);
    expect(serialized).not.toContain("88.765");
    expect(serialized).not.toContain('"annotations"');
    expect(serialized).not.toContain('"protocolExecution"');
  });

  it("builds an immutable, hash-bound append-only sidecar with exact ancestry", async () => {
    const verified = await verifiedArchive();
    const submitted = independentAnnotation();
    const result = await buildIndependentReviewSidecarExport(verified, submitted, {
      blinded: true,
      independentlyProduced: true,
      replay: replayDeclaration,
    });

    expect(result.sidecar).toMatchObject({
      schemaVersion: "relay.body-native-independent-review-sidecar/1",
      admissionEffect: "NONE_REMAINS_QUARANTINED",
      blindingStatus:
        "PROCEDURAL_SELF_ATTESTED_UI_BLINDING_NOT_SEALED_PACKAGE",
      appendOnly: true,
      sourceEpisode: {
        episodeJsonSha256: verified.episodeJsonSha256,
        episodeId: "E001",
        episodeAncestorId: "P001/S001/E001",
      },
      derivedReviewState: {
        state: "needs-second-review",
        clinicalTruthStatus: "not-established-by-review",
      },
      replayDeclaration: {
        watchedContinuousVideo: true,
        overlaysVisible: true,
        referenceMarkerInspected: true,
        attemptMarkerInspected: true,
        encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
        episodeJsonSha256: verified.episodeJsonSha256,
      },
    });
    expect(result.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.sha256Sidecar).toContain(result.fileName);
    expect(Object.isFrozen(result.sidecar.reviewEntry.annotation)).toBe(true);
    submitted.realizedErrorType = "mutation after commitment";
    expect(result.sidecar.reviewEntry.annotation.realizedErrorType).not.toBe(
      "mutation after commitment",
    );

    const revealed = revealPostSubmissionCollectorComparison(
      verified,
      result.sidecar,
    );
    expect(revealed.disagreements.length).toBeGreaterThan(0);
    expect(revealed.interpretation).toContain("not-consensus");
  });

  it("rejects non-independent declarations, duplicate raters, and mutated lineage", async () => {
    const verified = await verifiedArchive();
    await expect(
      buildIndependentReviewSidecarExport(verified, independentAnnotation(), {
        blinded: false,
        independentlyProduced: true,
        replay: replayDeclaration,
      }),
    ).rejects.toThrow(/procedural UI-blinding/);
    await expect(
      buildIndependentReviewSidecarExport(
        verified,
        independentAnnotation({ raterCode: "COLLECTOR-RATER-001" }),
        {
          blinded: true,
          independentlyProduced: true,
          replay: replayDeclaration,
        },
      ),
    ).rejects.toThrow(/distinct from every collector/);

    const built = await buildIndependentReviewSidecarExport(
      verified,
      independentAnnotation(),
      {
        blinded: true,
        independentlyProduced: true,
        replay: replayDeclaration,
      },
    );
    const mutated = {
      ...built.sidecar,
      sourceEpisode: {
        ...built.sidecar.sourceEpisode,
        episodeAncestorId: "P999/S999/E999",
      },
    };
    expect(() =>
      revealPostSubmissionCollectorComparison(verified, mutated),
    ).toThrow(/different episode ancestor/);
  });

  it("accepts a qualitative uncertain review and rejects bypassed numeric claims", async () => {
    const verified = await verifiedArchive();
    const qualitative = independentAnnotation({
      assessment: "uncertain",
      realizedErrorType:
        "The attempt appears to stop below the captured reference peak.",
      metric: undefined,
      observedValue: undefined,
      targetValue: undefined,
      delta: undefined,
      unit: undefined,
      cue: undefined,
    });
    const built = await buildIndependentReviewSidecarExport(
      verified,
      qualitative,
      {
        blinded: true,
        independentlyProduced: true,
        replay: replayDeclaration,
      },
    );
    expect(built.sidecar.reviewEntry.annotation).toMatchObject({
      assessment: "uncertain",
      realizedErrorType:
        "The attempt appears to stop below the captured reference peak.",
    });
    expect(built.sidecar.reviewEntry.annotation.metric).toBeUndefined();

    await expect(
      buildIndependentReviewSidecarExport(
        verified,
        independentAnnotation({
          assessment: "needs-correction",
          metric: "left shoulder included angle",
          observedValue: 70,
          targetValue: 89,
          delta: 19,
          unit: "degrees",
          cue: "Raise farther.",
        }),
        {
          blinded: true,
          independentlyProduced: true,
          replay: replayDeclaration,
        },
      ),
    ).rejects.toThrow(/accepts only a qualitative uncertain observation/);

    await expect(
      buildIndependentReviewSidecarExport(
        verified,
        independentAnnotation({
          assessment: "acceptable",
          realizedErrorType: undefined,
          evidenceJoints: [],
          evidenceStartMs: undefined,
          evidenceEndMs: undefined,
        }),
        {
          blinded: true,
          independentlyProduced: true,
          replay: replayDeclaration,
        },
      ),
    ).rejects.toThrow(/accepts only a qualitative uncertain observation/);

    await expect(
      buildIndependentReviewSidecarExport(
        verified,
        qualitative,
        {
          blinded: true,
          independentlyProduced: true,
          replay: {
            ...replayDeclaration,
            overlaysVisible: false,
          },
        },
      ),
    ).rejects.toThrow(/visible overlays/);
  });
});
