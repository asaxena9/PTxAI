import { zipSync } from "fflate";
import { describe, expect, it } from "vitest";
import {
  preflightAdmissionPackageV1,
  type AdmissionPackageV1Input,
  type RawChecksummedArtifact,
} from "../src/research/admission-package";
import {
  canonicalJson,
  sha256Bytes,
} from "../src/research/capture-bundle";
import { fitMediaClockCalibration } from "../src/research/media-clock-calibration";

const utf8 = (value: string): Uint8Array => new TextEncoder().encode(value);

async function checksummedArtifact(
  fileName: string,
  bytes: Uint8Array,
): Promise<RawChecksummedArtifact> {
  const sha256 = await sha256Bytes(bytes);
  return {
    fileName,
    bytes: Uint8Array.from(bytes),
    checksumSidecar: {
      fileName: `${fileName}.sha256`,
      bytes: utf8(`${sha256}  ${fileName}\n`),
    },
  };
}

interface ArchiveFixture {
  readonly artifact: RawChecksummedArtifact;
  readonly episodeJsonSha256: string;
  readonly videoSha256: string;
  readonly videoTimingSha256: string;
}

async function archiveFixture(
  includeUntrustedIdentityExtensions = false,
): Promise<ArchiveFixture> {
  const videoBytes = Uint8Array.from([0x1a, 0x45, 0xdf, 0xa3, 1, 2, 3, 4]);
  const videoSha256 = await sha256Bytes(videoBytes);
  const videoTimingBytes = utf8("[]\n");
  const videoTimingSha256 = await sha256Bytes(videoTimingBytes);
  const episode = {
    schemaVersion: "relay.body-native-episode/2",
    admissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW",
    metadata: { episodeCode: "E001" },
    lineage: {
      episodeAncestorId: "P001/S001/E001",
      syntheticDescendantsCountAsIndependent: false,
    },
    session: {
      poseModel: {
        name: "MediaPipe Pose Landmarker Full",
        asset: "/models/pose_landmarker_full.task",
        assetBytes: 9_398_198,
        assetSha256:
          "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
        inferenceCadenceMs: 66,
        requestedDelegate: "GPU",
        realizedDelegate: "unobserved-gpu-with-cpu-fallback",
        ...(includeUntrustedIdentityExtensions
          ? { runtimeLibrary: "@mediapipe/tasks-vision", runtimeVersion: "1.0.0" }
          : {}),
      },
      video: {
        fileName: "continuous.webm",
        mimeType: "video/webm",
        bytes: videoBytes.byteLength,
        sha256: videoSha256,
        continuousEpisode: true,
      },
      replayAudit: {
        encodedReplayTimelineCalibrated: false,
        renderedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
      },
    },
    environment: {
      collectorBuildId: "relay.body-native-research-collector/2.0.0",
      ...(includeUntrustedIdentityExtensions
        ? { appBuildSha256: "a".repeat(64) }
        : {}),
    },
  };
  const episodeBytes = utf8(canonicalJson(episode));
  const episodeJsonSha256 = await sha256Bytes(episodeBytes);
  const members: Record<string, Uint8Array> = {
    "episode.json": episodeBytes,
    "continuous.video-timing.json": videoTimingBytes,
    "continuous/continuous.webm": videoBytes,
  };
  const manifestLines = await Promise.all(
    Object.entries(members)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(async ([path, bytes]) => `${await sha256Bytes(bytes)}  ${path}`),
  );
  members["SHA256SUMS.txt"] = utf8(`${manifestLines.join("\n")}\n`);
  const zipMembers = Object.fromEntries(
    await Promise.all(
      Object.entries(members).map(async ([path, bytes]) => [
        path,
        new Uint8Array(
          await new Blob([Uint8Array.from(bytes)]).arrayBuffer(),
        ),
      ] as const),
    ),
  );
  const archiveBytes = zipSync(zipMembers, { level: 0 });
  return {
    artifact: await checksummedArtifact(
      "P001-S001-E001.relay-episode-v2.zip",
      archiveBytes,
    ),
    episodeJsonSha256,
    videoSha256,
    videoTimingSha256,
  };
}

async function reviewArtifact(
  episodeJsonSha256: string,
  raterCode: string,
  annotationId: string,
): Promise<RawChecksummedArtifact> {
  const review = {
    schemaVersion: "relay.body-native-independent-review-sidecar/1",
    evidenceClass: "HUMAN_INDEPENDENT_OBSERVABLE_MOTION_REVIEW",
    admissionEffect: "NONE_REMAINS_QUARANTINED",
    appendOnly: true,
    sourceEpisode: {
      episodeJsonPath: "episode.json",
      episodeJsonSha256,
      episodeId: "E001",
      episodeAncestorId: "P001/S001/E001",
      sourceAdmissionStatus: "QUARANTINED_PENDING_INDEPENDENT_REVIEW",
    },
    replayDeclaration: {
      episodeJsonSha256,
      episodeId: "E001",
      episodeAncestorId: "P001/S001/E001",
      raterCode,
      encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
    },
    reviewEntry: {
      episodeId: "E001",
      episodeAncestorId: "P001/S001/E001",
      annotation: {
        annotationId,
        raterCode,
        assessment: "uncertain",
      },
    },
  };
  return checksummedArtifact(
    `E001-${raterCode}.independent-review.json`,
    utf8(`${JSON.stringify(review, null, 2)}\n`),
  );
}

async function calibrationArtifact(
  extra: Record<string, unknown> = {},
): Promise<RawChecksummedArtifact> {
  const observation = (frameIdentity: number, mediaTimeMs: number) => ({
    frameIdentity,
    mediaTimeMs,
    callbackMonotonicTimeMs: 1_000 + mediaTimeMs,
    identityConfidence: 1,
  });
  const fit = fitMediaClockCalibration(
    [observation(0, 0), observation(1, 40), observation(2, 80)],
    [observation(0, 8), observation(1, 48), observation(2, 88)],
  );
  const artifact = { ...fit, ...extra };
  return checksummedArtifact(
    "media-clock-calibration-v1.json",
    utf8(`${JSON.stringify(artifact, null, 2)}\n`),
  );
}

async function packageFixture(
  includeUntrustedIdentityExtensions = false,
): Promise<AdmissionPackageV1Input & { archiveFixture: ArchiveFixture }> {
  const archive = await archiveFixture(includeUntrustedIdentityExtensions);
  return {
    archive: archive.artifact,
    reviews: [
      await reviewArtifact(archive.episodeJsonSha256, "RATER-B", "ANN-B"),
      await reviewArtifact(archive.episodeJsonSha256, "RATER-A", "ANN-A"),
    ],
    calibration: await calibrationArtifact(),
    archiveFixture: archive,
  };
}

describe("byte-rooted admission package preflight", () => {
  it("retains deterministic full lineage while quarantining current v1", async () => {
    const fixture = await packageFixture();
    const forward = await preflightAdmissionPackageV1(fixture);
    const reversed = await preflightAdmissionPackageV1({
      ...fixture,
      reviews: [...fixture.reviews].reverse(),
    });

    expect(forward).toEqual(reversed);
    expect(forward).toMatchObject({
      schemaVersion: "relay.body-native-admission-package-preflight/1",
      status: "QUARANTINED",
      admitted: false,
      evidenceClass: "QUARANTINED_UNADMITTED",
      permittedUse: "integrity-and-lineage-audit-only",
      lineage: {
        archive: {
          episodeId: "E001",
          episodeAncestorId: "P001/S001/E001",
          episodeJsonSha256: fixture.archiveFixture.episodeJsonSha256,
          continuousVideoSha256: fixture.archiveFixture.videoSha256,
          videoTimingSha256: fixture.archiveFixture.videoTimingSha256,
        },
        calibration: {
          episodeBindingStatus: "ABSENT_FROM_CALIBRATION_SCHEMA_V1",
        },
      },
    });
    expect(forward.lineage.archive.members.map((member) => member.path)).toEqual([
      "continuous.video-timing.json",
      "continuous/continuous.webm",
      "episode.json",
    ]);
    expect(forward.lineage.reviews.map((review) => review.raterCode)).toEqual([
      "RATER-A",
      "RATER-B",
    ]);
    expect(forward.packageRoot.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(Object.isFrozen(forward.lineage.archive.members)).toBe(true);
  });

  it("rejects any artifact tampering against its raw checksum sidecar", async () => {
    const fixture = await packageFixture();
    const tamperedBytes = Uint8Array.from(fixture.reviews[0].bytes);
    tamperedBytes[0] ^= 1;

    await expect(
      preflightAdmissionPackageV1({
        ...fixture,
        reviews: [{ ...fixture.reviews[0], bytes: tamperedBytes }],
      }),
    ).rejects.toThrow(/SHA-256 does not match/);
  });

  it("rejects duplicate review bytes, identities, and raters", async () => {
    const fixture = await packageFixture();
    const duplicate = await checksummedArtifact(
      "E001-copy.independent-review.json",
      fixture.reviews[0].bytes,
    );

    await expect(
      preflightAdmissionPackageV1({
        ...fixture,
        reviews: [fixture.reviews[0], duplicate],
      }),
    ).rejects.toThrow(/Duplicate review (bytes|annotation ID|rater)/);
  });

  it("rejects malformed and noncanonical JSON even when its checksum is valid", async () => {
    const fixture = await packageFixture();
    const malformed = await checksummedArtifact(
      "media-clock-calibration-v1.json",
      utf8("{not-json}\n"),
    );
    await expect(
      preflightAdmissionPackageV1({ ...fixture, calibration: malformed }),
    ).rejects.toThrow(/not valid JSON/);

    const reviewObject = JSON.parse(
      new TextDecoder().decode(fixture.reviews[0].bytes),
    ) as object;
    const noncanonical = await checksummedArtifact(
      fixture.reviews[0].fileName,
      utf8(JSON.stringify(reviewObject)),
    );
    await expect(
      preflightAdmissionPackageV1({ ...fixture, reviews: [noncanonical] }),
    ).rejects.toThrow(/not the producer-canonical/);
  });

  it("makes the missing episode/video/timing calibration binding explicit", async () => {
    const report = await preflightAdmissionPackageV1(await packageFixture());
    const gate = report.gates.find(
      (entry) => entry.id === "episode-video-timing-calibration-binding",
    );

    expect(gate).toMatchObject({
      passed: false,
      facts: {
        episodeJsonBound: false,
        continuousVideoBound: false,
        videoTimingArtifactBound: false,
      },
    });
    expect(gate?.issues.join(" ")).toMatch(/not cryptographically bound/);
  });

  it("has no path to ALLOWED even when callers inject plausible v2 extensions", async () => {
    const fixture = await packageFixture(true);
    const calibration = await calibrationArtifact({
      episodeBinding: {
        episodeJsonSha256: fixture.archiveFixture.episodeJsonSha256,
        continuousVideoSha256: fixture.archiveFixture.videoSha256,
        videoTimingSha256: fixture.archiveFixture.videoTimingSha256,
      },
    });
    const report = await preflightAdmissionPackageV1({ ...fixture, calibration });
    const identityGate = report.gates.find(
      (entry) => entry.id === "runtime-and-app-build-identity",
    );

    expect(identityGate).toMatchObject({
      passed: false,
      facts: {
        untrustedExtensionRuntimeVersionPresent: true,
        untrustedExtensionAppBuildSha256Present: true,
        trustedByEpisodeV2Schema: false,
      },
    });
    expect(report.status).toBe("QUARANTINED");
    expect(report.admitted).toBe(false);
    expect(report.gates.at(-1)).toMatchObject({
      id: "v1-admission-contract",
      passed: false,
      facts: {
        admissionImplementationPresent: false,
        mapsToContinuousCaptureAdmissionEpisode: false,
      },
    });
  });
});
