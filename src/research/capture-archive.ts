import { zipSync } from "fflate";
import {
  canonicalJson,
  buildResearchReplayAttachments,
  sha256Bytes,
  sha256Text,
  type ResearchBundle,
  type ResearchExport,
} from "./capture-bundle";
import {
  auditLocalReplayIntegrity,
  REPLAY_INTEGRITY_EVIDENCE_SCOPE,
} from "./capture-replay";

export interface ResearchArchive {
  fileName: string;
  blob: Blob;
  bytes: number;
  sha256: string;
  sha256Sidecar: string;
  memberCount: number;
}

function safeName(value: string): string {
  return value.trim().replace(/[^a-zA-Z0-9_-]+/g, "-") || "episode";
}

function decodeFrozenBundle(researchExport: ResearchExport): ResearchBundle {
  let parsed: unknown;
  try {
    parsed = JSON.parse(researchExport.json);
  } catch {
    throw new Error("The frozen episode JSON is not valid JSON.");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("The frozen episode JSON must contain one bundle object.");
  }
  return parsed as ResearchBundle;
}

function utf8(value: string): Uint8Array {
  return new TextEncoder().encode(value);
}

/**
 * `fflate` uses `instanceof Uint8Array` to distinguish a file from a nested
 * directory. In browser-like test/runtime realms, `TextEncoder` can return a
 * typed array owned by a different realm, which is then misread as a directory
 * tree. Round-trip through Blob so every ZIP member is owned by this realm.
 */
async function archiveUtf8(value: string): Promise<Uint8Array> {
  return new Uint8Array(
    await new Blob([value], { type: "text/plain;charset=utf-8" }).arrayBuffer(),
  );
}

/**
 * Create one archive only from the already-frozen episode JSON. The mutable
 * in-memory `bundle` is deliberately ignored after verifying the export digest,
 * so a later UI mutation cannot produce split-brain members.
 */
export async function buildResearchArchive(
  researchExport: ResearchExport,
  continuousVideo: Blob,
): Promise<ResearchArchive> {
  if ((await sha256Text(researchExport.json)) !== researchExport.sha256) {
    throw new Error("Episode JSON no longer matches its export digest.");
  }
  const bundle = decodeFrozenBundle(researchExport);
  const declaredVideo = bundle.session.video;
  const videoBytes = new Uint8Array(await continuousVideo.arrayBuffer());
  if (
    continuousVideo.size !== declaredVideo.bytes ||
    (continuousVideo.type || declaredVideo.mimeType) !== declaredVideo.mimeType ||
    (await sha256Bytes(videoBytes)) !== declaredVideo.sha256
  ) {
    throw new Error("Continuous video bytes do not match the declared episode asset.");
  }

  const poseAttachments = buildResearchReplayAttachments(
    bundle.session.poseFrames,
  );
  const recomputedReplayAudit = await auditLocalReplayIntegrity({
    declaredVideo,
    observedVideo: { fileName: declaredVideo.fileName, blob: continuousVideo },
    decodedVideo:
      bundle.session.replayAudit.integrityAudit.decodedVideo.observed,
    videoFrames: bundle.session.videoFrameTimings,
    poseAttachments,
    intervals: bundle.session.intervals,
    confirmation: {
      ...bundle.session.replayAudit.humanConfirmation,
      evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
    },
  });
  if (
    !recomputedReplayAudit.valid ||
    canonicalJson(recomputedReplayAudit) !==
      canonicalJson(bundle.session.replayAudit.integrityAudit)
  ) {
    throw new Error(
      "Replay integrity does not exactly match recomputation from the frozen episode and actual continuous video bytes.",
    );
  }

  const textMembers: Record<string, string> = {
    "episode.json": researchExport.json,
    "protocol.json": `${JSON.stringify(bundle.protocol.protocol, null, 2)}\n`,
    "continuous.pose.json": `${JSON.stringify(bundle.session.poseFrames, null, 2)}\n`,
    "continuous.video-timing.json": `${JSON.stringify(bundle.session.videoFrameTimings, null, 2)}\n`,
    "segments.json": `${JSON.stringify(bundle.session.intervals, null, 2)}\n`,
    "derived/aids.json": `${JSON.stringify(bundle.session.derivedAids, null, 2)}\n`,
    "protocol-execution.json": `${JSON.stringify(bundle.session.protocolExecution, null, 2)}\n`,
    "replay-audit.json": `${JSON.stringify(bundle.session.replayAudit, null, 2)}\n`,
    "environment.json": `${JSON.stringify(bundle.environment, null, 2)}\n`,
    "annotations/collector.json": `${JSON.stringify(bundle.annotations, null, 2)}\n`,
    "CLAIM_BOUNDARY.txt": `${bundle.claimBoundary.join("\n")}\n`,
  };
  const binaryMembers: Record<string, Uint8Array> = {
    [`continuous/${declaredVideo.fileName}`]: videoBytes,
  };

  const expectedContentDigests = new Map(
    bundle.contentDigests.map((entry) => [entry.path, entry]),
  );
  const contentPaths = [
    ...Object.keys(textMembers).filter(
      (path) => path !== "episode.json" && path !== "annotations/collector.json" && path !== "CLAIM_BOUNDARY.txt",
    ),
    ...Object.keys(binaryMembers),
  ];
  for (const path of contentPaths) {
    const bytes = textMembers[path] ? utf8(textMembers[path]) : binaryMembers[path];
    const observed = {
      bytes: bytes.byteLength,
      sha256: await sha256Bytes(bytes),
    };
    const expected = expectedContentDigests.get(path);
    if (
      !expected ||
      expected.bytes !== observed.bytes ||
      expected.sha256 !== observed.sha256
    ) {
      throw new Error(`Frozen content digest does not match archive member ${path}.`);
    }
  }
  if (expectedContentDigests.size !== contentPaths.length) {
    throw new Error("Frozen content-digest paths do not exactly match archive-bound content members.");
  }

  const checksums: string[] = [];
  for (const [path, value] of Object.entries(textMembers).sort()) {
    checksums.push(`${await sha256Text(value)}  ${path}`);
  }
  for (const [path, value] of Object.entries(binaryMembers).sort()) {
    checksums.push(`${await sha256Bytes(value)}  ${path}`);
  }
  textMembers["SHA256SUMS.txt"] = `${checksums.join("\n")}\n`;

  const archiveMembers: Record<string, Uint8Array> = {};
  for (const [path, value] of Object.entries(textMembers)) {
    archiveMembers[path] = await archiveUtf8(value);
  }
  Object.assign(archiveMembers, binaryMembers);
  const zipped = zipSync(archiveMembers, { level: 0 });
  const sha256 = await sha256Bytes(zipped);
  const stem = [
    safeName(bundle.metadata.participantCode),
    safeName(bundle.metadata.sessionCode),
    safeName(bundle.metadata.episodeCode),
  ].join("-");
  const fileName = `${stem}.relay-episode-v2.zip`;
  return {
    fileName,
    blob: new Blob([Uint8Array.from(zipped)], { type: "application/zip" }),
    bytes: zipped.byteLength,
    sha256,
    sha256Sidecar: `${sha256}  ${fileName}\n`,
    memberCount: Object.keys(archiveMembers).length,
  };
}
