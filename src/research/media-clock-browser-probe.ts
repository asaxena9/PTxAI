import {
  fitMediaClockCalibration,
  type MediaClockCalibrationFit,
  type MediaClockFrameObservation,
} from "./media-clock-calibration";
import {
  decodeMediaClockMarkerFromCanvas,
  drawMediaClockMarker,
  type MediaClockMarkerDecode,
  type MediaClockMarkerGeometry,
} from "./media-clock-marker";

export const SYNTHETIC_MEDIA_CLOCK_PROBE_SCHEMA =
  "relay.synthetic-media-clock-browser-probe/1" as const;

export const SYNTHETIC_MEDIA_CLOCK_PROBE_SCOPE =
  "SYNTHETIC_CANVAS_BROWSER_MECHANICS_ONLY" as const;

export const SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG = {
  width: 768,
  height: 432,
  frameCount: 48,
  firstFrameIdentity: 1,
  targetFrameIntervalMs: 50,
  maximumWallTimeMs: 20_000,
} as const;

export const SYNTHETIC_MEDIA_CLOCK_PROBE_CLAIM_BOUNDARY = [
  "This trial observes a generated canvas stream; it contains no person, camera sensor, pose, or movement.",
  "A synthetic-stream fit does not establish transfer to a physical camera stream.",
  "This report is not admission calibration and cannot set encodedReplayTimelineCalibrated=true.",
  "The fit characterizes only the exact browser, recorder, codec, and synthetic path retained in this report.",
] as const;

const RECORDER_MIME_CANDIDATES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
  "video/mp4",
] as const;

const MARKER_GEOMETRY: MediaClockMarkerGeometry = {
  x: 24,
  y: 368,
  cellWidth: 28,
  cellHeight: 40,
};

interface ManualCanvasCaptureTrack extends MediaStreamTrack {
  requestFrame: () => void;
}

export interface SyntheticMediaClockProbeElements {
  readonly sourceCanvas: HTMLCanvasElement;
  readonly sourceVideo: HTMLVideoElement;
  readonly sourceDecodeCanvas: HTMLCanvasElement;
  readonly replayVideo: HTMLVideoElement;
  readonly replayDecodeCanvas: HTMLCanvasElement;
}

export interface SyntheticMediaClockRawObservation {
  readonly trace: "source-presented" | "encoded-replay";
  readonly callbackMonotonicTimeMs: number;
  readonly mediaTimeMs: number;
  readonly expectedDisplayTimeMs: number;
  readonly presentedFrames: number;
  readonly decoded: MediaClockMarkerDecode;
}

export interface SyntheticMediaClockProbeReport {
  readonly schemaVersion: typeof SYNTHETIC_MEDIA_CLOCK_PROBE_SCHEMA;
  readonly status: "COMPLETED" | "UNSUPPORTED_OR_FAILED";
  readonly evidenceClass: "MEASURED-PILOT";
  readonly scope: typeof SYNTHETIC_MEDIA_CLOCK_PROBE_SCOPE;
  readonly cameraTransferEstablished: false;
  readonly admissionCalibration: false;
  readonly startedAtIso: string;
  readonly completedAtIso: string;
  readonly elapsedWallTimeMs: number;
  readonly config: typeof SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG;
  readonly runtime: {
    readonly userAgent: string;
    readonly platform: string;
    readonly hardwareConcurrency: number | null;
    readonly secureContext: boolean;
  };
  readonly recorder: {
    readonly requestedMimeType: string | null;
    readonly realizedMimeType: string | null;
    readonly blobMimeType: string | null;
    readonly blobByteLength: number | null;
    readonly blobSha256: string | null;
  };
  readonly rawObservations: {
    readonly source: readonly SyntheticMediaClockRawObservation[];
    readonly encoded: readonly SyntheticMediaClockRawObservation[];
  };
  readonly fit: MediaClockCalibrationFit | null;
  readonly issues: readonly string[];
  readonly claimBoundary: typeof SYNTHETIC_MEDIA_CLOCK_PROBE_CLAIM_BOUNDARY;
}

export interface SyntheticMediaClockProbeRun {
  readonly report: SyntheticMediaClockProbeReport;
  readonly recordedBlob: Blob | null;
}

export type SyntheticMediaClockProbeProgress =
  | "preparing"
  | "recording"
  | "hashing"
  | "replaying"
  | "fitting";

export interface SyntheticMediaClockProbeOptions {
  readonly onProgress?: (progress: SyntheticMediaClockProbeProgress) => void;
}

let probeRunning = false;

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function waitForDecodedIdentity(
  observations: readonly SyntheticMediaClockRawObservation[],
  frameIdentity: number,
  timeoutMs: number,
): Promise<void> {
  const deadline = performance.now() + timeoutMs;
  while (
    !observations.some(
      (entry) =>
        entry.decoded.valid && entry.decoded.frameIdentity === frameIdentity,
    )
  ) {
    if (performance.now() >= deadline) {
      throw new Error(
        `Source presentation did not acknowledge frame identity ${frameIdentity}.`,
      );
    }
    await delay(4);
  }
}

function runtimeRecord(): SyntheticMediaClockProbeReport["runtime"] {
  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    hardwareConcurrency: Number.isFinite(navigator.hardwareConcurrency)
      ? navigator.hardwareConcurrency
      : null,
    secureContext: window.isSecureContext,
  };
}

export function selectSupportedMediaRecorderMimeType(
  isTypeSupported: (mimeType: string) => boolean,
): string | null {
  return (
    RECORDER_MIME_CANDIDATES.find((mimeType) => isTypeSupported(mimeType)) ??
    null
  );
}

export function calibrationObservationsFromRaw(
  observations: readonly SyntheticMediaClockRawObservation[],
  firstFrameIdentity = SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.firstFrameIdentity,
  frameCount = SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.frameCount,
): MediaClockFrameObservation[] {
  const lastFrameIdentity = firstFrameIdentity + frameCount - 1;
  return observations.flatMap((observation) => {
    const identity = observation.decoded.frameIdentity;
    if (
      !observation.decoded.valid ||
      identity === null ||
      identity < firstFrameIdentity ||
      identity > lastFrameIdentity
    ) {
      return [];
    }
    return [
      {
        frameIdentity: identity,
        mediaTimeMs: observation.mediaTimeMs,
        callbackMonotonicTimeMs: observation.callbackMonotonicTimeMs,
        identityConfidence: observation.decoded.confidence,
      },
    ];
  });
}

function drawSyntheticFrame(
  canvas: HTMLCanvasElement,
  frameIdentity: number,
): void {
  canvas.width = SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.width;
  canvas.height = SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("The source canvas could not create a 2D context.");

  const hueShift = (frameIdentity * 13) % 80;
  context.fillStyle = `hsl(${158 + hueShift / 8} 55% 16%)`;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(168, 255, 207, 0.12)";
  context.beginPath();
  context.arc(
    90 + ((frameIdentity * 17) % 560),
    105 + ((frameIdentity * 7) % 170),
    76,
    0,
    Math.PI * 2,
  );
  context.fill();
  context.fillStyle = "#f5f3ed";
  context.font = "600 28px system-ui, sans-serif";
  context.fillText("SYNTHETIC CLOCK FRAME", 28, 52);
  context.font = "700 88px ui-monospace, monospace";
  context.fillText(frameIdentity.toString().padStart(4, "0"), 28, 156);
  context.fillStyle = "rgba(245, 243, 237, 0.68)";
  context.font = "500 18px system-ui, sans-serif";
  context.fillText("generated pixels · no camera · no person", 30, 198);
  drawMediaClockMarker(context, frameIdentity, MARKER_GEOMETRY);
}

function decodePresentedFrame(
  video: HTMLVideoElement,
  decodeCanvas: HTMLCanvasElement,
): MediaClockMarkerDecode {
  const width = video.videoWidth || SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.width;
  const height = video.videoHeight || SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.height;
  decodeCanvas.width = width;
  decodeCanvas.height = height;
  const context = decodeCanvas.getContext("2d", {
    alpha: false,
    willReadFrequently: true,
  });
  if (!context) {
    return {
      valid: false,
      frameIdentity: null,
      confidence: 0,
      threshold: 0.5,
      bits: [],
      reason: "The decode canvas could not create a 2D context.",
    };
  }
  try {
    context.drawImage(video, 0, 0, width, height);
    return decodeMediaClockMarkerFromCanvas(context, MARKER_GEOMETRY);
  } catch (error) {
    return {
      valid: false,
      frameIdentity: null,
      confidence: 0,
      threshold: 0.5,
      bits: [],
      reason:
        error instanceof Error
          ? `Video frame decode failed: ${error.message}`
          : "Video frame decode failed.",
    };
  }
}

function startVideoFrameObserver(
  trace: SyntheticMediaClockRawObservation["trace"],
  video: HTMLVideoElement,
  decodeCanvas: HTMLCanvasElement,
  destination: SyntheticMediaClockRawObservation[],
): () => void {
  let active = true;
  let callbackId: number | null = null;
  const observe: VideoFrameRequestCallback = (now, metadata) => {
    destination.push({
      trace,
      callbackMonotonicTimeMs: now,
      mediaTimeMs: metadata.mediaTime * 1000,
      expectedDisplayTimeMs: metadata.expectedDisplayTime,
      presentedFrames: metadata.presentedFrames,
      decoded: decodePresentedFrame(video, decodeCanvas),
    });
    if (active) callbackId = video.requestVideoFrameCallback(observe);
  };
  callbackId = video.requestVideoFrameCallback(observe);
  return () => {
    active = false;
    if (callbackId !== null) video.cancelVideoFrameCallback(callbackId);
  };
}

function waitForEvent(
  target: EventTarget,
  eventName: string,
  timeoutMs: number,
  errorTarget?: HTMLMediaElement,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error(`Timed out waiting for ${eventName}.`));
    }, timeoutMs);
    const onEvent = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(
        new Error(
          errorTarget?.error?.message ?? `A media error occurred before ${eventName}.`,
        ),
      );
    };
    const cleanup = () => {
      window.clearTimeout(timeout);
      target.removeEventListener(eventName, onEvent);
      errorTarget?.removeEventListener("error", onError);
    };
    target.addEventListener(eventName, onEvent, { once: true });
    errorTarget?.addEventListener("error", onError, { once: true });
  });
}

function recordStopPromise(
  recorder: MediaRecorder,
  chunks: Blob[],
): Promise<void> {
  return new Promise((resolve, reject) => {
    recorder.addEventListener(
      "dataavailable",
      (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      },
    );
    recorder.addEventListener("stop", () => resolve(), { once: true });
    recorder.addEventListener(
      "error",
      () => reject(new Error("MediaRecorder reported an error.")),
      { once: true },
    );
  });
}

async function sha256Blob(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

function unsupportedReport(
  startedAtIso: string,
  startedAtMonotonicMs: number,
  issues: readonly string[],
  rawSource: readonly SyntheticMediaClockRawObservation[] = [],
  rawEncoded: readonly SyntheticMediaClockRawObservation[] = [],
  recorder: SyntheticMediaClockProbeReport["recorder"] = {
    requestedMimeType: null,
    realizedMimeType: null,
    blobMimeType: null,
    blobByteLength: null,
    blobSha256: null,
  },
): SyntheticMediaClockProbeReport {
  return {
    schemaVersion: SYNTHETIC_MEDIA_CLOCK_PROBE_SCHEMA,
    status: "UNSUPPORTED_OR_FAILED",
    evidenceClass: "MEASURED-PILOT",
    scope: SYNTHETIC_MEDIA_CLOCK_PROBE_SCOPE,
    cameraTransferEstablished: false,
    admissionCalibration: false,
    startedAtIso,
    completedAtIso: new Date().toISOString(),
    elapsedWallTimeMs: performance.now() - startedAtMonotonicMs,
    config: SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG,
    runtime: runtimeRecord(),
    recorder,
    rawObservations: { source: rawSource, encoded: rawEncoded },
    fit: null,
    issues,
    claimBoundary: SYNTHETIC_MEDIA_CLOCK_PROBE_CLAIM_BOUNDARY,
  };
}

/**
 * Run one bounded synthetic timing trial. A module-level mutex rejects overlap,
 * including overlap started by a second component instance.
 */
export async function runSyntheticMediaClockProbe(
  elements: SyntheticMediaClockProbeElements,
  options: SyntheticMediaClockProbeOptions = {},
): Promise<SyntheticMediaClockProbeRun> {
  if (probeRunning) {
    throw new Error("A synthetic media-clock trial is already running.");
  }
  probeRunning = true;
  const startedAtIso = new Date().toISOString();
  const startedAtMonotonicMs = performance.now();
  const rawSource: SyntheticMediaClockRawObservation[] = [];
  const rawEncoded: SyntheticMediaClockRawObservation[] = [];
  let stream: MediaStream | null = null;
  let stopSourceObserver: (() => void) | null = null;
  let stopReplayObserver: (() => void) | null = null;
  let replayUrl: string | null = null;
  let recorder: MediaRecorder | null = null;
  let recordedBlob: Blob | null = null;
  let requestedMimeType: string | null = null;
  let blobSha256: string | null = null;

  const wallTimeGuard = window.setTimeout(() => {
    if (recorder?.state === "recording") recorder.stop();
    stream?.getTracks().forEach((track) => track.stop());
    elements.replayVideo.pause();
  }, SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.maximumWallTimeMs);

  try {
    options.onProgress?.("preparing");
    if (
      typeof HTMLCanvasElement.prototype.captureStream !== "function" ||
      typeof window.MediaRecorder !== "function" ||
      typeof HTMLVideoElement.prototype.requestVideoFrameCallback !== "function" ||
      typeof crypto?.subtle?.digest !== "function"
    ) {
      return {
        report: unsupportedReport(startedAtIso, startedAtMonotonicMs, [
          "This browser lacks canvas capture, MediaRecorder, requestVideoFrameCallback, or Web Crypto SHA-256.",
        ]),
        recordedBlob: null,
      };
    }

    requestedMimeType = selectSupportedMediaRecorderMimeType((mimeType) =>
      MediaRecorder.isTypeSupported(mimeType),
    );
    if (!requestedMimeType) {
      return {
        report: unsupportedReport(startedAtIso, startedAtMonotonicMs, [
          "No declared MediaRecorder video MIME candidate is supported.",
        ]),
        recordedBlob: null,
      };
    }

    drawSyntheticFrame(elements.sourceCanvas, 0);
    stream = elements.sourceCanvas.captureStream(0);
    const videoTrack = stream.getVideoTracks()[0] as
      | ManualCanvasCaptureTrack
      | undefined;
    if (!videoTrack || typeof videoTrack.requestFrame !== "function") {
      return {
        report: unsupportedReport(startedAtIso, startedAtMonotonicMs, [
          "The canvas capture track does not expose manual requestFrame().",
        ]),
        recordedBlob: null,
      };
    }

    elements.sourceVideo.muted = true;
    elements.sourceVideo.playsInline = true;
    elements.sourceVideo.srcObject = stream;
    videoTrack.requestFrame();
    await elements.sourceVideo.play();
    stopSourceObserver = startVideoFrameObserver(
      "source-presented",
      elements.sourceVideo,
      elements.sourceDecodeCanvas,
      rawSource,
    );

    recorder = new MediaRecorder(stream, { mimeType: requestedMimeType });
    const chunks: Blob[] = [];
    const recorderStopped = recordStopPromise(recorder, chunks);
    recorder.start();
    options.onProgress?.("recording");

    for (
      let offset = 0;
      offset < SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.frameCount;
      offset += 1
    ) {
      const identity =
        SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.firstFrameIdentity + offset;
      const requestedAtMs = performance.now();
      drawSyntheticFrame(elements.sourceCanvas, identity);
      videoTrack.requestFrame();
      await waitForDecodedIdentity(rawSource, identity, 250);
      const remainingIntervalMs = Math.max(
        0,
        SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.targetFrameIntervalMs -
          (performance.now() - requestedAtMs),
      );
      await delay(remainingIntervalMs);
    }
    await delay(SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.targetFrameIntervalMs);
    recorder.stop();
    await recorderStopped;
    stopSourceObserver();
    stopSourceObserver = null;

    recordedBlob = new Blob(chunks, {
      type: recorder.mimeType || requestedMimeType,
    });
    if (recordedBlob.size === 0) {
      return {
        report: unsupportedReport(
          startedAtIso,
          startedAtMonotonicMs,
          ["MediaRecorder produced an empty Blob."],
          rawSource,
          rawEncoded,
          {
            requestedMimeType,
            realizedMimeType: recorder.mimeType || null,
            blobMimeType: recordedBlob.type || null,
            blobByteLength: 0,
            blobSha256: null,
          },
        ),
        recordedBlob,
      };
    }

    options.onProgress?.("hashing");
    blobSha256 = await sha256Blob(recordedBlob);
    options.onProgress?.("replaying");
    replayUrl = URL.createObjectURL(recordedBlob);
    elements.replayVideo.muted = true;
    elements.replayVideo.playsInline = true;
    elements.replayVideo.preload = "auto";
    const metadataLoaded = waitForEvent(
      elements.replayVideo,
      "loadedmetadata",
      4_000,
      elements.replayVideo,
    );
    elements.replayVideo.src = replayUrl;
    elements.replayVideo.load();
    await metadataLoaded;
    stopReplayObserver = startVideoFrameObserver(
      "encoded-replay",
      elements.replayVideo,
      elements.replayDecodeCanvas,
      rawEncoded,
    );
    const replayEnded = waitForEvent(
      elements.replayVideo,
      "ended",
      8_000,
      elements.replayVideo,
    );
    await elements.replayVideo.play();
    await replayEnded;
    await delay(0);
    stopReplayObserver();
    stopReplayObserver = null;

    options.onProgress?.("fitting");
    const sourceForFit = calibrationObservationsFromRaw(rawSource);
    const encodedForFit = calibrationObservationsFromRaw(rawEncoded);
    const fit = fitMediaClockCalibration(sourceForFit, encodedForFit);
    const issues = [
      ...fit.issues,
      ...(sourceForFit.length === 0
        ? ["No valid source frame identities were decoded."]
        : []),
      ...(encodedForFit.length === 0
        ? ["No valid encoded frame identities were decoded."]
        : []),
    ];
    const report: SyntheticMediaClockProbeReport = {
      schemaVersion: SYNTHETIC_MEDIA_CLOCK_PROBE_SCHEMA,
      status: "COMPLETED",
      evidenceClass: "MEASURED-PILOT",
      scope: SYNTHETIC_MEDIA_CLOCK_PROBE_SCOPE,
      cameraTransferEstablished: false,
      admissionCalibration: false,
      startedAtIso,
      completedAtIso: new Date().toISOString(),
      elapsedWallTimeMs: performance.now() - startedAtMonotonicMs,
      config: SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG,
      runtime: runtimeRecord(),
      recorder: {
        requestedMimeType,
        realizedMimeType: recorder.mimeType || null,
        blobMimeType: recordedBlob.type || null,
        blobByteLength: recordedBlob.size,
        blobSha256,
      },
      rawObservations: { source: rawSource, encoded: rawEncoded },
      fit,
      issues,
      claimBoundary: SYNTHETIC_MEDIA_CLOCK_PROBE_CLAIM_BOUNDARY,
    };
    return { report, recordedBlob };
  } catch (error) {
    const issue =
      error instanceof Error ? error.message : "The browser probe failed.";
    return {
      report: unsupportedReport(
        startedAtIso,
        startedAtMonotonicMs,
        [issue],
        rawSource,
        rawEncoded,
        {
          requestedMimeType,
          realizedMimeType: recorder?.mimeType || null,
          blobMimeType: recordedBlob?.type || null,
          blobByteLength: recordedBlob?.size ?? null,
          blobSha256,
        },
      ),
      recordedBlob,
    };
  } finally {
    window.clearTimeout(wallTimeGuard);
    stopSourceObserver?.();
    stopReplayObserver?.();
    if (recorder?.state === "recording") recorder.stop();
    stream?.getTracks().forEach((track) => track.stop());
    elements.sourceVideo.pause();
    elements.sourceVideo.srcObject = null;
    if (replayUrl) URL.revokeObjectURL(replayUrl);
    probeRunning = false;
  }
}
