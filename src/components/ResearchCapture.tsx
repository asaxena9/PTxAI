"use client";

import {
  ArrowLeft,
  Camera,
  CheckCircle2,
  Download,
  FlaskConical,
  LockKeyhole,
  Play,
  ShieldCheck,
  Square,
  Video,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { MediaPipeMotionProvider } from "../motion/mediapipe-provider";
import type { JointName, MotionProviderStatus, PoseFrame } from "../motion/types";
import { buildResearchArchive, type ResearchArchive } from "../research/capture-archive";
import {
  buildResearchExport,
  buildResearchReplayAttachments,
  canonicalJson,
  downloadBlob,
  RESEARCH_COLLECTOR_BUILD_ID,
  serializePoseFrame,
  sha256Bytes,
  sha256Text,
  storeResearchAids,
  type CaptureKind,
  type ResearchAnnotation,
  type ResearchContinuousCaptureSession,
  type ResearchEpisodeDraft,
  type ResearchExport,
  type ResearchPoseFrame,
  type ResearchPoseVideoTimingAttachment,
  type ResearchReplayAudit,
} from "../research/capture-bundle";
import {
  beginCaptureInterval,
  endCaptureInterval,
  validateCaptureTimingSession,
  type CaptureInterval,
  type OpenCaptureInterval,
  type VideoFrameTimingSample,
} from "../research/capture-timing";
import {
  auditLocalReplayIntegrity,
  observeDecodedContinuousVideo,
  RECORDED_BLOB_TIMELINE_MAPPING,
  REPLAY_INTEGRITY_EVIDENCE_SCOPE,
  selectLatestCausalPoseForReplayTime,
} from "../research/capture-replay";
import {
  executeObservableProtocol,
  type ProtocolExecutorPoseFrame,
} from "../research/protocol-executor";
import {
  freezeObservableProtocol,
  LATERAL_ARM_RAISE_REFERENCE_V1,
  type FrozenObservableProtocol,
} from "../research/protocol";

type CaptureStep = "setup" | "capture" | "replay" | "annotate" | "export";
type SessionStatus = "idle" | "starting" | "recording" | "finishing" | "finished";
type SessionBeforeReplay = Omit<ResearchContinuousCaptureSession, "replayAudit">;

const STEPS: readonly CaptureStep[] = [
  "setup",
  "capture",
  "replay",
  "annotate",
  "export",
];

const CONNECTIONS: Array<[JointName, JointName]> = [
  ["left-shoulder", "right-shoulder"],
  ["left-shoulder", "left-elbow"],
  ["left-elbow", "left-wrist"],
  ["right-shoulder", "right-elbow"],
  ["right-elbow", "right-wrist"],
  ["left-shoulder", "left-hip"],
  ["right-shoulder", "right-hip"],
  ["left-hip", "right-hip"],
  ["left-hip", "left-knee"],
  ["left-knee", "left-ankle"],
  ["right-hip", "right-knee"],
  ["right-knee", "right-ankle"],
];

function safeCode(value: string): string {
  return value.trim().replace(/[^a-zA-Z0-9_-]+/g, "-") || "episode";
}

function createCode(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

function drawLandmarks(
  canvas: HTMLCanvasElement | null,
  frame: Pick<ResearchPoseFrame, "frameWidth" | "frameHeight" | "image">,
): void {
  if (!canvas) return;
  const width = frame.frameWidth || 1280;
  const height = frame.frameHeight || 720;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  context.lineWidth = Math.max(3, width / 320);
  context.strokeStyle = "rgba(168, 255, 207, 0.9)";
  context.fillStyle = "#f28a67";
  for (const [startName, endName] of CONNECTIONS) {
    const start = frame.image[startName];
    const end = frame.image[endName];
    if (!start || !end || start.visibility < 0.35 || end.visibility < 0.35)
      continue;
    context.beginPath();
    context.moveTo(start.x * width, start.y * height);
    context.lineTo(end.x * width, end.y * height);
    context.stroke();
  }
  for (const landmark of Object.values(frame.image)) {
    if (!landmark || landmark.visibility < 0.35) continue;
    context.beginPath();
    context.arc(
      landmark.x * width,
      landmark.y * height,
      Math.max(3, width / 240),
      0,
      Math.PI * 2,
    );
    context.fill();
  }
}

function cameraSettings(video: HTMLVideoElement | null) {
  const stream = video?.srcObject;
  if (!(stream instanceof MediaStream)) return {};
  const settings = stream.getVideoTracks()[0]?.getSettings() ?? {};
  // Device and group identifiers are unnecessary for the mechanics pilot and
  // may be stable identifiers. Retain only declared mechanics settings.
  return Object.fromEntries(
    (["width", "height", "frameRate", "facingMode"] as const).flatMap(
      (key) => {
        const value = settings[key];
        return typeof value === "string" || typeof value === "number"
          ? [[key, value] as const]
          : [];
      },
    ),
  );
}

function formatDegrees(value: number | undefined): string {
  return value === undefined ? "—" : `${value.toFixed(1)}°`;
}

export default function ResearchCapture() {
  const liveVideoRef = useRef<HTMLVideoElement>(null);
  const liveCanvasRef = useRef<HTMLCanvasElement>(null);
  const replayVideoRef = useRef<HTMLVideoElement>(null);
  const replayCanvasRef = useRef<HTMLCanvasElement>(null);
  const providerRef = useRef<MediaPipeMotionProvider | null>(null);
  const videoCallbackIdRef = useRef<number | null>(null);
  const latestFrameRef = useRef<PoseFrame | null>(null);
  const latestVideoTimingRef = useRef<VideoFrameTimingSample | null>(null);
  const pendingSessionStartRef = useRef(false);
  const sessionActiveRef = useRef(false);
  const pendingSessionFinishRef = useRef<(() => void) | null>(null);
  const sessionStartedAtIsoRef = useRef("");
  const sessionStartedMonotonicMsRef = useRef(0);
  const recorderStartedMonotonicMsRef = useRef(0);
  const sessionMediaOriginMsRef = useRef(0);
  const poseFramesRef = useRef<ResearchPoseFrame[]>([]);
  const videoTimingsRef = useRef<VideoFrameTimingSample[]>([]);
  const intervalsRef = useRef<CaptureInterval[]>([]);
  const openIntervalRef = useRef<OpenCaptureInterval | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const videoChunksRef = useRef<Blob[]>([]);
  const replayUrlRef = useRef<string | null>(null);
  const consentConfirmedAtRef = useRef("");

  const [step, setStep] = useState<CaptureStep>("setup");
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");
  const [cameraStatus, setCameraStatus] = useState<MotionProviderStatus>({
    state: "idle",
    label: "Camera not started",
  });
  const [latestFrame, setLatestFrame] = useState<PoseFrame | null>(null);
  const [openKind, setOpenKind] = useState<CaptureKind | null>(null);
  const [completedKinds, setCompletedKinds] = useState<CaptureKind[]>([]);
  const [frozenProtocol, setFrozenProtocol] =
    useState<FrozenObservableProtocol | null>(null);
  const [session, setSession] = useState<SessionBeforeReplay | null>(null);
  const [continuousVideo, setContinuousVideo] = useState<Blob | null>(null);
  const [replayUrl, setReplayUrl] = useState<string | null>(null);
  const [replayReviewerCode, setReplayReviewerCode] = useState("");
  const [watchedReplay, setWatchedReplay] = useState(false);
  const [replayMediaTimeMs, setReplayMediaTimeMs] = useState(0);
  const [overlayConfirmed, setOverlayConfirmed] = useState(false);
  const [referenceConfirmed, setReferenceConfirmed] = useState(false);
  const [attemptConfirmed, setAttemptConfirmed] = useState(false);
  const [replayAudit, setReplayAudit] = useState<ResearchReplayAudit | null>(null);
  const [collectorRaterCode, setCollectorRaterCode] = useState("");
  const [collectorConfidence, setCollectorConfidence] = useState("0.8");
  const [collectorNotes, setCollectorNotes] = useState("");
  const [notice, setNotice] = useState("");
  const [exportResult, setExportResult] = useState<ResearchExport | null>(null);
  const [archive, setArchive] = useState<ResearchArchive | null>(null);
  const [preparing, setPreparing] = useState(false);

  const [participantCode, setParticipantCode] = useState("");
  const [sessionCode, setSessionCode] = useState(() => createCode("session"));
  const [episodeCode, setEpisodeCode] = useState(() => createCode("episode"));
  const [deviceNotes, setDeviceNotes] = useState("");
  const [calibrationNotes, setCalibrationNotes] = useState("");
  const [adultSelfConfirmed, setAdultSelfConfirmed] = useState(false);
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [boundaryConfirmed, setBoundaryConfirmed] = useState(false);

  const stopVideoCallbacks = useCallback(() => {
    const video = liveVideoRef.current;
    if (
      video &&
      videoCallbackIdRef.current !== null &&
      typeof video.cancelVideoFrameCallback === "function"
    ) {
      video.cancelVideoFrameCallback(videoCallbackIdRef.current);
    }
    videoCallbackIdRef.current = null;
  }, []);

  const abortLocalCapture = useCallback(() => {
    pendingSessionStartRef.current = false;
    sessionActiveRef.current = false;
    pendingSessionFinishRef.current = null;
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
    recorderRef.current = null;
    stopVideoCallbacks();
    providerRef.current?.dispose();
    providerRef.current = null;
  }, [stopVideoCallbacks]);

  const handleCameraStatus = useCallback(
    (status: MotionProviderStatus) => {
      setCameraStatus(status);
      if (status.state === "error" || status.state === "permission-denied") {
        abortLocalCapture();
        latestFrameRef.current = null;
        setLatestFrame(null);
        setSessionStatus("idle");
        setStep("setup");
      }
    },
    [abortLocalCapture],
  );

  useEffect(() => {
    const onPageHide = () => abortLocalCapture();
    window.addEventListener("pagehide", onPageHide);
    return () => {
      window.removeEventListener("pagehide", onPageHide);
      abortLocalCapture();
    };
  }, [abortLocalCapture]);

  useEffect(() => {
    return () => {
      if (replayUrlRef.current) URL.revokeObjectURL(replayUrlRef.current);
    };
  }, []);

  function startVideoTimingLoop(video: HTMLVideoElement) {
    if (typeof video.requestVideoFrameCallback !== "function") {
      throw new Error(
        "This pilot requires a browser with requestVideoFrameCallback timing.",
      );
    }
    const callback: VideoFrameRequestCallback = (now, metadata) => {
      const sourceMediaTimeMs = metadata.mediaTime * 1000;
      if (pendingSessionStartRef.current) {
        const recorder = recorderRef.current;
        if (!recorder || recorder.state !== "inactive") {
          setNotice("The continuous video recorder could not start cleanly.");
          pendingSessionStartRef.current = false;
        } else {
          try {
            recorder.start(1000);
          } catch (error) {
            pendingSessionStartRef.current = false;
            setNotice(
              error instanceof Error
                ? error.message
                : "The continuous video recorder could not start.",
            );
            cancelCameraAndReset();
            return;
          }
          recorderStartedMonotonicMsRef.current = now;
          sessionStartedMonotonicMsRef.current = now;
          sessionStartedAtIsoRef.current = new Date().toISOString();
          sessionMediaOriginMsRef.current = sourceMediaTimeMs;
          const first: VideoFrameTimingSample = {
            sequence: 0,
            callbackMonotonicTimeMs: now,
            mediaTimeMs: 0,
            presentedFrames: metadata.presentedFrames,
          };
          videoTimingsRef.current = [first];
          latestVideoTimingRef.current = first;
          poseFramesRef.current = [];
          intervalsRef.current = [];
          openIntervalRef.current = null;
          pendingSessionStartRef.current = false;
          sessionActiveRef.current = true;
          setSessionStatus("recording");
        }
      } else if (sessionActiveRef.current) {
        const sample: VideoFrameTimingSample = {
          sequence: videoTimingsRef.current.length,
          callbackMonotonicTimeMs: now,
          mediaTimeMs: Math.max(0, sourceMediaTimeMs - sessionMediaOriginMsRef.current),
          presentedFrames: metadata.presentedFrames,
        };
        videoTimingsRef.current.push(sample);
        latestVideoTimingRef.current = sample;
        const finish = pendingSessionFinishRef.current;
        if (finish) {
          pendingSessionFinishRef.current = null;
          sessionActiveRef.current = false;
          finish();
        }
      }
      videoCallbackIdRef.current = video.requestVideoFrameCallback(callback);
    };
    videoCallbackIdRef.current = video.requestVideoFrameCallback(callback);
  }

  async function startCamera() {
    setNotice("");
    if (!participantCode.trim() || !sessionCode.trim() || !episodeCode.trim()) {
      setNotice("Complete the anonymous episode identity first.");
      return;
    }
    if (!adultSelfConfirmed || !rightsConfirmed || !boundaryConfirmed) {
      setNotice("All three limited-pilot confirmations are required.");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      setNotice("This pilot requires local continuous video recording.");
      return;
    }
    try {
      // Freeze the actual pre-camera authorization instant; never fabricate it
      // later when the export is built.
      consentConfirmedAtRef.current = new Date().toISOString();
      const protocol = await freezeObservableProtocol(
        LATERAL_ARM_RAISE_REFERENCE_V1,
      );
      setFrozenProtocol(protocol);
      providerRef.current?.dispose();
      const provider = new MediaPipeMotionProvider("user");
      providerRef.current = provider;
      await provider.start(
        liveVideoRef.current,
        (frame) => {
          latestFrameRef.current = frame;
          setLatestFrame(frame);
          drawLandmarks(liveCanvasRef.current, {
            frameWidth: frame.frameWidth,
            frameHeight: frame.frameHeight,
            image: frame.image,
          });
          if (!sessionActiveRef.current) return;
          const timing = latestVideoTimingRef.current;
          if (!timing) return;
          const attachment: ResearchPoseVideoTimingAttachment = {
            strategy: "nearest-latest-causal-video-frame",
            ...timing,
          };
          poseFramesRef.current.push(
            serializePoseFrame(
              frame,
              sessionStartedMonotonicMsRef.current,
              poseFramesRef.current.length,
              attachment,
            ),
          );
        },
        handleCameraStatus,
      );
      if (
        provider.getStatus().state !== "running" ||
        !liveVideoRef.current ||
        !(liveVideoRef.current.srcObject instanceof MediaStream)
      ) {
        throw new Error("The live pose stream did not remain active.");
      }
      startVideoTimingLoop(liveVideoRef.current);
      setStep("capture");
    } catch (error) {
      await providerRef.current?.stop();
      providerRef.current?.dispose();
      providerRef.current = null;
      stopVideoCallbacks();
      consentConfirmedAtRef.current = "";
      setNotice(
        error instanceof Error ? error.message : "The local camera could not start.",
      );
    }
  }

  function cancelCameraAndReset() {
    abortLocalCapture();
    latestFrameRef.current = null;
    latestVideoTimingRef.current = null;
    setLatestFrame(null);
    setCameraStatus({ state: "idle", label: "Camera stopped" });
    setSessionStatus("idle");
    setCompletedKinds([]);
    setOpenKind(null);
    consentConfirmedAtRef.current = "";
    setStep("setup");
    setNotice("Camera stopped. No episode was exported.");
  }

  function createRecorder(): MediaRecorder {
    const stream = liveVideoRef.current?.srcObject;
    if (!(stream instanceof MediaStream)) {
      throw new Error("The camera stream is not available.");
    }
    const preferred = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ].find((type) => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(
      stream,
      preferred ? { mimeType: preferred } : undefined,
    );
    videoChunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) videoChunksRef.current.push(event.data);
    };
    recorder.onerror = () => {
      setNotice("Continuous video recording failed; the camera was stopped.");
      cancelCameraAndReset();
    };
    return recorder;
  }

  function beginContinuousSession() {
    if (sessionStatus !== "idle" || cameraStatus.state !== "running" || !latestFrameRef.current) {
      setNotice("Wait until the camera and pose stream are ready.");
      return;
    }
    try {
      recorderRef.current = createRecorder();
      pendingSessionStartRef.current = true;
      setSessionStatus("starting");
      setNotice("");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Recording could not start.");
    }
  }

  function toggleInterval(kind: "reference" | "attempt") {
    const timing = latestVideoTimingRef.current;
    if (!timing || sessionStatus !== "recording") return;
    if (openIntervalRef.current) {
      if (openIntervalRef.current.kind !== kind) {
        setNotice(`Finish the ${openIntervalRef.current.kind} interval first.`);
        return;
      }
      const closed = endCaptureInterval(openIntervalRef.current, timing);
      if (!closed) {
        setNotice("The interval needs more than one video frame.");
        return;
      }
      intervalsRef.current.push(closed);
      openIntervalRef.current = null;
      setOpenKind(null);
      setCompletedKinds((current) => [...current, kind]);
      return;
    }
    const expected = intervalsRef.current.length === 0 ? "reference" : "attempt";
    if (kind !== expected || intervalsRef.current.some((item) => item.kind === kind)) {
      setNotice(`The next interval must be ${expected}.`);
      return;
    }
    const opened = beginCaptureInterval(kind, timing);
    if (!opened) {
      setNotice("A valid video-frame boundary is required.");
      return;
    }
    openIntervalRef.current = opened;
    setOpenKind(kind);
    setNotice("");
  }

  function stopRecorder(): Promise<Blob> {
    const recorder = recorderRef.current;
    if (!recorder) return Promise.reject(new Error("Missing continuous recorder."));
    return new Promise((resolve, reject) => {
      recorder.onerror = () => reject(new Error("Continuous video recording failed."));
      recorder.onstop = () => {
        const type = recorder.mimeType || "video/webm";
        resolve(new Blob(videoChunksRef.current, { type }));
      };
      recorder.stop();
    });
  }

  async function finishContinuousSession() {
    if (
      sessionStatus !== "recording" ||
      openIntervalRef.current ||
      intervalsRef.current.length !== 2 ||
      !frozenProtocol
    ) {
      setNotice("Finish both marked intervals before ending the continuous session.");
      return;
    }
    setSessionStatus("finishing");
    setNotice("");
    try {
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          pendingSessionFinishRef.current = null;
          reject(
            new Error(
              "No closing video-frame callback arrived within two seconds.",
            ),
          );
        }, 2000);
        pendingSessionFinishRef.current = () => {
          window.clearTimeout(timeout);
          resolve();
        };
      });
      const endedAt = new Date().toISOString();
      const durationMs =
        (videoTimingsRef.current.at(-1)?.callbackMonotonicTimeMs ??
          sessionStartedMonotonicMsRef.current) - sessionStartedMonotonicMsRef.current;
      const realizedCameraSettings = cameraSettings(liveVideoRef.current);
      const blobPromise = stopRecorder();
      await providerRef.current?.stop();
      providerRef.current?.dispose();
      providerRef.current = null;
      stopVideoCallbacks();
      setLatestFrame(null);
      const blob = await blobPromise;
      const videoSha = await sha256Bytes(new Uint8Array(await blob.arrayBuffer()));
      const poseFrames = [...poseFramesRef.current];
      const timings = [...videoTimingsRef.current];
      const intervals = [...intervalsRef.current];
      const timingValidation = validateCaptureTimingSession({
        videoFrames: timings,
        poses: poseFrames.map((frame) => ({
          sequence: frame.sequenceNumber,
          callbackMonotonicTimeMs: frame.callbackMonotonicTimeMs,
        })),
        intervals,
      });
      if (!timingValidation.valid) {
        throw new Error(
          `Timing audit failed: ${timingValidation.issues
            .map((entry) => entry.message)
            .join(" ")}`,
        );
      }
      const aids = await storeResearchAids(poseFrames);
      const execution = executeObservableProtocol(
        frozenProtocol,
        poseFrames as ProtocolExecutorPoseFrame[],
        intervals,
      );
      const fileName = `${safeCode(participantCode)}-${safeCode(sessionCode)}-${safeCode(
        episodeCode,
      )}-continuous.webm`;
      if (replayUrlRef.current) URL.revokeObjectURL(replayUrlRef.current);
      const nextReplayUrl = URL.createObjectURL(blob);
      replayUrlRef.current = nextReplayUrl;
      setContinuousVideo(blob);
      setReplayUrl(nextReplayUrl);
      setSession({
        sessionCaptureId: createCode("continuous"),
        startedAt: sessionStartedAtIsoRef.current,
        endedAt,
        durationMs,
        cameraFacing: "user",
        cameraSettings: realizedCameraSettings,
        clocks: {
          performanceTimeOriginMs: performance.timeOrigin,
          sessionStartedMonotonicMs: sessionStartedMonotonicMsRef.current,
          videoRecorderStartedMonotonicMs: recorderStartedMonotonicMsRef.current,
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
        videoFrameTimings: timings,
        intervals,
        derivedAids: aids,
        protocolExecution: execution,
        video: {
          fileName,
          mimeType: blob.type || "video/webm",
          bytes: blob.size,
          sha256: videoSha,
          continuousEpisode: true,
        },
      });
      setSessionStatus("finished");
      setStep("replay");
    } catch (error) {
      abortLocalCapture();
      setSessionStatus("idle");
      setCompletedKinds([]);
      setOpenKind(null);
      setStep("setup");
      setNotice(error instanceof Error ? error.message : "The session could not close.");
    }
  }

  function handleReplayTime() {
    if (!session || !replayVideoRef.current) return;
    setReplayMediaTimeMs(replayVideoRef.current.currentTime * 1000);
    const selected = selectLatestCausalPoseForReplayTime(
      replayVideoRef.current.currentTime * 1000,
      buildResearchReplayAttachments(session.poseFrames),
      session.videoFrameTimings,
    );
    if (selected) {
      const frame = session.poseFrames.find(
        (candidate) => candidate.sequenceNumber === selected.pose.sequence,
      );
      if (frame) drawLandmarks(replayCanvasRef.current, frame);
    }
  }

  async function confirmReplay() {
    if (!session || !continuousVideo || !replayReviewerCode.trim()) {
      setNotice("A replay reviewer code is required.");
      return;
    }
    if (!watchedReplay || !overlayConfirmed || !referenceConfirmed || !attemptConfirmed) {
      setNotice("Confirm the full continuous replay, overlay, and both marked intervals.");
      return;
    }
    const declaredAt = new Date().toISOString();
    if (!replayVideoRef.current) {
      setNotice("The local continuous replay is unavailable.");
      return;
    }
    let decodedVideo;
    try {
      decodedVideo = await observeDecodedContinuousVideo(replayVideoRef.current);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The local video did not decode.");
      return;
    }
    const integrityAudit = await auditLocalReplayIntegrity({
      declaredVideo: session.video,
      observedVideo: {
        fileName: session.video.fileName,
        blob: continuousVideo,
      },
      decodedVideo,
      videoFrames: session.videoFrameTimings,
      poseAttachments: buildResearchReplayAttachments(session.poseFrames),
      intervals: session.intervals,
      confirmation: {
        reviewerCode: replayReviewerCode.trim(),
        watchedContinuousVideo: watchedReplay,
        overlaysVisible: overlayConfirmed,
        referenceIntervalConfirmed: referenceConfirmed,
        attemptIntervalConfirmed: attemptConfirmed,
        declaredAt,
        evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
      },
    });
    if (!integrityAudit.valid) {
      setNotice(
        `Replay integrity failed: ${integrityAudit.issues
          .map((entry) => entry.message)
          .join(" ")}`,
      );
      return;
    }
    const audit: ResearchReplayAudit = {
      status: integrityAudit.valid ? "PASS" : "FAIL",
      videoAssetSha256: session.video.sha256,
      poseFrameCount: session.poseFrames.length,
      videoTimingCount: session.videoFrameTimings.length,
      intervalBoundaryCoverage: integrityAudit.boundaryCoverage.valid,
      causalOverlaySelection: integrityAudit.boundaryCoverage.valid,
      humanConfirmation: {
        reviewerCode: replayReviewerCode.trim(),
        watchedContinuousVideo: true,
        overlaysVisible: true,
        referenceIntervalConfirmed: true,
        attemptIntervalConfirmed: true,
        declaredAt,
        evidenceScope: REPLAY_INTEGRITY_EVIDENCE_SCOPE,
      },
      encodedReplayTimelineMapping: RECORDED_BLOB_TIMELINE_MAPPING,
      encodedReplayTimelineCalibrated: false,
      renderedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
      issues: integrityAudit.issues.map((entry) => entry.message),
      integrityAudit,
    };
    setReplayAudit(audit);
    setNotice("");
    setStep("annotate");
  }

  async function collectorAnnotation(): Promise<ResearchAnnotation> {
    if (!session || !frozenProtocol) throw new Error("The frozen session is missing.");
    const execution = session.protocolExecution;
    const executionSha = await sha256Text(canonicalJson(execution));
    const confidence = Number(collectorConfidence);
    if (!collectorRaterCode.trim()) throw new Error("Collector rater code is required.");
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      throw new Error("Collector confidence must be between 0 and 1.");
    }
    if (execution.status === "unobservable") {
      return {
        annotationId: createCode("annotation"),
        raterCode: collectorRaterCode.trim(),
        raterRole: "collector",
        reviewStatus: "draft-unreviewed",
        movementIdentity: frozenProtocol.protocol.movementFamily,
        assessment: "unobservable",
        evidenceJoints: [...frozenProtocol.protocol.target.evidenceJoints],
        desiredTargetSource: "unknown",
        confidence,
        abstainReason: execution.abstainReason ?? "The observable metric failed closed.",
        notes: collectorNotes || undefined,
        protocolSha256: frozenProtocol.sha256,
        protocolExecutionSha256: executionSha,
        createdAt: new Date().toISOString(),
      };
    }
    const delta = execution.targetMinusObservedDegrees ?? 0;
    const attemptTime = execution.evidence?.attemptSessionTimeMs;
    return {
      annotationId: createCode("annotation"),
      raterCode: collectorRaterCode.trim(),
      raterRole: "collector",
      reviewStatus: "draft-unreviewed",
      movementIdentity: frozenProtocol.protocol.movementFamily,
      assessment: "uncertain",
      realizedErrorType:
        "derived reference-relative estimator difference; practical significance uncalibrated",
      evidenceJoints: [...frozenProtocol.protocol.target.evidenceJoints],
      evidenceStartMs: attemptTime,
      evidenceEndMs: attemptTime,
      evidenceClock: "session-pose-callback-offset-ms",
      metric: frozenProtocol.protocol.target.metricId,
      observedValue: execution.attemptPeakDegrees,
      targetValue: execution.referencePeakDegrees,
      delta,
      unit: "degrees",
      desiredTargetSource: "captured-reference",
      cue: undefined,
      confidence,
      notes:
        collectorNotes ||
        "Numeric estimator difference retained descriptively; no acceptable/correction label or action recommendation is established.",
      protocolSha256: frozenProtocol.sha256,
      protocolExecutionSha256: executionSha,
      referenceEvidenceSequenceNumber:
        execution.evidence?.referenceSequenceNumber,
      attemptEvidenceSequenceNumber: execution.evidence?.attemptSequenceNumber,
      createdAt: new Date().toISOString(),
    };
  }

  async function prepareExport() {
    if (!session || !replayAudit || !continuousVideo || !frozenProtocol) return;
    setPreparing(true);
    setNotice("");
    try {
      const annotation = await collectorAnnotation();
      const draft: ResearchEpisodeDraft = {
        metadata: {
          participantCode,
          sessionCode,
          episodeCode,
          movementFamily: frozenProtocol.protocol.movementFamily,
          cameraView: frozenProtocol.protocol.cameraView,
          deviceNotes,
          calibrationNotes,
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
          confirmedAt: consentConfirmedAtRef.current,
        },
        protocol: frozenProtocol,
        session: { ...session, replayAudit },
        annotations: [annotation],
        environment: {
          collectorBuildId: RESEARCH_COLLECTOR_BUILD_ID,
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          language: navigator.language,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
          secureContext: window.isSecureContext,
          origin: window.location.origin,
          recordedAt: new Date().toISOString(),
        },
      };
      const nextExport = await buildResearchExport(draft);
      const nextArchive = await buildResearchArchive(nextExport, continuousVideo);
      setExportResult(nextExport);
      setArchive(nextArchive);
      setStep("export");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The episode is incomplete.");
    } finally {
      setPreparing(false);
    }
  }

  function downloadPackage() {
    if (!archive) return;
    downloadBlob(archive.blob, archive.fileName);
    downloadBlob(
      new Blob([archive.sha256Sidecar], { type: "text/plain" }),
      `${archive.fileName}.sha256`,
    );
  }

  const referenceDone = completedKinds.includes("reference");
  const attemptDone = completedKinds.includes("attempt");
  const execution = session?.protocolExecution;

  return (
    <main className="research-capture-screen">
      <header className="research-header">
        <Link href="/" className="research-back">
          <ArrowLeft size={19} /> Relay
        </Link>
        <div className="research-step-label">
          First-party evidence · {STEPS.indexOf(step) + 1} of {STEPS.length}
        </div>
      </header>

      <section className="research-hero">
        <div>
          <div className="flow-eyebrow">
            <FlaskConical size={17} /> Data engine v2
          </div>
          <h1>One stream. One clock. No invented correction truth.</h1>
          <p>
            Capture a same-person left-arm reference and attempt inside one
            uninterrupted local video. RAW pose, causal timing, and tagged
            analytic aids remain together for later matched decoder tests.
          </p>
        </div>
        <div className="research-boundary-card">
          <ShieldCheck size={22} />
          <div>
            <strong>Adult researcher self-capture only.</strong>
            <span>
              Internal methods validation—not medical advice, participant
              recruitment, clinical correctness, or training-ready data.
            </span>
          </div>
        </div>
      </section>

      <section className="research-workspace">
        <div className="research-progress" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
          {STEPS.map((item, index) => (
            <div key={item} className={index <= STEPS.indexOf(step) ? "active" : ""}>
              <span>{index + 1}</span>
              <small>{item}</small>
            </div>
          ))}
        </div>

        {step === "setup" && (
          <div className="research-panel">
            <div className="research-panel-heading">
              <div>
                <span>Frozen pilot</span>
                <h2>Reference-relative left lateral arm raise.</h2>
              </div>
              <LockKeyhole size={22} />
            </div>
            <p className="research-helper">
              Front view. Raise the left arm comfortably to the side for the
              reference, return to neutral, then make an attempt. The captured
              reference is only a comparator; it is not assumed ideal or safe.
            </p>
            <div className="research-form-grid three">
              <label>
                Anonymous participant code
                <input value={participantCode} onChange={(event) => setParticipantCode(event.target.value)} placeholder="SELF-P001" />
              </label>
              <label>
                Session code
                <input value={sessionCode} onChange={(event) => setSessionCode(event.target.value)} />
              </label>
              <label>
                Episode code
                <input value={episodeCode} onChange={(event) => setEpisodeCode(event.target.value)} />
              </label>
            </div>
            <div className="research-form-grid two" style={{ marginTop: 16 }}>
              <label>
                Device notes
                <input value={deviceNotes} onChange={(event) => setDeviceNotes(event.target.value)} placeholder="Phone or laptop model/browser" />
              </label>
              <label>
                Camera calibration notes
                <input value={calibrationNotes} onChange={(event) => setCalibrationNotes(event.target.value)} placeholder="Front view, distance, fixed placement" />
              </label>
            </div>
            <div className="research-consent-block">
              <label>
                <input type="checkbox" checked={adultSelfConfirmed} onChange={(event) => setAdultSelfConfirmed(event.target.checked)} />
                I am an adult researcher recording only myself for this limited methods pilot.
              </label>
              <label>
                <input type="checkbox" checked={rightsConfirmed} onChange={(event) => setRightsConfirmed(event.target.checked)} />
                I have the right to record these signals and will keep the export internal.
              </label>
              <label>
                <input type="checkbox" checked={boundaryConfirmed} onChange={(event) => setBoundaryConfirmed(event.target.checked)} />
                I understand this cannot infer pain, load, force, safety, diagnosis, or clinical need.
              </label>
            </div>
            <button type="button" className="primary-action" onClick={startCamera}>
              <Camera size={18} /> Start local camera
            </button>
          </div>
        )}

        {(step === "setup" || step === "capture" || step === "replay") && (
          <div className="research-camera-strip">
            <div className="research-camera front">
              {step !== "replay" ? (
                <>
                  <video ref={liveVideoRef} playsInline muted />
                  <canvas ref={liveCanvasRef} aria-hidden="true" />
                </>
              ) : replayUrl ? (
                <>
                  <video
                    ref={replayVideoRef}
                    src={replayUrl}
                    playsInline
                    controls
                    onTimeUpdate={handleReplayTime}
                    onSeeked={handleReplayTime}
                    onEnded={() => setWatchedReplay(true)}
                  />
                  <canvas ref={replayCanvasRef} aria-hidden="true" />
                </>
              ) : null}
              <div className="research-camera-status">
                <span className={sessionStatus === "recording" || step === "replay" ? "live" : ""} />
                {step === "replay"
                  ? "Local continuous replay"
                  : openKind
                    ? `Marking ${openKind}`
                    : sessionStatus === "recording"
                      ? "Continuous episode recording"
                      : cameraStatus.label}
              </div>
            </div>
            <div className="research-signal-contract">
              <strong>Signals retained</strong>
              <span>Continuous image + world pose and confidence</span>
              <span>Video-frame callback timing and interval markers</span>
              <span>One continuous local RGB video</span>
              <span>Deterministic RAW+AIDS from identical raw frames</span>
              <small>Force, load, contact, pain, intent and clinical truth are absent.</small>
            </div>
          </div>
        )}

        {step === "capture" && (
          <div className="research-panel">
            <div className="research-panel-heading">
              <div>
                <span>Continuous episode</span>
                <h2>Keep neutral time and transitions.</h2>
              </div>
              <Video size={22} />
            </div>
            {sessionStatus === "idle" && (
              <>
                <p className="research-helper">
                  Start once, stand neutral, mark the reference, return to neutral,
                  mark the attempt, return to neutral, then finish. The camera stops
                  before replay and annotation.
                </p>
                <div className="research-actions">
                  <button type="button" className="primary-action" disabled={!latestFrame} onClick={beginContinuousSession}>
                    <Play size={17} /> Begin uninterrupted episode
                  </button>
                  <button type="button" className="secondary-action" onClick={cancelCameraAndReset}>
                    Stop camera
                  </button>
                </div>
              </>
            )}
            {sessionStatus === "starting" && <p className="research-helper">Starting on the next exact video-frame callback…</p>}
            {(sessionStatus === "recording" || sessionStatus === "finishing") && (
              <>
                <div className="research-record-grid">
                  {(["reference", "attempt"] as const).map((kind) => {
                    const done = kind === "reference" ? referenceDone : attemptDone;
                    const active = openKind === kind;
                    const blocked =
                      sessionStatus !== "recording" ||
                      Boolean(openKind && !active) ||
                      (kind === "attempt" && !referenceDone) ||
                      done;
                    return (
                      <article key={kind} className={`research-record-card ${done ? "complete" : ""}`}>
                        <div className="research-record-heading">
                          <span className="research-record-number">{done ? <CheckCircle2 size={18} /> : kind === "reference" ? "1" : "2"}</span>
                          <div>
                            <strong>{kind === "reference" ? "Captured comparator" : "Observed attempt"}</strong>
                            <small>{kind === "reference" ? "Comfortable left-arm raise; not assumed ideal." : "Same person, camera and protocol."}</small>
                          </div>
                        </div>
                        <button type="button" className={active ? "capture-stop" : "secondary-action"} disabled={blocked} onClick={() => toggleInterval(kind)}>
                          {active ? <Square size={17} fill="currentColor" /> : <Play size={17} />}
                          {active ? `End ${kind}` : `Mark ${kind}`}
                        </button>
                      </article>
                    );
                  })}
                </div>
                <div className="research-actions">
                  <button type="button" className="primary-action" disabled={!referenceDone || !attemptDone || Boolean(openKind) || sessionStatus !== "recording"} onClick={finishContinuousSession}>
                    <Square size={17} /> Finish after neutral return
                  </button>
                  <button type="button" className="secondary-action" onClick={cancelCameraAndReset}>
                    Cancel and stop camera
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {step === "replay" && session && (
          <div className="research-panel">
            <div className="research-panel-heading">
              <div>
                <span>Replay gate</span>
                <h2>Watch the evidence before describing it.</h2>
              </div>
              <Play size={22} />
            </div>
            <p className="research-helper">
              Play the full uninterrupted video. Source-clock pose attachment is
              causal, but MediaRecorder&apos;s encoded time zero is not yet calibrated
              to that source clock. The visible markers and overlay are therefore
              approximate replay aids, not exact boundary evidence or anatomical truth.
            </p>
            <div className="research-record-grid">
              {session.intervals.map((interval) => {
                const active =
                  replayMediaTimeMs >= interval.begin.mediaTimeMs &&
                  replayMediaTimeMs <= interval.end.mediaTimeMs;
                return (
                  <article key={interval.kind} className={`research-record-card ${active ? "complete" : ""}`}>
                    <strong>{interval.kind}</strong>
                    <small>
                      Approx. {(interval.begin.mediaTimeMs / 1000).toFixed(2)}–{(interval.end.mediaTimeMs / 1000).toFixed(2)} s
                    </small>
                    <button
                      type="button"
                      className="secondary-action"
                      onClick={() => {
                        if (!replayVideoRef.current) return;
                        replayVideoRef.current.currentTime = interval.begin.mediaTimeMs / 1000;
                      }}
                    >
                      Seek to marker
                    </button>
                  </article>
                );
              })}
            </div>
            <div className="research-form-grid two">
              <label>
                Replay reviewer code
                <input value={replayReviewerCode} onChange={(event) => setReplayReviewerCode(event.target.value)} placeholder="R-REPLAY-01" />
              </label>
              <label>
                Timing summary
                <input readOnly value={`${session.poseFrames.length} pose frames · ${session.videoFrameTimings.length} video timings`} />
              </label>
            </div>
            <div className="research-consent-block">
              <label>
                <input type="checkbox" checked={watchedReplay} readOnly disabled />
                Full playback reached the end in this browser.
              </label>
              {[
                [overlayConfirmed, setOverlayConfirmed, "The approximate pose overlay remained visible during replay."],
                [referenceConfirmed, setReferenceConfirmed, "The approximate reference marker visually covers the intended reference movement."],
                [attemptConfirmed, setAttemptConfirmed, "The approximate attempt marker visually covers the intended attempt."],
              ].map(([checked, setter, label]) => (
                <label key={String(label)}>
                  <input type="checkbox" checked={checked as boolean} onChange={(event) => (setter as (value: boolean) => void)(event.target.checked)} />
                  {String(label)}
                </label>
              ))}
            </div>
            <button type="button" className="primary-action" onClick={confirmReplay}>
              <CheckCircle2 size={18} /> Bind replay confirmation
            </button>
          </div>
        )}

        {step === "annotate" && session && frozenProtocol && (
          <div className="research-panel">
            <div className="research-panel-heading">
              <div>
                <span>Deterministic observation</span>
                <h2>Bind the collector note to computed evidence.</h2>
              </div>
              <FlaskConical size={22} />
            </div>
            <div className="research-record-grid">
              <article className="research-record-card complete">
                <strong>Captured reference peak</strong>
                <div className="research-record-stats"><span>{formatDegrees(execution?.referencePeakDegrees)}</span></div>
              </article>
              <article className="research-record-card complete">
                <strong>Observed attempt peak</strong>
                <div className="research-record-stats"><span>{formatDegrees(execution?.attemptPeakDegrees)}</span><span>Δ {formatDegrees(execution?.targetMinusObservedDegrees)}</span></div>
              </article>
            </div>
            <p className="research-helper">
              Status: <strong>{execution?.status}</strong>. The metric is a front-view
              MediaPipe image-angle estimate with sensitivity checks at 0.55, 0.65,
              and 0.75 confidence. It is not an anatomical or clinical angle.
            </p>
            <div className="research-form-grid two">
              <label>
                Collector rater code
                <input value={collectorRaterCode} onChange={(event) => setCollectorRaterCode(event.target.value)} placeholder="R-COLLECTOR-01" />
              </label>
              <label>
                Collector confidence (0–1)
                <input type="number" min="0" max="1" step="0.05" value={collectorConfidence} onChange={(event) => setCollectorConfidence(event.target.value)} />
              </label>
            </div>
            <label className="research-wide-field">
              Provenance notes (optional)
              <textarea value={collectorNotes} onChange={(event) => setCollectorNotes(event.target.value)} placeholder="Track loss, camera movement, unexpected interruption, or why this should remain unobservable." />
            </label>
            <div className="research-review-warning">
              <ShieldCheck size={20} />
              <span>
                This collector record is not consensus. The package remains
                quarantined until a separate blinded reviewer route appends an
                independent annotation and preserves any disagreement.
              </span>
            </div>
            <button type="button" className="primary-action" disabled={preparing} onClick={prepareExport}>
              {preparing ? "Closing immutable package…" : "Build quarantined evidence package"}
            </button>
          </div>
        )}

        {step === "export" && exportResult && archive && (
          <div className="research-panel research-export-panel">
            <div className="research-export-icon"><Download size={28} /></div>
            <span>Quarantined package ready</span>
            <h2>RAW and RAW+AIDS now share one immutable ancestor.</h2>
            <p>
              The archive contains the continuous video, raw pose, causal timing,
              interval markers, frozen protocol, deterministic aids, computed
              reference comparison, replay audit, environment, annotation, and
              member checksums. It is still not training-ready.
            </p>
            <div className="research-hash-card">
              <small>Archive SHA-256</small>
              <code>{archive.sha256}</code>
            </div>
            <div className="research-hash-card">
              <small>Episode JSON SHA-256</small>
              <code>{exportResult.sha256}</code>
            </div>
            <div className="research-export-actions">
              <button type="button" className="primary-action" onClick={downloadPackage}>
                <Download size={18} /> Download package + outer checksum
              </button>
              <Link href="/research-review" className="secondary-action">
                Open blinded review tool
              </Link>
            </div>
          </div>
        )}

        {notice && <div className="research-notice">{notice}</div>}
      </section>
    </main>
  );
}
