"use client";

import type {
  FrameHandler,
  JointName,
  LandmarkSet,
  MotionProvider,
  MotionProviderStatus,
  PoseLandmark,
  StatusHandler,
} from "./types";

const LANDMARK_INDEX: Partial<Record<number, JointName>> = {
  0: "nose",
  1: "left-eye-inner",
  2: "left-eye",
  3: "left-eye-outer",
  4: "right-eye-inner",
  5: "right-eye",
  6: "right-eye-outer",
  7: "left-ear",
  8: "right-ear",
  9: "mouth-left",
  10: "mouth-right",
  11: "left-shoulder",
  12: "right-shoulder",
  13: "left-elbow",
  14: "right-elbow",
  15: "left-wrist",
  16: "right-wrist",
  17: "left-pinky",
  18: "right-pinky",
  19: "left-index",
  20: "right-index",
  21: "left-thumb",
  22: "right-thumb",
  23: "left-hip",
  24: "right-hip",
  25: "left-knee",
  26: "right-knee",
  27: "left-ankle",
  28: "right-ankle",
  29: "left-heel",
  30: "right-heel",
  31: "left-foot-index",
  32: "right-foot-index",
};

interface MediaPipeResult {
  landmarks: Array<
    Array<{
      x: number;
      y: number;
      z: number;
      visibility?: number;
      presence?: number;
    }>
  >;
  worldLandmarks: Array<
    Array<{
      x: number;
      y: number;
      z: number;
      visibility?: number;
      presence?: number;
    }>
  >;
}

interface PoseLandmarkerLike {
  detectForVideo(video: HTMLVideoElement, timestamp: number): MediaPipeResult;
  close(): void;
}

export type CameraFacing = "user" | "environment";

function toSet(
  landmarks:
    | MediaPipeResult["landmarks"][number]
    | MediaPipeResult["worldLandmarks"][number]
    | undefined,
): LandmarkSet {
  const set: LandmarkSet = {};
  if (!landmarks) return set;
  landmarks.forEach((landmark, index) => {
    const name = LANDMARK_INDEX[index];
    if (!name) return;
    set[name] = {
      x: landmark.x,
      y: landmark.y,
      z: landmark.z,
      visibility: landmark.visibility ?? 0,
      presence: landmark.presence ?? landmark.visibility ?? 0,
    } satisfies PoseLandmark;
  });
  return set;
}

export class MediaPipeMotionProvider implements MotionProvider {
  readonly id = "mediapipe" as const;
  private landmarker: PoseLandmarkerLike | null = null;
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private animationFrame = 0;
  private running = false;
  private lastInferenceAt = -Infinity;
  private status: MotionProviderStatus = {
    state: "idle",
    label: "Camera not started",
  };

  constructor(private readonly facingMode: CameraFacing = "user") {}

  getStatus(): MotionProviderStatus {
    return this.status;
  }

  async initialize(): Promise<void> {
    if (this.landmarker) return;
    if (
      typeof window === "undefined" ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      this.status = {
        state: "unsupported",
        label: "Camera processing is unavailable",
        detail: "Use a current mobile browser or the guided replay.",
      };
      throw new Error(this.status.detail);
    }
    this.status = {
      state: "loading",
      label: "Loading on-device movement model",
    };
    const { FilesetResolver, PoseLandmarker } = await import(
      "@mediapipe/tasks-vision"
    );
    const vision = await FilesetResolver.forVisionTasks("/mediapipe");
    const options = {
      baseOptions: {
        modelAssetPath: "/models/pose_landmarker_full.task",
        delegate: "GPU" as const,
      },
      runningMode: "VIDEO" as const,
      numPoses: 1,
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
      outputSegmentationMasks: false,
    };
    try {
      this.landmarker = (await PoseLandmarker.createFromOptions(
        vision,
        options,
      )) as unknown as PoseLandmarkerLike;
    } catch {
      this.landmarker = (await PoseLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: {
          modelAssetPath: "/models/pose_landmarker_full.task",
          delegate: "CPU",
        },
      })) as unknown as PoseLandmarkerLike;
    }
    this.status = {
      state: "ready",
      label: "On-device movement model ready",
    };
  }

  async start(
    video: HTMLVideoElement | null,
    onFrame: FrameHandler,
    onStatus?: StatusHandler,
  ): Promise<void> {
    if (!video) throw new Error("A video element is required for live motion.");
    this.video = video;
    let initialInferenceErrorReported = false;
    try {
      await this.initialize();
      onStatus?.(this.status);
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: this.facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 24, max: 30 },
        },
        audio: false,
      });
      video.srcObject = this.stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();
      this.running = true;
      this.status = {
        state: "running",
        label: "Camera processing on this device",
      };
      onStatus?.(this.status);

      let firstInference = true;
      const tick = () => {
        if (!this.running || !this.landmarker) return;
        const now = performance.now();
        if (video.readyState >= 2 && now - this.lastInferenceAt >= 66) {
          this.lastInferenceAt = now;
          const initialInference = firstInference;
          firstInference = false;
          try {
            const result = this.landmarker.detectForVideo(video, now);
            const image = toSet(result.landmarks[0]);
            const world = toSet(result.worldLandmarks[0]);
            onFrame({
              timestampMs: now,
              image,
              world,
              frameWidth: video.videoWidth || 1280,
              frameHeight: video.videoHeight || 720,
              source: "live",
            });
          } catch (error) {
            this.releaseCameraResources();
            this.status = {
              state: "error",
              label: "Movement processing paused",
              detail:
                error instanceof Error
                  ? error.message
                  : "The on-device pose model could not process this frame.",
            };
            onStatus?.(this.status);
            if (initialInference) {
              initialInferenceErrorReported = true;
              throw error;
            }
            return;
          }
        }
        this.animationFrame = requestAnimationFrame(tick);
      };
      tick();
    } catch (error) {
      this.releaseCameraResources();
      if (initialInferenceErrorReported) throw error;
      const denied =
        error instanceof DOMException &&
        (error.name === "NotAllowedError" ||
          error.name === "PermissionDeniedError");
      this.status = denied
        ? {
            state: "permission-denied",
            label: "Camera access is off",
            detail:
              "Allow camera access in your browser, then run the check again.",
          }
        : {
            state: "error",
            label: "Camera check could not start",
            detail:
              error instanceof Error
                ? error.message
                : "Use the guided replay while you adjust camera access.",
          };
      onStatus?.(this.status);
      throw error;
    }
  }

  private releaseCameraResources(): void {
    this.running = false;
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = 0;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    this.video = null;
  }

  async stop(): Promise<void> {
    this.releaseCameraResources();
    this.status = { state: "ready", label: "Camera paused" };
  }

  dispose(): void {
    void this.stop();
    this.landmarker?.close();
    this.landmarker = null;
  }
}
