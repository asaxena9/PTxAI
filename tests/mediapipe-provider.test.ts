import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { createFromOptions } = vi.hoisted(() => ({
  createFromOptions: vi.fn(),
}));

vi.mock("@mediapipe/tasks-vision", () => ({
  FilesetResolver: {
    forVisionTasks: vi.fn().mockResolvedValue({}),
  },
  PoseLandmarker: {
    createFromOptions,
  },
}));

import { MediaPipeMotionProvider } from "../src/motion/mediapipe-provider";
import type { MotionProviderStatus } from "../src/motion/types";

function cameraFixture() {
  const events: string[] = [];
  const track = {
    stop: vi.fn(() => events.push("track-stop")),
  };
  const stream = {
    getTracks: vi.fn(() => [track]),
  } as unknown as MediaStream;
  const video = document.createElement("video");
  Object.defineProperty(video, "readyState", {
    configurable: true,
    value: 2,
  });
  Object.defineProperty(video, "videoWidth", {
    configurable: true,
    value: 1280,
  });
  Object.defineProperty(video, "videoHeight", {
    configurable: true,
    value: 720,
  });
  return { events, stream, track, video };
}

describe("MediaPipeMotionProvider camera lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn() },
    });
    Object.defineProperty(window, "requestAnimationFrame", {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });
    Object.defineProperty(window, "cancelAnimationFrame", {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stops an acquired camera track before reporting a startup error", async () => {
    const { events, stream, track, video } = cameraFixture();
    createFromOptions.mockResolvedValue({
      detectForVideo: vi.fn(),
      close: vi.fn(),
    });
    vi.spyOn(navigator.mediaDevices, "getUserMedia").mockResolvedValue(stream);
    vi.spyOn(video, "play").mockRejectedValue(new Error("playback failed"));
    const statuses: MotionProviderStatus[] = [];
    const provider = new MediaPipeMotionProvider();

    await expect(
      provider.start(video, vi.fn(), (status) => {
        statuses.push(status);
        if (status.state === "error") events.push("status-error");
      }),
    ).rejects.toThrow("playback failed");

    expect(track.stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
    expect(events).toEqual(["track-stop", "status-error"]);
    expect(statuses.at(-1)?.state).toBe("error");
  });

  it("stops tracks and pending callbacks before reporting an inference error", async () => {
    const { events, stream, track, video } = cameraFixture();
    const detectForVideo = vi
      .fn()
      .mockReturnValueOnce({ landmarks: [], worldLandmarks: [] })
      .mockImplementationOnce(() => {
        throw new Error("inference failed");
      });
    createFromOptions.mockResolvedValue({
      detectForVideo,
      close: vi.fn(),
    });
    vi.spyOn(navigator.mediaDevices, "getUserMedia").mockResolvedValue(stream);
    vi.spyOn(video, "play").mockResolvedValue();
    vi.spyOn(performance, "now")
      .mockReturnValueOnce(100)
      .mockReturnValueOnce(200);
    let scheduled: FrameRequestCallback | undefined;
    const requestFrame = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback) => {
        scheduled = callback;
        return 41;
      });
    const cancelFrame = vi
      .spyOn(window, "cancelAnimationFrame")
      .mockImplementation(() => events.push("callback-cancel"));
    const statuses: MotionProviderStatus[] = [];
    const provider = new MediaPipeMotionProvider();

    await provider.start(video, vi.fn(), (status) => {
      statuses.push(status);
      if (status.state === "error") events.push("status-error");
    });
    expect(requestFrame).toHaveBeenCalledOnce();
    expect(scheduled).toBeDefined();

    scheduled?.(200);

    expect(cancelFrame).toHaveBeenCalledWith(41);
    expect(track.stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
    expect(events).toEqual([
      "callback-cancel",
      "track-stop",
      "status-error",
    ]);
    expect(statuses.at(-1)).toMatchObject({
      state: "error",
      label: "Movement processing paused",
      detail: "inference failed",
    });
  });

  it("rejects start when the first inference fails", async () => {
    const { events, stream, track, video } = cameraFixture();
    createFromOptions.mockResolvedValue({
      detectForVideo: vi.fn(() => {
        throw new Error("first inference failed");
      }),
      close: vi.fn(),
    });
    vi.spyOn(navigator.mediaDevices, "getUserMedia").mockResolvedValue(stream);
    vi.spyOn(video, "play").mockResolvedValue();
    vi.spyOn(performance, "now").mockReturnValue(100);
    const statuses: MotionProviderStatus[] = [];
    const provider = new MediaPipeMotionProvider();

    await expect(
      provider.start(video, vi.fn(), (status) => {
        statuses.push(status);
        if (status.state === "error") events.push("status-error");
      }),
    ).rejects.toThrow("first inference failed");

    expect(track.stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
    expect(window.requestAnimationFrame).not.toHaveBeenCalled();
    expect(statuses.at(-1)).toMatchObject({ state: "error" });
    expect(events).toEqual(["track-stop", "status-error"]);
  });

  it("keeps explicit stop idempotent and releases the live camera", async () => {
    const { stream, track, video } = cameraFixture();
    createFromOptions.mockResolvedValue({
      detectForVideo: vi.fn(() => ({ landmarks: [], worldLandmarks: [] })),
      close: vi.fn(),
    });
    vi.spyOn(navigator.mediaDevices, "getUserMedia").mockResolvedValue(stream);
    vi.spyOn(video, "play").mockResolvedValue();
    vi.spyOn(window, "requestAnimationFrame").mockReturnValue(17);
    const cancelFrame = vi.spyOn(window, "cancelAnimationFrame");
    const provider = new MediaPipeMotionProvider();

    await provider.start(video, vi.fn());
    await provider.stop();
    await provider.stop();

    expect(cancelFrame).toHaveBeenCalledTimes(1);
    expect(cancelFrame).toHaveBeenCalledWith(17);
    expect(track.stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
    expect(provider.getStatus()).toEqual({
      state: "ready",
      label: "Camera paused",
    });
  });
});
