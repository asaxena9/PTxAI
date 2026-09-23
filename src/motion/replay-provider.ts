"use client";

import type {
  FrameHandler,
  MotionProvider,
  StatusHandler,
} from "./types";
import type { PoseFrame } from "./types";

export class ReplayMotionProvider implements MotionProvider {
  readonly id = "replay" as const;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private frames: PoseFrame[];
  private speed: number;
  private index = 0;
  private running = false;

  constructor(frames: PoseFrame[], speed = 1) {
    this.frames = frames;
    this.speed = Math.max(0.25, speed);
  }

  async initialize(): Promise<void> {
    return Promise.resolve();
  }

  async start(
    _video: HTMLVideoElement | null,
    onFrame: FrameHandler,
    onStatus?: StatusHandler,
  ): Promise<void> {
    await this.stop();
    this.running = true;
    onStatus?.({
      state: "running",
      label: "Guided replay · simulated landmarks",
    });
    const play = () => {
      if (!this.running) return;
      const frame = this.frames[this.index];
      if (!frame) {
        this.running = false;
        onStatus?.({ state: "ready", label: "Replay complete" });
        return;
      }
      onFrame(frame);
      if (!this.running) return;
      const next = this.frames[this.index + 1];
      const delay = next
        ? Math.max(8, (next.timestampMs - frame.timestampMs) / this.speed)
        : 0;
      this.index += 1;
      this.timer = setTimeout(play, delay);
    };
    play();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }
}
