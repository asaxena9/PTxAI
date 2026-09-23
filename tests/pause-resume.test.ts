import { afterEach, describe, expect, it, vi } from "vitest";
import { createDemoSession } from "../src/domain/templates";
import { ExerciseEngine } from "../src/motion/exercise-engine";
import { ReplayMotionProvider } from "../src/motion/replay-provider";
import { createReplayFrames } from "../src/motion/replay-fixtures";
import type { EngineOutput, MotionEvent } from "../src/motion/types";

afterEach(() => {
  vi.useRealTimers();
});

describe("pause and resume", () => {
  it("resumes replay at the next frame instead of restarting it", async () => {
    vi.useFakeTimers();
    const frames = createReplayFrames({
      exerciseId: "heel-slide",
      side: "right",
      reps: 1,
      holdSeconds: 0,
      intervalMs: 100,
    }).slice(0, 4);
    const provider = new ReplayMotionProvider(frames, 1);
    const seen: number[] = [];

    await provider.start(null, (frame) => seen.push(frame.timestampMs));
    await provider.stop();
    await provider.start(null, (frame) => seen.push(frame.timestampMs));
    await provider.stop();

    expect(seen).toEqual([frames[0].timestampMs, frames[1].timestampMs]);
  });

  it("does not credit a long user pause toward hold or tempo", () => {
    const exercise = createDemoSession().exercises[0];
    const engine = new ExerciseEngine(exercise);
    const frames = createReplayFrames({
      exerciseId: exercise.exerciseId,
      side: exercise.side,
      reps: 1,
      holdSeconds: 0,
      intervalMs: 100,
    });
    const events: MotionEvent[] = [];
    let output: EngineOutput | undefined;
    let paused = false;
    let offset = 0;

    for (const sourceFrame of frames) {
      const frame = {
        ...sourceFrame,
        timestampMs: sourceFrame.timestampMs + offset,
      };
      output = engine.process(frame);
      events.push(...output.events);
      if (!paused && output.phase === "holding") {
        engine.pause();
        paused = true;
        offset = 10000;
      }
    }

    expect(output?.repsCompleted).toBe(1);
    expect(output?.repRecords[0].holdMs ?? Infinity).toBeLessThan(500);
    expect(events.some((event) => event.type === "hold-short")).toBe(true);
    expect(events.some((event) => event.type === "hold-met")).toBe(false);
  });
});
