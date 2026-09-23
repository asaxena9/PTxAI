import { describe, expect, it } from "vitest";
import { createDemoSession } from "../src/domain/templates";
import { ExerciseEngine } from "../src/motion/exercise-engine";
import {
  createReplayFrames,
  REPLAY_SCENARIOS,
  type ReplayScenarioId,
} from "../src/motion/replay-fixtures";
import type {
  EngineOutput,
  MotionEvent,
} from "../src/motion/types";

function replay(
  exerciseIndex: number,
  scenario: ReplayScenarioId,
): { output: EngineOutput; events: MotionEvent[] } {
  const exercise = createDemoSession().exercises[exerciseIndex];
  const engine = new ExerciseEngine(exercise);
  const events: MotionEvent[] = [];
  let output: EngineOutput | undefined;
  for (const frame of createReplayFrames({
    exerciseId: exercise.exerciseId,
    side: exercise.side,
    scenario,
    reps: 1,
    holdSeconds: exercise.dose.holdSeconds,
    intervalMs: 100,
  })) {
    output = engine.process(frame);
    events.push(...output.events);
  }
  if (!output) throw new Error("Replay emitted no frames.");
  return { output, events };
}

describe("repeatable movement replay matrix", () => {
  it("publishes every required failure and timing scenario", () => {
    const ids = REPLAY_SCENARIOS.map((scenario) => scenario.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toEqual(
      expect.arrayContaining([
        "correct",
        "incomplete-range",
        "incomplete-return",
        "fast",
        "slow",
        "tracking-loss",
        "occlusion",
        "poor-camera",
        "posture-variation",
        "approach-limit",
        "exceed-limit",
        "uncertain-angle",
      ]),
    );
  });

  it("separates approaching from reaching a hard maximum", () => {
    const approach = replay(0, "approach-limit").events;
    const exceed = replay(0, "exceed-limit").events;
    expect(
      approach.some((event) => event.type === "restriction-approaching"),
    ).toBe(true);
    expect(
      approach.some((event) => event.type === "restriction-reached"),
    ).toBe(false);
    expect(
      exceed.some((event) => event.type === "restriction-reached"),
    ).toBe(true);
  });

  it("distinguishes insufficient movement from failure to return", () => {
    const short = replay(0, "incomplete-range");
    const noReturn = replay(0, "incomplete-return");
    expect(short.output.repsCompleted).toBe(0);
    expect(noReturn.output.repsCompleted).toBe(0);
    expect(
      noReturn.events.some((event) => event.type === "incomplete-rep"),
    ).toBe(true);
  });

  it("evaluates configured tempo from observed movement time", () => {
    const fast = replay(0, "fast").events;
    const slow = replay(0, "slow").events;
    expect(fast.some((event) => event.type === "tempo-missed")).toBe(true);
    expect(slow.some((event) => event.type === "tempo-met")).toBe(true);
  });

  it.each(["tracking-loss", "occlusion"] as const)(
    "does not credit unobserved time toward a hold during %s",
    (scenario) => {
      const { output, events } = replay(0, scenario);
      expect(events.some((event) => event.type === "tracking-lost")).toBe(true);
      expect(events.some((event) => event.type === "hold-short")).toBe(true);
      expect(output.repRecords[0]?.holdMs ?? 0).toBeLessThan(2000);
    },
  );

  it("does not count movement from an unusable camera position", () => {
    const { output, events } = replay(0, "poor-camera");
    expect(output.repsCompleted).toBe(0);
    expect(events.some((event) => event.type === "tracking-lost")).toBe(true);
  });

  it("observes broad visible posture changes only for good tracking", () => {
    const { events } = replay(1, "posture-variation");
    expect(
      events.some((event) => event.type === "posture-observation"),
    ).toBe(true);
  });

  it("records the seated extension target independently of rep counting", () => {
    const { output, events } = replay(2, "correct");
    expect(output.repsCompleted).toBe(1);
    expect(events.some((event) => event.type === "target-reached")).toBe(true);
  });
});
