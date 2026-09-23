import { describe, expect, it } from "vitest";
import { createDemoSession } from "../src/domain/templates";
import type { ExercisePrescription } from "../src/domain/types";
import { ExerciseEngine } from "../src/motion/exercise-engine";
import {
  createReplayFrames,
  type ReplayScenarioId,
} from "../src/motion/replay-fixtures";
import type { EngineOutput, MotionEvent } from "../src/motion/types";

function run(
  exercise: ExercisePrescription,
  scenario: ReplayScenarioId = "correct",
  reps = 1,
): { output: EngineOutput; events: MotionEvent[] } {
  const engine = new ExerciseEngine(exercise);
  const events: MotionEvent[] = [];
  let output: EngineOutput | undefined;
  for (const frame of createReplayFrames({
    exerciseId: exercise.exerciseId,
    side: exercise.side,
    scenario,
    reps,
    holdSeconds: exercise.dose.holdSeconds,
    intervalMs: 100,
  })) {
    output = engine.process(frame);
    events.push(...output.events);
  }
  if (!output) throw new Error("Fixture returned no frames.");
  return { output, events };
}

describe("shared live/replay movement pipeline", () => {
  const session = createDemoSession();

  it.each([
    ["heel slide", 0],
    ["straight-leg raise", 1],
    ["seated knee extension", 2],
  ])("counts a complete %s only after return", (_label, index) => {
    const { output } = run(session.exercises[index], "correct");
    expect(output.repsCompleted).toBe(1);
    expect(output.repRecords[0].trackingGoodRatio).toBeGreaterThan(0.95);
  });

  it("does not count an incomplete heel-slide return", () => {
    const { output } = run(session.exercises[0], "incomplete-return");
    expect(output.repsCompleted).toBe(0);
  });

  it("separates target and maximum events", () => {
    const { events } = run(session.exercises[0], "exceed-limit");
    expect(events.some((item) => item.type === "target-reached")).toBe(true);
    expect(
      events.some((item) => item.type === "restriction-approaching"),
    ).toBe(true);
    expect(events.some((item) => item.type === "restriction-reached")).toBe(
      true,
    );
  });

  it("abstains from angle rules when image and inferred 3D disagree", () => {
    const { output, events } = run(
      session.exercises[0],
      "uncertain-angle",
    );
    expect(
      events.some(
        (item) =>
          item.type === "restriction-approaching" ||
          item.type === "restriction-reached",
      ),
    ).toBe(false);
    expect(output.measurements.confidence.canAssessAngle).toBe(true);
    expect(output.measurements.confidence.canAssessBodyFeatures).toBe(true);
    expect(
      events.some((item) => item.type === "tracking-lost"),
    ).toBe(true);
  });

  it("records a maximum estimate for every assessable heel-slide rep", () => {
    const { output } = run(session.exercises[0], "correct", 3);
    expect(output.repRecords).toHaveLength(3);
    expect(
      output.repRecords.every(
        (record) =>
          record.maxKneeFlexion !== undefined &&
          record.maxKneeFlexion > 60,
      ),
    ).toBe(true);
  });

  it("records provider-extensible body-feature peaks with coverage", () => {
    const { output } = run(session.exercises[1], "correct");
    const metrics = output.repRecords[0].featurePeaks.map(
      (peak) => peak.metric,
    );
    expect(metrics).toContain("leg-lift");
    expect(metrics).toContain("knee-bend");
    expect(metrics).toContain("trunk-compensation");
    expect(
      output.repRecords[0].featurePeaks.every(
        (peak) => peak.assessableCoverage > 0.9,
      ),
    ).toBe(true);
  });

  it("requires a stable 1.5-second neutral baseline before counting", () => {
    const exercise = session.exercises[0];
    const engine = new ExerciseEngine(exercise);
    const frames = createReplayFrames({
      exerciseId: exercise.exerciseId,
      side: exercise.side,
      reps: 1,
      holdSeconds: exercise.dose.holdSeconds,
      intervalMs: 100,
    });
    let output = engine.process(frames[0]);
    for (const frame of frames.slice(1, 12)) output = engine.process(frame);
    expect(output.repsCompleted).toBe(0);
    expect(output.coachMessage).toContain("stable baseline");
  });

  it("uses confident temporal evidence only to corroborate a plausible hold", () => {
    const exercise = session.exercises[0];
    const engine = new ExerciseEngine(exercise);
    const frames = createReplayFrames({
      exerciseId: exercise.exerciseId,
      side: exercise.side,
      reps: 1,
      holdSeconds: exercise.dose.holdSeconds,
      intervalMs: 100,
    });
    const events: MotionEvent[] = [];
    for (const [index, frame] of frames.slice(0, 28).entries()) {
      const output = engine.process(
        frame,
        index === 27
          ? {
              phase: "holding",
              confidence: 0.9,
              margin: 0.5,
              applicable: true,
              modelId: "test-temporal",
            }
          : undefined,
      );
      events.push(...output.events);
    }
    expect(events.some((event) => event.type === "hold-started")).toBe(true);
  });

  it("preserves body-feature scopes across exercises", () => {
    const heel = run(session.exercises[0]).output.measurements.features;
    const raise = run(session.exercises[1]).output.measurements.features;
    const extension = run(session.exercises[2]).output.measurements.features;

    expect(heel.find((feature) => feature.id === "knee-flexion")).toBeDefined();
    expect(heel.find((feature) => feature.metric === "knee-flexion")).toBeDefined();
    expect(
      raise.find((feature) => feature.id === "leg-chain-lift"),
    ).toBeDefined();
    expect(
      raise.find((feature) => feature.id === "shoulder-hip-stability"),
    ).toBeDefined();
    expect(
      extension.find((feature) => feature.id === "extension-progress"),
    ).toBeDefined();
  });
});
