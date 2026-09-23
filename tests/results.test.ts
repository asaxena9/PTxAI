import { describe, expect, it } from "vitest";
import { createDemoSession } from "../src/domain/templates";
import { ExerciseEngine } from "../src/motion/exercise-engine";
import { createReplayFrames } from "../src/motion/replay-fixtures";
import {
  createRunResult,
  exportRunJson,
  summarizeExercise,
  summarizeRun,
} from "../src/session/results";
import type { ExerciseRunResult } from "../src/session/results";
import type { MotionEvent, RepRecord } from "../src/motion/types";

describe("session summaries", () => {
  it("summarizes exercise-specific estimates without video", () => {
    const session = createDemoSession();
    const exercise = session.exercises[0];
    const engine = new ExerciseEngine(exercise);
    let output;
    const events: MotionEvent[] = [];
    for (const frame of createReplayFrames({
      exerciseId: exercise.exerciseId,
      side: exercise.side,
      reps: 1,
      holdSeconds: exercise.dose.holdSeconds,
    })) {
      output = engine.process(frame);
      events.push(...output.events);
    }
    const result = createRunResult(
      session,
      new Date(0).toISOString(),
      "replay",
      [
        {
          exerciseId: exercise.id,
          kind: exercise.exerciseId,
          name: exercise.name,
          side: exercise.side,
          plannedSets: 1,
          sets: [
            {
              set: 1,
              plannedReps: 1,
              reps: output!.repRecords,
              events,
            },
          ],
        },
      ],
      false,
    );
    const summary = summarizeRun(result)[0];
    expect(summary.completedReps).toBe(1);
    expect(summary.medianPeak).toBeGreaterThan(60);
    expect(summary.constraints).toHaveLength(2);
    expect(summary.constraints[0].met).toBe(1);
    expect(summary.constraints[1].reached).toBe(0);
    expect(output!.repRecords[0].featurePeaks).not.toHaveLength(0);

    const exported = exportRunJson(result);
    expect(exported).toContain("Structured results only");
    expect(exported.toLowerCase()).not.toContain("video\":\"");
  });

  it("reports assessable target, minimum, and range misses honestly", () => {
    const exercise = createDemoSession().exercises[0];
    exercise.rules = [
      {
        id: "target",
        scope: exercise.rules[0].scope,
        metric: "knee-flexion",
        kind: "target",
        unit: "degrees",
        value: 80,
        hard: false,
        label: "Aim for 80°",
      },
      {
        id: "minimum",
        scope: exercise.rules[0].scope,
        metric: "knee-flexion",
        kind: "minimum",
        unit: "degrees",
        value: 70,
        hard: false,
        label: "Reach at least 70°",
      },
      {
        id: "range",
        scope: exercise.rules[0].scope,
        metric: "knee-flexion",
        kind: "range",
        unit: "degrees",
        min: 60,
        max: 90,
        hard: false,
        label: "Stay between 60° and 90°",
      },
    ];
    const rep = (repNumber: number, peak: number): RepRecord => ({
      rep: repNumber,
      startedAtMs: repNumber * 1000,
      completedAtMs: repNumber * 1000 + 500,
      durationMs: 500,
      holdMs: 0,
      maxKneeFlexion: peak,
      trackingGoodRatio: 1,
      targetReached: false,
      restrictionApproached: false,
      restrictionReached: false,
      observations: [],
      featurePeaks: [
        {
          metric: "knee-flexion",
          joints: ["right-hip", "right-knee", "right-ankle"],
          value: peak,
          unit: "degrees",
          quality: "good",
          assessableCoverage: 1,
          description: "Selected knee flexion",
        },
      ],
    });
    const result: ExerciseRunResult = {
      exerciseId: exercise.id,
      kind: exercise.exerciseId,
      name: exercise.name,
      side: exercise.side,
      plannedSets: 1,
      sets: [
        {
          set: 1,
          plannedReps: 2,
          reps: [rep(1, 80), rep(2, 50)],
          events: [],
        },
      ],
    };

    const constraints = summarizeExercise(exercise, result).constraints;
    expect(constraints.map(({ met, missed }) => ({ met, missed }))).toEqual([
      { met: 1, missed: 1 },
      { met: 1, missed: 1 },
      { met: 1, missed: 1 },
    ]);
  });
});
