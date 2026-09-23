import { describe, expect, it } from "vitest";
import { createDemoSession } from "../src/domain/templates";
import { ConstraintEvaluator } from "../src/motion/constraint-evaluator";
import type { MotionMeasurements } from "../src/motion/types";

function measurements(overrides: Partial<MotionMeasurements>): MotionMeasurements {
  const base: MotionMeasurements = {
    kneeFlexion: 4,
    kneeFlexionImage: 4,
    kneeFlexionWorld: 4,
    angleAgreementDelta: 0,
    legLift: 35,
    extensionProgress: 90,
    kneeBend: 4,
    trunkCompensation: 2,
    values: {
      "knee-flexion": 4,
      "leg-lift": 35,
      "extension-progress": 90,
      "knee-bend": 4,
      "trunk-compensation": 2,
    },
    features: [],
    confidence: {
      score: 0.95,
      status: "good",
      canAssessBodyFeatures: true,
      canAssessAngle: true,
      reasons: ["good"],
    },
  };
  const merged = {
    ...base,
    ...overrides,
    values: { ...base.values, ...overrides.values },
  };
  if (!overrides.features) {
    merged.features = Object.entries(merged.values).map(([metric, value]) => ({
      id: `test-${metric}`,
      metric,
      joints: [],
      value,
      unit: metric.includes("progress")
        ? ("percent" as const)
        : ("degrees" as const),
      quality: merged.confidence.status,
      description: `Test feature ${metric}`,
    }));
  }
  return merged;
}

describe("body constraint evaluator", () => {
  it("applies a hard maximum to a joint relationship", () => {
    const exercise = createDemoSession().exercises[0];
    const evaluator = new ConstraintEvaluator(exercise);
    const approaching = evaluator.evaluateFrame(
      measurements({
        kneeFlexion: 82,
        values: { "knee-flexion": 82 },
      }),
      1000,
      { repElapsedMs: 500, holdElapsedMs: 0 },
    );
    const reached = evaluator.evaluateFrame(
      measurements({
        kneeFlexion: 91,
        values: { "knee-flexion": 91 },
      }),
      1200,
      { repElapsedMs: 700, holdElapsedMs: 0 },
    );
    expect(
      approaching.some((item) => item.type === "restriction-approaching"),
    ).toBe(true);
    expect(reached.some((item) => item.type === "restriction-reached")).toBe(
      true,
    );
  });

  it("requires a sustained broad posture observation", () => {
    const exercise = createDemoSession().exercises[1];
    const evaluator = new ConstraintEvaluator(exercise);
    const first = evaluator.evaluateFrame(
      measurements({
        trunkCompensation: 18,
        values: { "trunk-compensation": 18 },
      }),
      1000,
      { repElapsedMs: 500, holdElapsedMs: 0 },
    );
    const sustained = evaluator.evaluateFrame(
      measurements({
        trunkCompensation: 18,
        values: { "trunk-compensation": 18 },
      }),
      1400,
      { repElapsedMs: 900, holdElapsedMs: 0 },
    );
    expect(first).toHaveLength(0);
    expect(
      sustained.some((item) => item.type === "posture-observation"),
    ).toBe(true);
  });

  it("never fires a pose constraint from unavailable observations", () => {
    const exercise = createDemoSession().exercises[0];
    const evaluator = new ConstraintEvaluator(exercise);
    const events = evaluator.evaluateFrame(
      measurements({
        kneeFlexion: 95,
        values: { "knee-flexion": 95 },
        confidence: {
          score: 0.2,
          status: "lost",
          canAssessBodyFeatures: false,
          canAssessAngle: false,
          reasons: ["occluded"],
        },
      }),
      1000,
      { repElapsedMs: 500, holdElapsedMs: 0 },
    );
    expect(events).toHaveLength(0);
  });

  it("supports a minimum body-feature target", () => {
    const exercise = createDemoSession().exercises[2];
    const evaluator = new ConstraintEvaluator(exercise);
    const events = evaluator.evaluateFrame(
      measurements({
        extensionProgress: 88,
        values: { "extension-progress": 88 },
      }),
      1000,
      { repElapsedMs: 500, holdElapsedMs: 0 },
    );
    expect(events.some((event) => event.type === "target-reached")).toBe(true);
  });

  it("evaluates an absolute target near its confirmed value", () => {
    const exercise = createDemoSession().exercises[0];
    const evaluator = new ConstraintEvaluator(exercise);
    const events = evaluator.evaluateFrame(
      measurements({
        kneeFlexion: 76,
        values: { "knee-flexion": 76 },
      }),
      1000,
      { repElapsedMs: 500, holdElapsedMs: 0 },
    );
    expect(events.some((event) => event.type === "target-reached")).toBe(true);
  });

  it("supports maximum temporal constraints at completion", () => {
    const exercise = createDemoSession().exercises[0];
    exercise.rules.push({
      id: "tempo-max",
      scope: {
        kind: "temporal",
        joints: ["hip", "knee", "ankle"],
        relationship: "tempo",
        description: "Maximum movement duration",
      },
      metric: "rep-duration",
      kind: "maximum",
      unit: "seconds",
      value: 4,
      hard: false,
      label: "Finish within four seconds",
    });
    const evaluator = new ConstraintEvaluator(exercise);
    expect(
      evaluator
        .evaluateCompletion(5000, 2000, 5000)
        .some((event) => event.type === "tempo-missed"),
    ).toBe(true);
  });

  it("treats a spatial range as an endpoint band without flagging the return", () => {
    const exercise = createDemoSession().exercises[0];
    exercise.rules = [
      {
        id: "flexion-range",
        scope: {
          kind: "joint",
          joints: ["hip", "knee", "ankle"],
          relationship: "range",
          description: "Selected knee flexion range",
        },
        metric: "knee-flexion",
        kind: "range",
        unit: "degrees",
        min: 60,
        max: 90,
        hard: false,
        label: "Stay between 60° and 90°",
      },
    ];
    const evaluator = new ConstraintEvaluator(exercise);
    const entered = evaluator.evaluateFrame(
      measurements({
        kneeFlexion: 70,
        values: { "knee-flexion": 70 },
      }),
      1000,
      { repElapsedMs: 500, holdElapsedMs: 0 },
    );
    const returned = evaluator.evaluateFrame(
      measurements({
        kneeFlexion: 20,
        values: { "knee-flexion": 20 },
      }),
      1400,
      { repElapsedMs: 900, holdElapsedMs: 0 },
    );
    const overshot = evaluator.evaluateFrame(
      measurements({
        kneeFlexion: 96,
        values: { "knee-flexion": 96 },
      }),
      1600,
      { repElapsedMs: 1100, holdElapsedMs: 0 },
    );

    expect(entered.some((event) => event.type === "target-reached")).toBe(true);
    expect(returned).toHaveLength(0);
    expect(
      overshot.some((event) => event.type === "posture-observation"),
    ).toBe(true);
  });
});
