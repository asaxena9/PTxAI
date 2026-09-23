import { describe, expect, it } from "vitest";
import { REQUIRED_SCENARIO_TEXT } from "../src/domain/templates";
import { confirmedSessionSchema } from "../src/domain/schema";
import {
  applyClarification,
  interpretInstructions,
} from "../src/intelligence/interpreter";

describe("session interpretation", () => {
  it("parses the required right heel-slide maximum without clarification", () => {
    const result = interpretInstructions(REQUIRED_SCENARIO_TEXT);
    const exercise = result.session.exercises[0];
    const maximum = exercise.rules.find((rule) => rule.kind === "maximum");

    expect(result.session.clarifications).toHaveLength(0);
    expect(exercise.exerciseId).toBe("heel-slide");
    expect(exercise.side).toBe("right");
    expect(exercise.dose.sets).toBe(3);
    expect(exercise.dose.reps).toBe(10);
    expect(exercise.dose.holdSeconds).toBe(2);
    expect(maximum).toMatchObject({
      metric: "knee-flexion",
      kind: "maximum",
      value: 90,
      hard: true,
      scope: {
        kind: "joint",
        joints: ["hip", "knee", "ankle"],
      },
    });
    expect(confirmedSessionSchema.safeParse(result.session).success).toBe(true);
  });

  it("does not guess whether an unqualified angle is a goal or limit", () => {
    const result = interpretInstructions(
      "Right heel slides, 3 x 10, hold 2 seconds, to 90 degrees.",
    );
    const issue = result.session.clarifications.find(
      (item) => item.reason === "angle-meaning",
    );
    expect(issue).toBeDefined();
    expect(issue?.options.map((option) => option.id)).toEqual([
      "target",
      "maximum",
      "both",
    ]);

    const resolved = applyClarification(
      result.session,
      issue!.id,
      "maximum",
    );
    expect(
      resolved.exercises[0].rules.find((rule) => rule.kind === "maximum"),
    ).toMatchObject({ value: 90, hard: true });
  });

  it("maps multiple exercises and body-relationship constraints", () => {
    const result = interpretInstructions(
      "Right straight-leg raises, two sets of eight, hold for two seconds, keep the knee straight. Then right LAQ, 2 x 10.",
    );
    expect(result.session.exercises.map((item) => item.exerciseId)).toEqual([
      "straight-leg-raise",
      "seated-knee-extension",
    ]);
    expect(
      result.session.exercises[0].rules.find(
        (rule) => rule.metric === "knee-bend",
      )?.scope,
    ).toMatchObject({
      kind: "relationship",
      joints: ["hip", "knee", "ankle"],
    });
  });

  it("keeps qualitative slow separate from a measurable tempo minimum", () => {
    const qualitative = interpretInstructions(
      "Right heel slides, one set of ten, move slowly.",
    ).session.exercises[0];
    expect(qualitative.dose.tempo.cue).toBe("slow");
    expect(
      qualitative.rules.find((rule) => rule.metric === "rep-duration"),
    ).toBeUndefined();

    const measurable = interpretInstructions(
      "Right heel slides, one set of ten, take at least three seconds per repetition.",
    ).session.exercises[0];
    expect(measurable.dose.tempo.minRepDurationSeconds).toBe(3);
    expect(
      measurable.rules.find((rule) => rule.metric === "rep-duration"),
    ).toMatchObject({
      kind: "minimum",
      value: 3,
      scope: { kind: "temporal" },
    });
  });

  it("binds angle meaning to its local instruction clause", () => {
    const bounce = interpretInstructions(
      "Right heel slides, 3 sets of 10. Do not bounce. Aim for 90 degrees.",
    ).session.exercises[0];
    const pain = interpretInstructions(
      "Right heel slides, 3 sets of 10. Avoid pain and try to reach 90 degrees.",
    ).session.exercises[0];

    for (const exercise of [bounce, pain]) {
      expect(
        exercise.rules.find((rule) => rule.kind === "maximum"),
      ).toBeUndefined();
      expect(
        exercise.rules.find((rule) => rule.kind === "target"),
      ).toMatchObject({ value: 90, hard: false });
    }
  });

  it("preserves distinct target, minimum, and maximum angle clauses", () => {
    const targetAndMaximum = interpretInstructions(
      "Right heel slides, 3 sets of 10. Aim for 80 degrees but do not go beyond 90 degrees.",
    ).session.exercises[0];
    expect(
      targetAndMaximum.rules.find((rule) => rule.kind === "target"),
    ).toMatchObject({ value: 80, hard: false });
    expect(
      targetAndMaximum.rules.find((rule) => rule.kind === "maximum"),
    ).toMatchObject({ value: 90, hard: true });

    const rangeEdges = interpretInstructions(
      "Right heel slides, 3 sets of 10. At least 60 degrees but no more than 90 degrees.",
    ).session.exercises[0];
    expect(
      rangeEdges.rules.find((rule) => rule.kind === "minimum"),
    ).toMatchObject({ value: 60, hard: false });
    expect(
      rangeEdges.rules.find((rule) => rule.kind === "maximum"),
    ).toMatchObject({ value: 90, hard: true });
  });
});
