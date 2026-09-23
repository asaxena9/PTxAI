import {
  ConfirmedSession,
  ExerciseId,
  ExercisePrescription,
  MovementRule,
  SessionSource,
  constraintScopeForMetric,
  createId,
  EXERCISE_LABELS,
} from "./types";

export const REQUIRED_SCENARIO_TEXT =
  "Right heel slides, three sets of ten, hold for two seconds, and do not bend my knee beyond 90 degrees.";

const setupNotes: Record<ExerciseId, string[]> = {
  "heel-slide": [
    "Lie on your back with the working leg visible from hip to foot.",
    "Slide your heel toward you, then return until the leg is long.",
  ],
  "straight-leg-raise": [
    "Lie on your back with the working leg fully visible from hip to ankle.",
    "Keep the knee visually straight as the leg lifts and lowers.",
  ],
  "seated-knee-extension": [
    "Sit side-on in a stable chair with hip, knee, and ankle visible.",
    "Straighten the knee, pause, and return with control.",
  ],
};

function rule(
  metric: MovementRule["metric"],
  kind: MovementRule["kind"],
  value: number,
  label: string,
  hard = false,
  unit: MovementRule["unit"] = "degrees",
): MovementRule {
  return {
    id: createId("rule"),
    scope: constraintScopeForMetric(metric),
    metric,
    kind,
    value,
    unit,
    label,
    hard,
  };
}

function exercise(
  exerciseId: ExerciseId,
  side: "left" | "right",
  reps: number,
  holdSeconds: number,
  rules: MovementRule[],
): ExercisePrescription {
  return {
    id: createId("exercise"),
    exerciseId,
    name: EXERCISE_LABELS[exerciseId],
    side,
    dose: {
      sets: 1,
      reps,
      holdSeconds,
      tempo: { cue: "controlled", minRepDurationSeconds: 3 },
    },
    rules,
    notes: setupNotes[exerciseId],
  };
}

export function createDemoSession(
  source: SessionSource = "template",
): ConfirmedSession {
  const now = new Date().toISOString();
  return {
    schemaVersion: 1,
    id: createId("session"),
    title: "Movement recovery circuit",
    createdAt: now,
    updatedAt: now,
    source,
    sourceText:
      "A three-exercise right-side routine with controlled reps, short holds, whole-chain posture checks, and a 90° heel-slide limit.",
    confirmed: false,
    estimatesAcknowledged: false,
    clarifications: [],
    exercises: [
      exercise("heel-slide", "right", 3, 2, [
        rule("knee-flexion", "target", 80, "Aim for about 80°"),
        rule(
          "knee-flexion",
          "maximum",
          90,
          "Do not bend beyond 90°",
          true,
        ),
      ]),
      exercise("straight-leg-raise", "right", 3, 2, [
        rule(
          "knee-bend",
          "maximum",
          12,
          "Keep the knee visually straight",
          false,
        ),
        rule(
          "trunk-compensation",
          "visible-condition",
          12,
          "Keep the shoulder–hip relationship steady",
          false,
          "degrees",
        ),
      ]),
      exercise("seated-knee-extension", "right", 3, 2, [
        rule(
          "extension-progress",
          "minimum",
          85,
          "Aim for at least 85% extension",
          false,
          "percent",
        ),
      ]),
    ],
  };
}

export function createManualSession(): ConfirmedSession {
  const session = createDemoSession("manual");
  return {
    ...session,
    title: "My movement session",
    sourceText: "Configured manually",
    exercises: [
      exercise("heel-slide", "right", 10, 0, []),
      exercise("straight-leg-raise", "right", 10, 0, []),
      exercise("seated-knee-extension", "right", 10, 0, []),
    ],
  };
}

export { setupNotes };
