import {
  BodySide,
  Clarification,
  ClarificationOption,
  ConfirmedSession,
  EXERCISE_LABELS,
  ExerciseId,
  ExercisePrescription,
  InterpretationField,
  InterpretationResult,
  MovementRule,
  SessionSource,
  constraintScopeForMetric,
  createId,
} from "../domain/types";
import { setupNotes } from "../domain/templates";

const NUMBER_WORDS: Record<string, number> = {
  a: 1,
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  eleven: 11,
  twelve: 12,
  fifteen: 15,
  twenty: 20,
  thirty: 30,
  forty: 40,
  fortyfive: 45,
  sixty: 60,
  ninety: 90,
};

const EXERCISE_PATTERNS: Array<{
  id: ExerciseId;
  pattern: RegExp;
}> = [
  {
    id: "heel-slide",
    pattern: /\bheel\s*slides?\b/gi,
  },
  {
    id: "straight-leg-raise",
    pattern: /\b(?:straight[\s-]*leg\s*raises?|slrs?)\b/gi,
  },
  {
    id: "seated-knee-extension",
    pattern:
      /\b(?:(?:seated\s+)?(?:knee|leg)\s*extensions?|long[\s-]*arc\s+quads?|laq)\b/gi,
  },
];

function parseNumber(value?: string): number | undefined {
  if (!value) return undefined;
  const normalized = value.toLowerCase().replace(/[\s-]/g, "");
  const numeric = Number(normalized);
  if (Number.isFinite(numeric)) return numeric;
  return NUMBER_WORDS[normalized];
}

function makeRule(
  metric: MovementRule["metric"],
  kind: MovementRule["kind"],
  value: number,
  label: string,
  sourceText: string,
  hard = false,
  unit: MovementRule["unit"] = "degrees",
): MovementRule {
  return {
    id: createId("rule"),
    scope: constraintScopeForMetric(metric),
    metric,
    kind,
    unit,
    value,
    hard,
    label,
    sourceText,
  };
}

interface Segment {
  exerciseId: ExerciseId;
  text: string;
}

function splitExerciseSegments(text: string): Segment[] {
  const matches: Array<{ index: number; end: number; exerciseId: ExerciseId }> =
    [];

  EXERCISE_PATTERNS.forEach(({ id, pattern }) => {
    pattern.lastIndex = 0;
    for (const match of text.matchAll(pattern)) {
      if (typeof match.index === "number") {
        matches.push({
          index: match.index,
          end: match.index + match[0].length,
          exerciseId: id,
        });
      }
    }
  });

  matches.sort((a, b) => a.index - b.index);
  if (matches.length === 0) return [];

  return matches.map((match, index) => {
    const next = matches[index + 1];
    const prefix =
      index === 0 && match.index > 0 ? text.slice(0, match.index) : "";
    return {
      exerciseId: match.exerciseId,
      text: `${prefix}${text.slice(match.index, next?.index ?? text.length)}`,
    };
  });
}

function parseSide(text: string): BodySide | undefined {
  const right = /\bright\b/i.test(text);
  const left = /\bleft\b/i.test(text);
  if (right === left) return undefined;
  return right ? "right" : "left";
}

function parseDose(text: string): {
  sets?: number;
  reps?: number;
  holdSeconds?: number;
  minRepDurationSeconds?: number;
  pace: "comfortable" | "slow" | "controlled";
} {
  const setsOfReps = text.match(
    /\b([\w-]+)\s+sets?\s+(?:of\s+)?([\w-]+)(?:\s+(?:repetitions?|reps?))?/i,
  );
  const compactDose = text.match(
    /\b([\w-]+)\s*[x×]\s*([\w-]+)\b/i,
  );
  const setsOnly = text.match(/\b([\w-]+)\s+sets?\b/i);
  const repsOnly = text.match(/\b([\w-]+)\s+(?:repetitions?|reps?)\b/i);
  const hold = text.match(
    /\bhold(?:ing)?(?:\s+(?:it|each|for))?\s+(?:for\s+)?([\w-]+)\s*(?:seconds?|secs?|s)\b/i,
  );
  const repDuration = text.match(
    /\b(?:at\s+least|minimum(?:\s+of)?)\s+([\w-]+)\s*(?:seconds?|secs?|s)\s+(?:for|per)\s+(?:each\s+)?(?:repetition|rep)\b/i,
  );

  return {
    sets: parseNumber(setsOfReps?.[1] ?? compactDose?.[1] ?? setsOnly?.[1]),
    reps: parseNumber(setsOfReps?.[2] ?? compactDose?.[2] ?? repsOnly?.[1]),
    holdSeconds: parseNumber(hold?.[1]),
    minRepDurationSeconds: parseNumber(repDuration?.[1]),
    pace: /\bslow(?:ly)?\b/i.test(text)
      ? "slow"
      : /\bcontrolled?\b/i.test(text)
        ? "controlled"
        : "comfortable",
  };
}

function angleMetricFor(exerciseId: ExerciseId): MovementRule["metric"] {
  return exerciseId === "straight-leg-raise"
    ? "leg-lift"
    : "knee-flexion";
}

function instructionClauses(text: string): string[] {
  return text
    .split(
      /(?:[.;!?\n]+|\s+\bbut\b\s+|,\s*(?:and\s+)?(?=(?:do|don['’]?t|never|must|avoid|try|aim|target|goal|reach|at\s+least|minimum|min|no\s+more|maximum|max|limit)\b)|\s+\band\b\s+(?=(?:do|don['’]?t|never|must|try|aim|target|goal|at\s+least|minimum|min|no\s+more|maximum|max|limit)\b))/gi,
    )
    .map((clause) => clause.trim())
    .filter(Boolean);
}

function explicitAngleIntent(
  clause: string,
  value: number,
): { maximum: boolean; minimum: boolean; target: boolean } {
  const angle = String(value);
  const degree = `${angle}\\s*(?:°|degrees?)`;
  const maximum =
    new RegExp(
      `\\b(?:do\\s+not|don['’]?t|never|must\\s+not|avoid)\\b[^.!?;\\n]{0,55}\\b(?:beyond|over|past|above|exceed|more\\s+than)\\s*${degree}`,
      "i",
    ).test(clause) ||
    new RegExp(
      `\\b(?:no\\s+more\\s+than|not\\s+(?:beyond|over|past|above)|maximum|max(?:imum)?(?:\\s+of)?|limit(?:ed)?(?:\\s+to|\\s+of)?)\\s*${degree}`,
      "i",
    ).test(clause) ||
    new RegExp(`${degree}\\s*(?:maximum|max|limit)\\b`, "i").test(clause);
  const minimum = new RegExp(
    `\\b(?:at\\s+least|minimum(?:\\s+of)?|min(?:imum)?(?:\\s+of)?|not\\s+less\\s+than)\\s*${degree}`,
    "i",
  ).test(clause);
  const target = new RegExp(
    `\\b(?:try\\s+(?:to\\s+)?(?:reach|achieve|get\\s+to)|aim(?:\\s+for)?|target(?:\\s+of)?|goal(?:\\s+of)?|reach|work\\s+toward|up\\s+to)\\s*(?:about|approximately|around)?\\s*${degree}`,
    "i",
  ).test(clause);
  return { maximum, minimum, target };
}

function parseRules(
  exerciseId: ExerciseId,
  text: string,
): {
  rules: MovementRule[];
  ambiguousAngle?: number;
} {
  const rules: MovementRule[] = [];
  const ambiguousAngles: number[] = [];
  const metric = angleMetricFor(exerciseId);

  for (const clause of instructionClauses(text)) {
    const range = clause.match(
      /\b(?:between|from)\s+(\d{1,3})\s*(?:°|degrees?)?\s+(?:and|to)\s+(\d{1,3})\s*(?:°|degrees?)\b/i,
    );
    if (range) {
      rules.push({
        id: createId("rule"),
        scope: constraintScopeForMetric(metric),
        metric,
        kind: "range",
        unit: "degrees",
        min: Number(range[1]),
        max: Number(range[2]),
        hard: false,
        label: `Stay between ${range[1]}° and ${range[2]}°`,
        sourceText: range[0],
      });
      continue;
    }

    for (const angle of clause.matchAll(
      /\b(\d{1,3})\s*(?:°|degrees?)\b/gi,
    )) {
      const value = Number(angle[1]);
      const intent = explicitAngleIntent(clause, value);
      if (intent.maximum) {
        rules.push(
          makeRule(
            metric,
            "maximum",
            value,
            `Do not go beyond ${value}°`,
            clause,
            true,
          ),
        );
      }
      if (intent.minimum) {
        rules.push(
          makeRule(
            metric,
            "minimum",
            value,
            `Reach at least ${value}°`,
            clause,
          ),
        );
      }
      if (intent.target) {
        rules.push(
          makeRule(
            metric,
            "target",
            value,
            `Aim for about ${value}°`,
            clause,
          ),
        );
      }
      if (!intent.maximum && !intent.minimum && !intent.target) {
        ambiguousAngles.push(value);
      }
    }
  }

  if (
    exerciseId === "straight-leg-raise" &&
    /\bkeep\b[\s\S]{0,16}\bknee\b[\s\S]{0,16}\bstraight\b/i.test(text)
  ) {
    rules.push(
      makeRule(
        "knee-bend",
        "maximum",
        12,
        "Keep the knee visually straight",
        "keep the knee straight",
        false,
      ),
    );
  }

  const repDuration = parseDose(text).minRepDurationSeconds;
  if (repDuration) {
    rules.push(
      makeRule(
        "rep-duration",
        "minimum",
        repDuration,
        `Take at least ${repDuration} seconds per rep`,
        `${repDuration} seconds per repetition`,
        false,
        "seconds",
      ),
    );
  }

  return {
    rules,
    ambiguousAngle: ambiguousAngles[0],
  };
}

function sideClarification(exerciseId: string): Clarification {
  return {
    id: createId("clarify"),
    exerciseId,
    reason: "missing-side",
    question: "Which leg is this for?",
    detail:
      "Relay needs the working side to choose the right landmarks and camera guidance.",
    options: [
      {
        id: "right",
        label: "Right leg",
        description: "Track the right hip, knee, ankle, and heel.",
        patch: { side: "right" },
      },
      {
        id: "left",
        label: "Left leg",
        description: "Track the left hip, knee, ankle, and heel.",
        patch: { side: "left" },
      },
    ],
  };
}

function angleClarification(
  exerciseRecordId: string,
  exerciseId: ExerciseId,
  angle: number,
): Clarification {
  const metric = angleMetricFor(exerciseId);
  const bodyPhrase =
    exerciseId === "straight-leg-raise"
      ? "visible leg lift"
      : "estimated knee flexion";
  const targetRule = makeRule(
    metric,
    "target",
    angle,
    `Aim for about ${angle}° of ${bodyPhrase}`,
    `${angle}°`,
  );
  const maximumRule = makeRule(
    metric,
    "maximum",
    angle,
    `Do not go beyond ${angle}° of ${bodyPhrase}`,
    `${angle}°`,
    true,
  );
  return {
    id: createId("clarify"),
    exerciseId: exerciseRecordId,
    reason: "angle-meaning",
    question: `What does ${angle}° mean in these instructions?`,
    detail:
      "A goal and a hard limit produce different coaching. Relay will not guess.",
    options: [
      {
        id: "target",
        label: "A target",
        description: "Encourage movement toward this estimate.",
        patch: { rule: targetRule },
      },
      {
        id: "maximum",
        label: "A hard maximum",
        description: "Warn before it and tell me to stop at the limit.",
        patch: { rule: maximumRule },
      },
      {
        id: "both",
        label: "A target and maximum",
        description: "Encourage me toward it, but never beyond it.",
        patch: { rules: [targetRule, maximumRule] },
      },
    ],
  };
}

function doseClarification(exerciseId: string): Clarification {
  const options: ClarificationOption[] = [
    {
      id: "one-ten",
      label: "1 set of 10",
      description: "A short single set.",
      patch: { sets: 1, reps: 10 },
    },
    {
      id: "two-ten",
      label: "2 sets of 10",
      description: "Twenty total repetitions.",
      patch: { sets: 2, reps: 10 },
    },
    {
      id: "three-ten",
      label: "3 sets of 10",
      description: "Thirty total repetitions.",
      patch: { sets: 3, reps: 10 },
    },
  ];
  return {
    id: createId("clarify"),
    exerciseId,
    reason: "missing-dose",
    question: "How many sets and repetitions?",
    detail: "Relay found the exercise but not a complete dose.",
    options,
  };
}

export function interpretInstructions(
  input: string,
  source: SessionSource = "typed",
): InterpretationResult {
  const text = input.trim();
  const segments = splitExerciseSegments(text);
  const fields: InterpretationField[] = [];
  const clarifications: Clarification[] = [];
  const exercises: ExercisePrescription[] = [];
  const notices: string[] = [];

  for (const segment of segments) {
    const side = parseSide(segment.text) ?? parseSide(text);
    const dose = parseDose(segment.text);
    const { rules, ambiguousAngle } = parseRules(
      segment.exerciseId,
      segment.text,
    );
    const exerciseId = createId("exercise");
    const prescription: ExercisePrescription = {
      id: exerciseId,
      exerciseId: segment.exerciseId,
      name: EXERCISE_LABELS[segment.exerciseId],
      side: side ?? "right",
      dose: {
        sets: dose.sets ?? 1,
        reps: dose.reps ?? 10,
        holdSeconds: dose.holdSeconds ?? 0,
        tempo: {
          cue: dose.pace,
          minRepDurationSeconds: dose.minRepDurationSeconds,
        },
      },
      rules,
      notes: setupNotes[segment.exerciseId],
    };
    exercises.push(prescription);

    fields.push({
      field: "exercise",
      value: prescription.name,
      confidence: 0.99,
      evidence: segment.text.trim(),
    });
    if (side) {
      fields.push({
        field: "side",
        value: side,
        confidence: 0.98,
        evidence: side,
      });
    } else {
      clarifications.push(sideClarification(exerciseId));
    }
    if (dose.sets && dose.reps) {
      fields.push({
        field: "dose",
        value: `${dose.sets} × ${dose.reps}`,
        confidence: 0.97,
        evidence: segment.text.trim(),
      });
    } else {
      clarifications.push(doseClarification(exerciseId));
    }
    if (dose.holdSeconds !== undefined) {
      fields.push({
        field: "hold",
        value: `${dose.holdSeconds} seconds`,
        confidence: 0.96,
        evidence: segment.text.trim(),
      });
    }
    if (ambiguousAngle !== undefined) {
      clarifications.push(
        angleClarification(exerciseId, segment.exerciseId, ambiguousAngle),
      );
    }
  }

  if (exercises.length === 0) {
    notices.push(
      "Relay currently recognizes heel slides, straight-leg raises, and seated knee extensions.",
    );
  }

  const now = new Date().toISOString();
  const session: ConfirmedSession = {
    schemaVersion: 1,
    id: createId("session"),
    title:
      exercises.length === 1
        ? `${exercises[0].side === "right" ? "Right" : "Left"} ${exercises[0].name.toLowerCase()} session`
        : "Knee rehabilitation session",
    createdAt: now,
    updatedAt: now,
    source,
    sourceText: text,
    exercises,
    clarifications,
    confirmed: false,
    estimatesAcknowledged: false,
  };

  return { session, fields, notices };
}

export function applyClarification(
  session: ConfirmedSession,
  clarificationId: string,
  optionId: string,
): ConfirmedSession {
  const clarification = session.clarifications.find(
    (item) => item.id === clarificationId,
  );
  const option = clarification?.options.find((item) => item.id === optionId);
  if (!clarification || !option) return session;

  const exercises = session.exercises.map((exercise) => {
    if (exercise.id !== clarification.exerciseId) return exercise;
    return {
      ...exercise,
      side: option.patch.side ?? exercise.side,
      dose: {
        ...exercise.dose,
        sets: option.patch.sets ?? exercise.dose.sets,
        reps: option.patch.reps ?? exercise.dose.reps,
      },
      rules: option.patch.rules
        ? [...exercise.rules, ...option.patch.rules]
        : option.patch.rule
          ? [...exercise.rules, option.patch.rule]
          : exercise.rules,
    };
  });

  return {
    ...session,
    exercises,
    clarifications: session.clarifications.filter(
      (item) => item.id !== clarificationId,
    ),
    updatedAt: new Date().toISOString(),
  };
}
