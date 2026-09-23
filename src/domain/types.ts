export type ExerciseId =
  | "heel-slide"
  | "straight-leg-raise"
  | "seated-knee-extension";

export type BodySide = "left" | "right";

export type BodyJoint =
  | "nose"
  | "shoulder"
  | "elbow"
  | "wrist"
  | "hip"
  | "knee"
  | "ankle"
  | "heel"
  | "foot-index";

export type ConstraintScope =
  | {
      kind: "joint";
      joints: BodyJoint[];
      relationship: "angle" | "alignment" | "range";
      description: string;
    }
  | {
      kind: "relationship";
      joints: BodyJoint[];
      relationship:
        | "relative-angle"
        | "relative-position"
        | "segment-stability"
        | "whole-chain-motion";
      description: string;
    }
  | {
      kind: "whole-body";
      joints: BodyJoint[];
      relationship: "position-stability" | "movement-pattern";
      description: string;
    }
  | {
      kind: "temporal";
      joints: BodyJoint[];
      relationship: "duration" | "tempo" | "hold-stability";
      description: string;
    };

export type SessionSource =
  | "typed"
  | "spoken"
  | "upload"
  | "manual"
  | "template";

export type KnownMetricId =
  | "knee-flexion"
  | "leg-lift"
  | "extension-progress"
  | "rep-duration"
  | "hold-duration"
  | "knee-bend"
  | "trunk-compensation";

/**
 * Feature providers may register additional metric keys without changing the
 * session or constraint contracts.
 */
export type MetricId = KnownMetricId | (string & {});

export type RuleKind =
  | "target"
  | "minimum"
  | "maximum"
  | "range"
  | "visible-condition";

export type RuleUnit = "degrees" | "seconds" | "percent" | "observation";

export interface MovementRule {
  id: string;
  scope: ConstraintScope;
  metric: MetricId;
  kind: RuleKind;
  unit: RuleUnit;
  value?: number;
  min?: number;
  max?: number;
  hard: boolean;
  label: string;
  sourceText?: string;
}

export interface TempoPrescription {
  cue: "comfortable" | "slow" | "controlled";
  minRepDurationSeconds?: number;
  liftSeconds?: number;
  lowerSeconds?: number;
}

export interface ExerciseDose {
  sets: number;
  reps: number;
  holdSeconds: number;
  restSeconds?: number;
  tempo: TempoPrescription;
}

export interface ExercisePrescription {
  id: string;
  exerciseId: ExerciseId;
  name: string;
  side: BodySide;
  dose: ExerciseDose;
  rules: MovementRule[];
  notes: string[];
}

export type ClarificationReason =
  | "missing-side"
  | "angle-meaning"
  | "missing-dose";

export interface ClarificationOption {
  id: string;
  label: string;
  description: string;
  patch: {
    side?: BodySide;
    sets?: number;
    reps?: number;
    rule?: MovementRule;
    rules?: MovementRule[];
  };
}

export interface Clarification {
  id: string;
  exerciseId: string;
  reason: ClarificationReason;
  question: string;
  detail: string;
  options: ClarificationOption[];
}

export interface ConfirmedSession {
  schemaVersion: 1;
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  source: SessionSource;
  sourceText: string;
  exercises: ExercisePrescription[];
  clarifications: Clarification[];
  confirmed: boolean;
  estimatesAcknowledged: boolean;
}

export interface InterpretationField {
  field: string;
  value: string;
  confidence: number;
  evidence: string;
}

export interface InterpretationResult {
  session: ConfirmedSession;
  fields: InterpretationField[];
  notices: string[];
}

export const EXERCISE_LABELS: Record<ExerciseId, string> = {
  "heel-slide": "Heel slide",
  "straight-leg-raise": "Straight-leg raise",
  "seated-knee-extension": "Seated knee extension",
};

export function constraintScopeForMetric(metric: MetricId): ConstraintScope {
  switch (metric) {
    case "knee-flexion":
      return {
        kind: "joint",
        joints: ["hip", "knee", "ankle"],
        relationship: "angle",
        description: "Hip–knee–ankle angle at the selected knee",
      };
    case "knee-bend":
      return {
        kind: "relationship",
        joints: ["hip", "knee", "ankle"],
        relationship: "relative-angle",
        description: "Visible straightness of the selected leg",
      };
    case "leg-lift":
      return {
        kind: "relationship",
        joints: ["hip", "knee", "ankle"],
        relationship: "whole-chain-motion",
        description: "Lift of the hip–knee–ankle chain from its start position",
      };
    case "extension-progress":
      return {
        kind: "relationship",
        joints: ["hip", "knee", "ankle"],
        relationship: "relative-angle",
        description: "Knee extension relative to the seated start position",
      };
    case "trunk-compensation":
      return {
        kind: "whole-body",
        joints: ["shoulder", "hip"],
        relationship: "position-stability",
        description: "Broad visible shoulder–hip relationship stability",
      };
    case "rep-duration":
      return {
        kind: "temporal",
        joints: ["shoulder", "hip", "knee", "ankle"],
        relationship: "tempo",
        description: "Duration of the observed whole movement cycle",
      };
    case "hold-duration":
      return {
        kind: "temporal",
        joints: ["hip", "knee", "ankle"],
        relationship: "hold-stability",
        description: "Time that the endpoint body relationship remains stable",
      };
    default:
      return {
        kind: "relationship",
        joints: [],
        relationship: "relative-position",
        description: `Provider-defined body feature: ${metric}`,
      };
  }
}

export function createId(prefix: string): string {
  const random =
    typeof globalThis.crypto !== "undefined" &&
    typeof globalThis.crypto.randomUUID === "function"
      ? globalThis.crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
  return `${prefix}-${random}`;
}
