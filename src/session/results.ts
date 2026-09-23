import type {
  ConfirmedSession,
  ExerciseId,
  ExercisePrescription,
} from "../domain/types";
import { createId } from "../domain/types";
import { movementTargetSatisfied } from "../motion/constraint-evaluator";
import type { MotionEvent, RepRecord } from "../motion/types";

export interface ExerciseSetResult {
  set: number;
  plannedReps: number;
  reps: RepRecord[];
  events: MotionEvent[];
}

export interface ExerciseRunResult {
  exerciseId: string;
  kind: ExerciseId;
  name: string;
  side: "left" | "right";
  plannedSets: number;
  sets: ExerciseSetResult[];
}

export interface SessionRunResult {
  id: string;
  sessionId: string;
  title: string;
  startedAt: string;
  completedAt: string;
  completed: boolean;
  source: "live" | "replay";
  session: ConfirmedSession;
  exercises: ExerciseRunResult[];
}

export interface ExerciseSummary {
  exerciseId: string;
  name: string;
  side: "left" | "right";
  completedReps: number;
  plannedReps: number;
  completedSets: number;
  plannedSets: number;
  holdMet: number;
  holdAssessable: number;
  trackingCoverage: number;
  medianPeak?: number;
  peakRange?: [number, number];
  targetReached: number;
  restrictionApproached: number;
  restrictionReached: number;
  observations: string[];
  tempoMet: number;
  tempoMissed: number;
  limitedPrimaryPeaks: number;
  constraints: Array<{
    ruleId: string;
    label: string;
    metric: string;
    kind: ExercisePrescription["rules"][number]["kind"];
    unit: ExercisePrescription["rules"][number]["unit"];
    value?: number;
    min?: number;
    max?: number;
    assessableReps: number;
    met: number;
    missed: number;
    approached: number;
    reached: number;
    observed: number;
  }>;
}

export function createRunResult(
  session: ConfirmedSession,
  startedAt: string,
  source: "live" | "replay",
  exercises: ExerciseRunResult[],
  completed: boolean,
): SessionRunResult {
  return {
    id: createId("run"),
    sessionId: session.id,
    title: session.title,
    startedAt,
    completedAt: new Date().toISOString(),
    completed,
    source,
    session,
    exercises,
  };
}

function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function summarizeExercise(
  prescription: ExercisePrescription,
  result: ExerciseRunResult,
): ExerciseSummary {
  const reps = result.sets.flatMap((set) => set.reps);
  const events = result.sets.flatMap((set) => set.events);
  const primaryMetric =
    prescription.exerciseId === "straight-leg-raise"
      ? "leg-lift"
      : prescription.exerciseId === "seated-knee-extension"
        ? "extension-progress"
        : "knee-flexion";
  const primaryPeaks = reps.map((rep) =>
    rep.featurePeaks?.find((peak) => peak.metric === primaryMetric),
  );
  const peaks = reps
    .map((rep, index) => {
      if (primaryPeaks[index]) return primaryPeaks[index]?.value;
      if (prescription.exerciseId === "heel-slide") return rep.maxKneeFlexion;
      if (prescription.exerciseId === "straight-leg-raise")
        return rep.maxLegLift;
      return rep.maxExtensionProgress;
    })
    .filter((value): value is number => value !== undefined);
  const requiredHoldMs = prescription.dose.holdSeconds * 1000;
  const observations = Array.from(
    new Set(reps.flatMap((rep) => rep.observations)),
  );
  return {
    exerciseId: prescription.id,
    name: prescription.name,
    side: prescription.side,
    completedReps: reps.length,
    plannedReps: prescription.dose.sets * prescription.dose.reps,
    completedSets: result.sets.filter(
      (set) => set.reps.length >= set.plannedReps,
    ).length,
    plannedSets: prescription.dose.sets,
    holdMet:
      requiredHoldMs > 0
        ? reps.filter((rep) => rep.holdMs >= requiredHoldMs).length
        : 0,
    holdAssessable: requiredHoldMs > 0 ? reps.length : 0,
    trackingCoverage:
      reps.length > 0
        ? reps.reduce((sum, rep) => sum + rep.trackingGoodRatio, 0) /
          reps.length
        : 0,
    medianPeak: median(peaks),
    peakRange:
      peaks.length > 0 ? [Math.min(...peaks), Math.max(...peaks)] : undefined,
    targetReached: reps.filter((rep) => rep.targetReached).length,
    restrictionApproached: reps.filter(
      (rep) => rep.restrictionApproached,
    ).length,
    restrictionReached: reps.filter((rep) => rep.restrictionReached).length,
    observations,
    tempoMet: events.filter((event) => event.type === "tempo-met").length,
    tempoMissed: events.filter((event) => event.type === "tempo-missed").length,
    limitedPrimaryPeaks: primaryPeaks.filter(
      (peak) => peak?.quality === "limited" || peak?.quality === "lost",
    ).length,
    constraints: prescription.rules.map((rule) => {
      const ruleEvents = events.filter((event) => event.ruleId === rule.id);
      const assessableRepPeaks =
        rule.scope.kind === "temporal"
          ? []
          : reps
              .map((rep) =>
                rep.featurePeaks?.find(
                  (peak) =>
                    peak.metric === rule.metric && peak.quality !== "lost",
                ),
              )
              .filter((peak) => peak !== undefined);
      const assessableReps =
        rule.scope.kind === "temporal"
          ? reps.length
          : assessableRepPeaks.length;
      const spatialTarget =
        rule.scope.kind !== "temporal" &&
        (rule.kind === "target" ||
          rule.kind === "minimum" ||
          rule.kind === "range");
      const spatialMet = spatialTarget
        ? assessableRepPeaks.filter((peak) =>
            movementTargetSatisfied(rule, peak.value),
          ).length
        : 0;
      const eventMet = ruleEvents.filter(
        (event) =>
          event.type === "target-reached" ||
          event.type === "tempo-met" ||
          event.type === "hold-met",
      ).length;
      const met = spatialTarget ? spatialMet : eventMet;
      const eventMissed = ruleEvents.filter(
        (event) =>
          event.type === "tempo-missed" || event.type === "hold-short",
      ).length;
      return {
        ruleId: rule.id,
        label: rule.label,
        metric: rule.metric,
        kind: rule.kind,
        unit: rule.unit,
        value: rule.value,
        min: rule.min,
        max: rule.max,
        assessableReps,
        met,
        missed: spatialTarget ? assessableReps - met : eventMissed,
        approached: ruleEvents.filter(
          (event) => event.type === "restriction-approaching",
        ).length,
        reached: ruleEvents.filter(
          (event) => event.type === "restriction-reached",
        ).length,
        observed: ruleEvents.filter(
          (event) => event.type === "posture-observation",
        ).length,
      };
    }),
  };
}

export function summarizeRun(result: SessionRunResult): ExerciseSummary[] {
  return result.session.exercises.map((prescription) => {
    const run = result.exercises.find(
      (item) => item.exerciseId === prescription.id,
    ) ?? {
      exerciseId: prescription.id,
      kind: prescription.exerciseId,
      name: prescription.name,
      side: prescription.side,
      plannedSets: prescription.dose.sets,
      sets: [],
    };
    return summarizeExercise(prescription, run);
  });
}

export function exportRunJson(result: SessionRunResult): string {
  return JSON.stringify(
    {
      exportVersion: "relay.session-result/1",
      exportedAt: new Date().toISOString(),
      privacy: "Structured results only. No camera video is included.",
      result,
      summaries: summarizeRun(result),
    },
    null,
    2,
  );
}
