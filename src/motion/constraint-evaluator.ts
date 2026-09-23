import type {
  ExercisePrescription,
  MetricId,
  MovementRule,
} from "../domain/types";
import { createId } from "../domain/types";
import type { MotionEvent, MotionMeasurements } from "./types";

type RuleState = "clear" | "approaching" | "reached" | "met" | "observed";

function observedValue(
  metric: MetricId,
  measurements: MotionMeasurements,
  timing: { repElapsedMs: number; holdElapsedMs: number },
): number | undefined {
  if (metric === "rep-duration") return timing.repElapsedMs / 1000;
  if (metric === "hold-duration") return timing.holdElapsedMs / 1000;
  return measurements.values[metric];
}

function motionEvent(
  type: MotionEvent["type"],
  timestampMs: number,
  message: string,
  value?: number,
  confidence?: number,
  rule?: MovementRule,
  measurementQuality?: MotionEvent["measurementQuality"],
): MotionEvent {
  return {
    id: createId("event"),
    type,
    timestampMs,
    message,
    value,
    confidence,
    ruleId: rule?.id,
    metric: rule?.metric,
    measurementQuality,
  };
}

function bodyLabel(rule: MovementRule): string {
  if (rule.scope.kind === "temporal") return "movement timing";
  const joints = rule.scope.joints;
  if (joints.length === 1) return joints[0];
  return joints.join("–");
}

export function movementTargetSatisfied(
  rule: MovementRule,
  value: number,
): boolean {
  if (rule.kind === "target" && rule.value !== undefined) {
    const tolerance =
      rule.unit === "degrees" ? 5 : rule.unit === "percent" ? 3 : 0.25;
    return Math.abs(value - rule.value) <= tolerance;
  }
  if (rule.kind === "minimum" && rule.value !== undefined) {
    return value >= rule.value;
  }
  if (
    rule.kind === "range" &&
    rule.min !== undefined &&
    rule.max !== undefined
  ) {
    return value >= rule.min && value <= rule.max;
  }
  return false;
}

/**
 * Evaluates confirmed constraints against observable body-pose features.
 *
 * The evaluator is deliberately exercise-agnostic. Exercise extractors decide
 * which body relationships are observable; this layer applies target,
 * restriction, posture, range, tempo, and hold semantics to those observations.
 */
export class ConstraintEvaluator {
  private states = new Map<string, RuleState>();
  private exceededSince = new Map<string, number>();

  constructor(private readonly prescription: ExercisePrescription) {}

  evaluateFrame(
    measurements: MotionMeasurements,
    timestampMs: number,
    timing: { repElapsedMs: number; holdElapsedMs: number },
  ): MotionEvent[] {
    const events: MotionEvent[] = [];
    for (const rule of this.prescription.rules) {
      const value = observedValue(rule.metric, measurements, timing);
      const state = this.states.get(rule.id) ?? "clear";
      const feature = measurements.features.find(
        (candidate) => candidate.metric === rule.metric,
      );
      const qualityGood =
        rule.scope.kind === "temporal" || feature?.quality === "good";

      if (
        rule.scope.kind !== "temporal" &&
        (!qualityGood || value === undefined)
      ) {
        continue;
      }

      if (
        rule.kind === "maximum" &&
        rule.hard &&
        rule.value !== undefined &&
        value !== undefined
      ) {
        const approachAt = rule.value - 10;
        if (value >= rule.value && state !== "reached") {
          this.states.set(rule.id, "reached");
          events.push(
            motionEvent(
              "restriction-reached",
              timestampMs,
              `Relay’s estimate reached your ${rule.value}${
                rule.unit === "degrees" ? "°" : ` ${rule.unit}`
              } ${bodyLabel(rule)} limit. Stop moving farther.`,
              value,
              measurements.confidence.score,
              rule,
              feature?.quality,
            ),
          );
        } else if (
          value >= approachAt &&
          value < rule.value &&
          state === "clear"
        ) {
          this.states.set(rule.id, "approaching");
          events.push(
            motionEvent(
              "restriction-approaching",
              timestampMs,
              `Approaching your ${rule.value}${
                rule.unit === "degrees" ? "°" : ` ${rule.unit}`
              } ${bodyLabel(rule)} maximum. Ease up.`,
              value,
              measurements.confidence.score,
              rule,
              feature?.quality,
            ),
          );
        } else if (value < approachAt - 4) {
          this.states.set(rule.id, "clear");
        }
        continue;
      }

      if (
        rule.kind === "target" &&
        value !== undefined &&
        movementTargetSatisfied(rule, value) &&
        state !== "met"
      ) {
        this.states.set(rule.id, "met");
        events.push(
          motionEvent(
            "target-reached",
            timestampMs,
            `${rule.label}. Return with control.`,
            value,
            measurements.confidence.score,
            rule,
            feature?.quality,
          ),
        );
        continue;
      }

      if (
        rule.kind === "minimum" &&
        rule.scope.kind !== "temporal" &&
        value !== undefined &&
        movementTargetSatisfied(rule, value) &&
        state !== "met"
      ) {
        this.states.set(rule.id, "met");
        events.push(
          motionEvent(
            "target-reached",
            timestampMs,
            `${rule.label}. Return with control.`,
            value,
            measurements.confidence.score,
            rule,
            feature?.quality,
          ),
        );
        continue;
      }

      if (
        rule.kind === "range" &&
        value !== undefined &&
        movementTargetSatisfied(rule, value) &&
        state !== "met"
      ) {
        this.states.set(rule.id, "met");
        events.push(
          motionEvent(
            "target-reached",
            timestampMs,
            `${rule.label}.`,
            value,
            measurements.confidence.score,
            rule,
            feature?.quality,
          ),
        );
        continue;
      }
      if (
        rule.kind === "range" &&
        rule.min !== undefined &&
        rule.max !== undefined &&
        value !== undefined &&
        value > rule.max &&
        state !== "observed"
      ) {
        this.states.set(rule.id, "observed");
        events.push(
          motionEvent(
            "posture-observation",
            timestampMs,
            `${rule.label}: the estimate moved above the confirmed range.`,
            value,
            measurements.confidence.score,
            rule,
            feature?.quality,
          ),
        );
        continue;
      }

      const broadVisibleCondition =
        !rule.hard &&
        (rule.kind === "visible-condition" ||
          rule.kind === "maximum");
      if (
        broadVisibleCondition &&
        rule.value !== undefined &&
        value !== undefined
      ) {
        if (value >= rule.value) {
          const since = this.exceededSince.get(rule.id) ?? timestampMs;
          this.exceededSince.set(rule.id, since);
          if (timestampMs - since >= 300 && state !== "observed") {
            this.states.set(rule.id, "observed");
            events.push(
              motionEvent(
                "posture-observation",
                timestampMs,
                `${rule.label} appears to be changing. Reset if needed.`,
                value,
                measurements.confidence.score,
                rule,
                feature?.quality,
              ),
            );
          }
        } else if (value < rule.value - 4) {
          this.exceededSince.delete(rule.id);
          this.states.set(rule.id, "clear");
        }
      }
    }

    return events;
  }

  evaluateCompletion(
    durationMs: number,
    holdMs: number,
    timestampMs: number,
  ): MotionEvent[] {
    const events: MotionEvent[] = [];
    for (const rule of this.prescription.rules) {
      if (rule.value === undefined) continue;
      if (rule.metric === "rep-duration" && rule.kind === "minimum") {
        const seconds = durationMs / 1000;
        events.push(
          motionEvent(
            seconds >= rule.value ? "tempo-met" : "tempo-missed",
            timestampMs,
            seconds >= rule.value
              ? `Minimum ${rule.value}-second movement tempo met.`
              : `This repetition was faster than the ${rule.value}-second minimum.`,
            seconds,
            undefined,
            rule,
            "good",
          ),
        );
      }
      if (rule.metric === "rep-duration" && rule.kind === "maximum") {
        const seconds = durationMs / 1000;
        events.push(
          motionEvent(
            seconds <= rule.value ? "tempo-met" : "tempo-missed",
            timestampMs,
            seconds <= rule.value
              ? `Maximum ${rule.value}-second movement tempo met.`
              : `This repetition took longer than the ${rule.value}-second maximum.`,
            seconds,
            undefined,
            rule,
            "good",
          ),
        );
      }
      if (rule.metric === "hold-duration" && rule.kind === "minimum") {
        const seconds = holdMs / 1000;
        events.push(
          motionEvent(
            seconds >= rule.value ? "hold-met" : "hold-short",
            timestampMs,
            seconds >= rule.value
              ? `Minimum ${rule.value}-second hold met.`
              : `The observed hold was shorter than ${rule.value} seconds.`,
            seconds,
            undefined,
            rule,
            "good",
          ),
        );
      }
    }
    return events;
  }

  resetAttempt(): void {
    this.states.forEach((state, key) => {
      if (state !== "clear") this.states.set(key, "clear");
    });
    this.exceededSince.clear();
  }
}
