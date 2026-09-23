import type { ExercisePrescription } from "../domain/types";
import { createId } from "../domain/types";
import { measurePose } from "./geometry";
import { ConstraintEvaluator } from "./constraint-evaluator";
import type {
  BodyFeaturePeak,
  EngineOutput,
  ExercisePhase,
  MotionEvent,
  MotionMeasurements,
  PoseFrame,
  RepRecord,
  TemporalPhaseEvidence,
} from "./types";

interface FeaturePeakAccumulator {
  metric: BodyFeaturePeak["metric"];
  joints: BodyFeaturePeak["joints"];
  value: number;
  unit: BodyFeaturePeak["unit"];
  description: string;
  goodFrames: number;
  totalFrames: number;
}

interface ActiveRep {
  startedAtMs: number;
  observedDurationMs: number;
  lastGoodTimestampMs?: number;
  peak: number;
  reachedEndpoint: boolean;
  incompletePrompted: boolean;
  holdStartedAtMs?: number;
  holdMs: number;
  goodFrames: number;
  totalFrames: number;
  targetReached: boolean;
  restrictionApproached: boolean;
  restrictionReached: boolean;
  observations: string[];
  featurePeaks: Map<string, FeaturePeakAccumulator>;
}

interface NeutralCalibration {
  timestampMs: number;
  kneeFlexion?: number;
  legLift?: number;
  trunkCompensation?: number;
}

function event(
  type: MotionEvent["type"],
  timestampMs: number,
  message: string,
  value?: number,
  confidence?: number,
): MotionEvent {
  return {
    id: createId("event"),
    type,
    timestampMs,
    message,
    value,
    confidence,
  };
}

function metricFor(
  prescription: ExercisePrescription,
  measurements: MotionMeasurements,
): number | undefined {
  if (prescription.exerciseId === "straight-leg-raise")
    return measurements.legLift;
  if (prescription.exerciseId === "seated-knee-extension")
    return measurements.extensionProgress;
  return measurements.kneeFlexion;
}

function thresholds(prescription: ExercisePrescription): {
  start: number;
  top: number;
  returned: number;
} {
  if (prescription.exerciseId === "straight-leg-raise") {
    return { start: 10, top: 25, returned: 7 };
  }
  if (prescription.exerciseId === "seated-knee-extension") {
    return { start: 20, top: 60, returned: 12 };
  }
  return { start: 20, top: 40, returned: 14 };
}

export class ExerciseEngine {
  private phase: ExercisePhase = "ready";
  private previousMetric?: number;
  private previousTimestamp?: number;
  private activeRep?: ActiveRep;
  private records: RepRecord[] = [];
  private wasTrackingGood = true;
  private constraintEvaluator: ConstraintEvaluator;
  private calibrationSamples: NeutralCalibration[] = [];
  private calibration?: NeutralCalibration;

  constructor(private readonly prescription: ExercisePrescription) {
    this.constraintEvaluator = new ConstraintEvaluator(prescription);
  }

  reset(): void {
    this.phase = "ready";
    this.previousMetric = undefined;
    this.previousTimestamp = undefined;
    this.activeRep = undefined;
    this.records = [];
    this.wasTrackingGood = true;
    this.calibrationSamples = [];
    this.calibration = undefined;
    this.constraintEvaluator.resetAttempt();
  }

  pause(): void {
    if (this.activeRep) this.activeRep.lastGoodTimestampMs = undefined;
    this.previousMetric = undefined;
    this.previousTimestamp = undefined;
    this.phase = "paused";
  }

  process(
    frame: PoseFrame,
    temporalEvidence?: TemporalPhaseEvidence,
  ): EngineOutput {
    const rawMeasurements = measurePose(
      frame,
      this.prescription.exerciseId,
      this.prescription.side,
    );
    if (!this.calibration) {
      const calibrationMessage = this.updateCalibration(
        rawMeasurements,
        frame.timestampMs,
      );
      if (calibrationMessage) {
        const primaryMetric =
          this.prescription.exerciseId === "straight-leg-raise"
            ? "leg-lift"
            : this.prescription.exerciseId === "seated-knee-extension"
              ? "extension-progress"
              : "knee-flexion";
        const primaryFeature = rawMeasurements.features.find(
          (feature) => feature.metric === primaryMetric,
        );
        const calibrationEvents: MotionEvent[] = [];
        if (primaryFeature?.quality !== "good" && this.wasTrackingGood) {
          this.wasTrackingGood = false;
          calibrationEvents.push(
            event(
              "tracking-lost",
              frame.timestampMs,
              "Movement assessment paused. Reposition so the required joints are visible.",
              undefined,
              rawMeasurements.confidence.score,
            ),
          );
        }
        return this.output(
          rawMeasurements,
          frame.timestampMs,
          calibrationEvents,
          calibrationMessage,
          temporalEvidence,
        );
      }
      this.wasTrackingGood = true;
    }
    const measurements = this.applyCalibration(rawMeasurements);
    const events: MotionEvent[] = [];
    const currentMetric = metricFor(this.prescription, measurements);
    const primaryMetric =
      this.prescription.exerciseId === "straight-leg-raise"
        ? "leg-lift"
        : this.prescription.exerciseId === "seated-knee-extension"
          ? "extension-progress"
          : "knee-flexion";
    const primaryFeature = measurements.features.find(
      (feature) => feature.metric === primaryMetric,
    );
    const currentGood =
      currentMetric !== undefined && primaryFeature?.quality === "good";
    const now = frame.timestampMs;

    if (!currentGood) {
      if (this.wasTrackingGood) {
        events.push(
          event(
            "tracking-lost",
            now,
            measurements.confidence.canAssessBodyFeatures
              ? "Some movement checks are limited."
              : "Movement assessment paused. Reposition so the required joints are visible.",
            undefined,
            measurements.confidence.score,
          ),
        );
      }
      this.wasTrackingGood = false;
      if (this.activeRep) {
        this.activeRep.totalFrames += 1;
        this.activeRep.lastGoodTimestampMs = undefined;
        this.activeRep.featurePeaks.forEach((peak) => {
          peak.totalFrames += 1;
        });
      }
      this.phase = "paused";
      return this.output(
        measurements,
        now,
        events,
        measurements.confidence.reasons[0] ??
          "Reposition so Relay can assess the movement.",
        temporalEvidence,
      );
    }

    if (!this.wasTrackingGood) {
      events.push(
        event(
          "tracking-restored",
          now,
          "Tracking is back. Continue when ready.",
          undefined,
          measurements.confidence.score,
        ),
      );
      this.phase = this.activeRep ? "moving" : "ready";
    }
    this.wasTrackingGood = true;

    let observedDeltaMs = 0;
    if (this.activeRep) {
      observedDeltaMs =
        this.activeRep.lastGoodTimestampMs === undefined
          ? 0
          : Math.max(0, now - this.activeRep.lastGoodTimestampMs);
      this.activeRep.lastGoodTimestampMs = now;
      this.activeRep.observedDurationMs += observedDeltaMs;
      this.activeRep.goodFrames += 1;
      this.activeRep.totalFrames += 1;
      this.updateFeaturePeaks(measurements);
    }

    const constraintEvents = this.constraintEvaluator.evaluateFrame(
      measurements,
      now,
      {
        repElapsedMs: this.activeRep
          ? Math.max(0, now - this.activeRep.startedAtMs)
          : 0,
        holdElapsedMs: this.activeRep?.holdMs ?? 0,
      },
    );
    events.push(...constraintEvents);
    if (this.activeRep) {
      if (
        constraintEvents.some(
          (item) => item.type === "restriction-approaching",
        )
      ) {
        this.activeRep.restrictionApproached = true;
      }
      if (
        constraintEvents.some((item) => item.type === "restriction-reached")
      ) {
        this.activeRep.restrictionReached = true;
      }
      if (constraintEvents.some((item) => item.type === "target-reached")) {
        this.activeRep.targetReached = true;
      }
      constraintEvents
        .filter((item) => item.type === "posture-observation")
        .forEach((item) => this.activeRep?.observations.push(item.message));
    }

    if (currentMetric !== undefined) {
      this.advanceState(
        currentMetric,
        measurements,
        now,
        observedDeltaMs,
        events,
        temporalEvidence,
      );
    }

    this.previousMetric = currentMetric;
    this.previousTimestamp = now;

    const coachMessage = this.coachMessage(events, measurements);
    return this.output(
      measurements,
      now,
      events,
      coachMessage,
      temporalEvidence,
    );
  }

  private updateCalibration(
    measurements: MotionMeasurements,
    timestampMs: number,
  ): string | undefined {
    const primary = metricFor(this.prescription, measurements);
    const primaryMetric =
      this.prescription.exerciseId === "straight-leg-raise"
        ? "leg-lift"
        : this.prescription.exerciseId === "seated-knee-extension"
          ? "extension-progress"
          : "knee-flexion";
    const primaryFeature = measurements.features.find(
      (feature) => feature.metric === primaryMetric,
    );
    if (primary === undefined || primaryFeature?.quality !== "good") {
      this.calibrationSamples = [];
      return "Hold your starting position in view while Relay calibrates.";
    }

    this.calibrationSamples.push({
      timestampMs,
      kneeFlexion: measurements.kneeFlexion,
      legLift: measurements.legLift,
      trunkCompensation: measurements.trunkCompensation,
    });
    this.calibrationSamples = this.calibrationSamples.filter(
      (sample) => sample.timestampMs >= timestampMs - 2_000,
    );
    const elapsed =
      timestampMs - (this.calibrationSamples[0]?.timestampMs ?? timestampMs);
    const values = (
      key: "kneeFlexion" | "legLift" | "trunkCompensation",
    ) =>
      this.calibrationSamples
        .map((sample) => sample[key])
        .filter((value): value is number => value !== undefined);
    const spread = (key: "kneeFlexion" | "legLift" | "trunkCompensation") => {
      const observed = values(key);
      return observed.length > 1
        ? Math.max(...observed) - Math.min(...observed)
        : 0;
    };
    const stable =
      spread("kneeFlexion") <= 5 &&
      (this.prescription.exerciseId !== "straight-leg-raise" ||
        (spread("legLift") <= 4 && spread("trunkCompensation") <= 6));
    if (!stable) {
      this.calibrationSamples = this.calibrationSamples.slice(-2);
      return "Stay still in the starting position so Relay can set a stable baseline.";
    }
    if (elapsed < 1_500 || this.calibrationSamples.length < 8) {
      return "Hold the starting position while Relay checks a stable baseline.";
    }

    const average = (
      key: "kneeFlexion" | "legLift" | "trunkCompensation",
    ) => {
      const observed = values(key);
      return observed.length
        ? observed.reduce((sum, value) => sum + value, 0) / observed.length
        : undefined;
    };
    this.calibration = {
      timestampMs,
      kneeFlexion: average("kneeFlexion"),
      legLift: average("legLift"),
      trunkCompensation: average("trunkCompensation"),
    };
    return undefined;
  }

  private applyCalibration(
    measurements: MotionMeasurements,
  ): MotionMeasurements {
    if (!this.calibration) return measurements;
    const values = { ...measurements.values };
    const features = measurements.features.map((feature) => ({ ...feature }));
    let legLift = measurements.legLift;
    let extensionProgress = measurements.extensionProgress;
    let kneeBend = measurements.kneeBend;
    let trunkCompensation = measurements.trunkCompensation;

    if (this.prescription.exerciseId === "straight-leg-raise") {
      legLift =
        measurements.legLift !== undefined &&
        this.calibration.legLift !== undefined
          ? Math.max(0, measurements.legLift - this.calibration.legLift)
          : undefined;
      kneeBend =
        measurements.kneeFlexion !== undefined &&
        this.calibration.kneeFlexion !== undefined
          ? Math.max(
              0,
              measurements.kneeFlexion - this.calibration.kneeFlexion,
            )
          : undefined;
      trunkCompensation =
        measurements.trunkCompensation !== undefined &&
        this.calibration.trunkCompensation !== undefined
          ? Math.abs(
              measurements.trunkCompensation -
                this.calibration.trunkCompensation,
            )
          : undefined;
      values["leg-lift"] = legLift;
      values["knee-bend"] = kneeBend;
      values["trunk-compensation"] = trunkCompensation;
    }

    if (
      this.prescription.exerciseId === "seated-knee-extension" &&
      measurements.kneeFlexion !== undefined &&
      this.calibration.kneeFlexion !== undefined &&
      this.calibration.kneeFlexion > 5
    ) {
      extensionProgress = Math.min(
        100,
        Math.max(
          0,
          ((this.calibration.kneeFlexion - measurements.kneeFlexion) /
            this.calibration.kneeFlexion) *
            100,
        ),
      );
      values["extension-progress"] = extensionProgress;
    }

    for (const feature of features) {
      if (feature.metric === "leg-lift") feature.value = legLift;
      if (feature.metric === "extension-progress")
        feature.value = extensionProgress;
      if (feature.metric === "knee-bend") feature.value = kneeBend;
      if (feature.metric === "trunk-compensation")
        feature.value = trunkCompensation;
    }

    return {
      ...measurements,
      legLift,
      extensionProgress,
      kneeBend,
      trunkCompensation,
      values,
      features,
    };
  }

  private updateFeaturePeaks(measurements: MotionMeasurements): void {
    if (!this.activeRep) return;
    this.activeRep.featurePeaks.forEach((peak) => {
      peak.totalFrames += 1;
    });
    measurements.features.forEach((feature) => {
      if (feature.quality !== "good" || feature.value === undefined) return;
      const existing = this.activeRep?.featurePeaks.get(feature.metric);
      if (existing) {
        existing.value = Math.max(existing.value, feature.value);
        existing.goodFrames += 1;
        return;
      }
      this.activeRep?.featurePeaks.set(feature.metric, {
        metric: feature.metric,
        joints: feature.joints,
        value: feature.value,
        unit: feature.unit,
        description: feature.description,
        goodFrames: 1,
        totalFrames: Math.max(1, this.activeRep.totalFrames),
      });
    });
  }

  private advanceState(
    metric: number,
    measurements: MotionMeasurements,
    now: number,
    observedDeltaMs: number,
    events: MotionEvent[],
    temporalEvidence?: TemporalPhaseEvidence,
  ): void {
    const limits = thresholds(this.prescription);
    const previous = this.previousMetric ?? metric;
    const delta = metric - previous;
    const elapsed = Math.max(1, now - (this.previousTimestamp ?? now));
    const velocity = (delta / elapsed) * 1000;

    if (!this.activeRep && metric >= limits.start) {
      this.activeRep = {
        startedAtMs: now,
        observedDurationMs: 0,
        lastGoodTimestampMs: now,
        peak: metric,
        reachedEndpoint: false,
        incompletePrompted: false,
        holdMs: 0,
        goodFrames: 1,
        totalFrames: 1,
        targetReached: false,
        restrictionApproached: false,
        restrictionReached: false,
        observations: [],
        featurePeaks: new Map(),
      };
      this.updateFeaturePeaks(measurements);
      this.phase = "moving";
      events.push(event("rep-started", now, "Movement started."));
    }

    if (!this.activeRep) {
      this.phase = "ready";
      return;
    }

    this.activeRep.peak = Math.max(this.activeRep.peak, metric);
    if (metric >= limits.top) this.activeRep.reachedEndpoint = true;
    const learnedHoldEvidence =
      temporalEvidence?.applicable === true &&
      temporalEvidence.phase === "holding";
    const atTop =
      metric >= limits.top &&
      (Math.abs(velocity) < 8 || learnedHoldEvidence);
    if (atTop) {
      if (this.activeRep.holdStartedAtMs === undefined) {
        this.activeRep.holdStartedAtMs = now;
        events.push(event("hold-started", now, "Hold."));
      } else {
        this.activeRep.holdMs += observedDeltaMs;
      }
      this.phase = "holding";
      const required = this.prescription.dose.holdSeconds * 1000;
      const justMet =
        required > 0 &&
        this.activeRep.holdMs >= required &&
        this.activeRep.holdMs - elapsed < required;
      if (justMet) events.push(event("hold-met", now, "Hold complete."));
    } else if (
      this.activeRep.peak - metric >= 7 ||
      (temporalEvidence?.applicable === true &&
        temporalEvidence.phase === "returning" &&
        this.activeRep.peak - metric >= 4) ||
      (delta < -1 && this.phase === "holding")
    ) {
      this.phase = "returning";
    } else if (this.phase !== "returning") {
      this.phase = "moving";
    }

    if (
      this.phase === "returning" &&
      metric <= limits.returned &&
      !this.activeRep.reachedEndpoint
    ) {
      events.push(
        event(
          "incomplete-rep",
          now,
          "Movement returned before reaching the exercise endpoint.",
          metric,
          measurements.confidence.score,
        ),
      );
      this.activeRep = undefined;
      this.phase = "ready";
      this.constraintEvaluator.resetAttempt();
      return;
    }

    if (
      this.phase === "returning" &&
      metric <= limits.returned &&
      this.activeRep.reachedEndpoint
    ) {
      const completedAtMs = now;
      const completionEvents = this.constraintEvaluator.evaluateCompletion(
        this.activeRep.observedDurationMs,
        this.activeRep.holdMs,
        now,
      );
      const hasHoldRule = this.prescription.rules.some(
        (rule) => rule.metric === "hold-duration",
      );
      const hasTempoRule = this.prescription.rules.some(
        (rule) => rule.metric === "rep-duration",
      );
      if (
        !hasHoldRule &&
        this.prescription.dose.holdSeconds > 0 &&
        this.activeRep.holdMs <
          this.prescription.dose.holdSeconds * 1000
      ) {
        completionEvents.push(
          event(
            "hold-short",
            now,
            `The observed hold was shorter than ${this.prescription.dose.holdSeconds} seconds.`,
            this.activeRep.holdMs / 1000,
          ),
        );
      }
      const tempoMinimum =
        this.prescription.dose.tempo.minRepDurationSeconds;
      if (!hasTempoRule && tempoMinimum !== undefined) {
        const visibleSeconds = this.activeRep.observedDurationMs / 1000;
        completionEvents.push(
          event(
            visibleSeconds >= tempoMinimum ? "tempo-met" : "tempo-missed",
            now,
            visibleSeconds >= tempoMinimum
              ? `Minimum ${tempoMinimum}-second movement tempo met.`
              : `This repetition was faster than the ${tempoMinimum}-second minimum.`,
            visibleSeconds,
          ),
        );
      }
      const record: RepRecord = {
        rep: this.records.length + 1,
        startedAtMs: this.activeRep.startedAtMs,
        completedAtMs,
        durationMs: this.activeRep.observedDurationMs,
        holdMs: this.activeRep.holdMs,
        maxKneeFlexion:
          this.prescription.exerciseId === "heel-slide"
            ? this.activeRep.peak
            : measurements.kneeFlexion,
        maxLegLift:
          this.prescription.exerciseId === "straight-leg-raise"
            ? this.activeRep.peak
            : undefined,
        maxExtensionProgress:
          this.prescription.exerciseId === "seated-knee-extension"
            ? this.activeRep.peak
            : undefined,
        trackingGoodRatio:
          this.activeRep.goodFrames / Math.max(1, this.activeRep.totalFrames),
        targetReached: this.activeRep.targetReached,
        restrictionApproached: this.activeRep.restrictionApproached,
        restrictionReached: this.activeRep.restrictionReached,
        observations: [...this.activeRep.observations],
        featurePeaks: Array.from(this.activeRep.featurePeaks.values()).map(
          (peak) => {
            const assessableCoverage =
              peak.goodFrames / Math.max(1, peak.totalFrames);
            return {
              metric: peak.metric,
              joints: peak.joints,
              value: peak.value,
              unit: peak.unit,
              quality: assessableCoverage >= 0.9 ? "good" : "limited",
              assessableCoverage,
              description: peak.description,
            };
          },
        ),
      };
      events.push(...completionEvents);
      this.records.push(record);
      events.push(
        event(
          "rep-completed",
          now,
          `Rep ${record.rep} complete.`,
          record.rep,
        ),
      );
      this.activeRep = undefined;
      this.phase = "ready";
      this.constraintEvaluator.resetAttempt();
      return;
    }

    if (
      this.activeRep &&
      this.activeRep.observedDurationMs > 15000 &&
      !this.activeRep.incompletePrompted
    ) {
      events.push(
        event(
          "incomplete-rep",
          now,
          this.phase === "returning"
            ? "Return fully to your starting position to complete this rep."
            : "Complete the movement or return to your starting position.",
        ),
      );
      this.activeRep.incompletePrompted = true;
    }
  }

  private coachMessage(
    events: MotionEvent[],
    measurements: MotionMeasurements,
  ): string {
    const priority: MotionEvent["type"][] = [
      "restriction-reached",
      "tracking-lost",
      "restriction-approaching",
      "posture-observation",
      "incomplete-rep",
      "hold-short",
      "tempo-missed",
      "rep-completed",
      "hold-met",
      "tempo-met",
      "target-reached",
      "tracking-restored",
    ];
    for (const type of priority) {
      const match = [...events].reverse().find((item) => item.type === type);
      if (match) return match.message;
    }
    if (measurements.confidence.status !== "good")
      return measurements.confidence.reasons[0] ?? "Reposition the camera.";
    if (this.phase === "holding") {
      const hold = this.prescription.dose.holdSeconds;
      return hold > 0 ? `Hold for ${hold} seconds.` : "Pause with control.";
    }
    if (this.phase === "returning") return "Return to your starting position.";
    if (this.phase === "moving")
      return this.prescription.exerciseId === "heel-slide"
        ? "Slide your heel in slowly."
        : this.prescription.exerciseId === "straight-leg-raise"
          ? "Lift with the knee visually straight."
          : "Straighten your knee with control.";
    return "Ready when you are.";
  }

  private output(
    measurements: MotionMeasurements,
    now: number,
    events: MotionEvent[],
    coachMessage: string,
    temporalEvidence?: TemporalPhaseEvidence,
  ): EngineOutput {
    return {
      exerciseId: this.prescription.exerciseId,
      side: this.prescription.side,
      phase: this.phase,
      repsCompleted: this.records.length,
      measurements,
      holdElapsedMs: this.activeRep?.holdMs ?? 0,
      repElapsedMs: this.activeRep
        ? Math.max(0, now - this.activeRep.startedAtMs)
        : 0,
      coachMessage,
      events,
      repRecords: [...this.records],
      temporalEvidence,
    };
  }
}
