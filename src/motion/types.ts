import type {
  BodySide,
  ExerciseId,
  ExercisePrescription,
  MetricId,
} from "../domain/types";

export type JointName =
  | "nose"
  | "left-eye-inner"
  | "left-eye"
  | "left-eye-outer"
  | "right-eye-inner"
  | "right-eye"
  | "right-eye-outer"
  | "left-ear"
  | "right-ear"
  | "mouth-left"
  | "mouth-right"
  | "left-shoulder"
  | "right-shoulder"
  | "left-elbow"
  | "right-elbow"
  | "left-wrist"
  | "right-wrist"
  | "left-pinky"
  | "right-pinky"
  | "left-index"
  | "right-index"
  | "left-thumb"
  | "right-thumb"
  | "left-hip"
  | "right-hip"
  | "left-knee"
  | "right-knee"
  | "left-ankle"
  | "right-ankle"
  | "left-heel"
  | "right-heel"
  | "left-foot-index"
  | "right-foot-index";

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility: number;
  presence: number;
}

export type LandmarkSet = Partial<Record<JointName, PoseLandmark>>;

export interface PoseFrame {
  timestampMs: number;
  image: LandmarkSet;
  world: LandmarkSet;
  frameWidth: number;
  frameHeight: number;
  source: "live" | "replay";
}

export type TrackingStatus = "good" | "limited" | "lost";

export interface MeasurementConfidence {
  score: number;
  status: TrackingStatus;
  canAssessBodyFeatures: boolean;
  canAssessAngle: boolean;
  reasons: string[];
}

export interface MotionMeasurements {
  kneeFlexion?: number;
  kneeFlexionImage?: number;
  kneeFlexionWorld?: number;
  angleAgreementDelta?: number;
  legLift?: number;
  extensionProgress?: number;
  kneeBend?: number;
  trunkCompensation?: number;
  /** Exercise-independent values keyed by the confirmed constraint metric. */
  values: Partial<Record<MetricId, number>>;
  features: BodyPoseFeature[];
  confidence: MeasurementConfidence;
}

export interface BodyPoseFeature {
  id: string;
  metric: MetricId;
  joints: JointName[];
  value?: number;
  unit: "degrees" | "percent";
  quality: TrackingStatus;
  description: string;
}

export type ExercisePhase =
  | "ready"
  | "moving"
  | "holding"
  | "returning"
  | "paused";

export type MotionEventType =
  | "rep-started"
  | "hold-started"
  | "hold-met"
  | "rep-completed"
  | "incomplete-rep"
  | "target-reached"
  | "restriction-approaching"
  | "restriction-reached"
  | "tracking-lost"
  | "tracking-restored"
  | "posture-observation"
  | "tempo-met"
  | "tempo-missed"
  | "hold-short";

export interface MotionEvent {
  id: string;
  type: MotionEventType;
  timestampMs: number;
  message: string;
  value?: number;
  confidence?: number;
  ruleId?: string;
  metric?: MetricId;
  measurementQuality?: TrackingStatus;
}

export interface BodyFeaturePeak {
  metric: MetricId;
  joints: JointName[];
  value: number;
  unit: "degrees" | "percent";
  quality: TrackingStatus;
  assessableCoverage: number;
  description: string;
}

export interface RepRecord {
  rep: number;
  startedAtMs: number;
  completedAtMs: number;
  durationMs: number;
  holdMs: number;
  maxKneeFlexion?: number;
  maxLegLift?: number;
  maxExtensionProgress?: number;
  trackingGoodRatio: number;
  targetReached: boolean;
  restrictionApproached: boolean;
  restrictionReached: boolean;
  observations: string[];
  /**
   * Provider-extensible peak observations. New body features can be recorded
   * without changing the repetition schema.
   */
  featurePeaks: BodyFeaturePeak[];
}

export interface TemporalPhaseEvidence {
  phase: ExercisePhase;
  confidence: number;
  margin: number;
  applicable: boolean;
  modelId: string;
}

export interface EngineOutput {
  exerciseId: ExerciseId;
  side: BodySide;
  phase: ExercisePhase;
  repsCompleted: number;
  measurements: MotionMeasurements;
  holdElapsedMs: number;
  repElapsedMs: number;
  coachMessage: string;
  events: MotionEvent[];
  repRecords: RepRecord[];
  temporalEvidence?: TemporalPhaseEvidence;
}

export interface MotionProviderStatus {
  state:
    | "idle"
    | "loading"
    | "ready"
    | "running"
    | "permission-denied"
    | "unsupported"
    | "error";
  label: string;
  detail?: string;
}

export type FrameHandler = (frame: PoseFrame) => void;
export type StatusHandler = (status: MotionProviderStatus) => void;

export interface MotionProvider {
  readonly id: "mediapipe" | "replay";
  initialize(): Promise<void>;
  start(
    video: HTMLVideoElement | null,
    onFrame: FrameHandler,
    onStatus?: StatusHandler,
  ): Promise<void>;
  stop(): Promise<void>;
  dispose?(): void;
}

export interface ExerciseMetricContext {
  prescription: ExercisePrescription;
  frame: PoseFrame;
}
