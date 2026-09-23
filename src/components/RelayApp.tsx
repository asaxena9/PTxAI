"use client";

import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Camera,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  Download,
  FileText,
  FlaskConical,
  HeartPulse,
  History,
  LockKeyhole,
  Mic,
  MoreHorizontal,
  Pause,
  Play,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Upload,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  Clarification,
  ConfirmedSession,
  ExerciseId,
  ExercisePrescription,
  MovementRule,
} from "../domain/types";
import {
  EXERCISE_LABELS,
  constraintScopeForMetric,
  createId,
} from "../domain/types";
import {
  REQUIRED_SCENARIO_TEXT,
  createDemoSession,
  createManualSession,
} from "../domain/templates";
import {
  applyClarification,
  interpretInstructions,
} from "../intelligence/interpreter";
import { extractInstructionsFromFile } from "../intelligence/upload";
import { confirmedSessionSchema } from "../domain/schema";
import {
  MediaPipeMotionProvider,
  type CameraFacing,
} from "../motion/mediapipe-provider";
import { measurePose } from "../motion/geometry";
import { ExerciseEngine } from "../motion/exercise-engine";
import {
  REPLAY_SCENARIOS,
  createReplayFrames,
  type ReplayScenarioId,
} from "../motion/replay-fixtures";
import { ReplayMotionProvider } from "../motion/replay-provider";
import { evaluateSetupFeatureReadiness } from "../motion/setup-quality";
import {
  advanceTemporalSampleSchedule,
  loadTemporalMovementModel,
  type TemporalMovementModel,
  type TemporalPrediction,
} from "../motion/temporal-model";
import type {
  EngineOutput,
  JointName,
  MotionEvent,
  MotionMeasurements,
  MotionProvider,
  MotionProviderStatus,
  PoseFrame,
} from "../motion/types";
import { useSpokenCoach } from "../hooks/use-spoken-coach";
import {
  createRunResult,
  exportRunJson,
  summarizeRun,
  type ExerciseRunResult,
  type ExerciseSummary,
  type SessionRunResult,
} from "../session/results";
import {
  clearHistory,
  deleteRun,
  loadHistory,
  trySaveRun,
} from "../session/storage";

type Screen =
  | "home"
  | "clarify"
  | "review"
  | "camera"
  | "active"
  | "summary"
  | "history"
  | "privacy";

type SessionMode = "live" | "replay";

const INITIAL_TEXT = REQUIRED_SCENARIO_TEXT;

function Brand({ inverse = false }: { inverse?: boolean }) {
  return (
    <div className={`brand ${inverse ? "brand-inverse" : ""}`}>
      <span className="brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span>Relay</span>
    </div>
  );
}

function IconButton({
  label,
  onClick,
  children,
  className = "",
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      className={`icon-button ${className}`}
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      {children}
    </button>
  );
}

function ProgressRail({
  step,
}: {
  step: "tell" | "check" | "position" | "move" | "review";
}) {
  const steps = ["tell", "check", "position", "move", "review"] as const;
  const active = steps.indexOf(step);
  return (
    <div className="progress-rail" aria-label={`Step ${active + 1} of 5`}>
      {steps.map((item, index) => (
        <span
          key={item}
          className={index <= active ? "complete" : ""}
          aria-hidden="true"
        />
      ))}
    </div>
  );
}

function AppHeader({
  onHome,
  onHistory,
  onPrivacy,
  compact = false,
}: {
  onHome: () => void;
  onHistory: () => void;
  onPrivacy: () => void;
  compact?: boolean;
}) {
  return (
    <header className={`app-header ${compact ? "compact" : ""}`}>
      <button className="brand-button" type="button" onClick={onHome}>
        <Brand />
      </button>
      <nav aria-label="Relay">
        <button
          type="button"
          onClick={onHistory}
          aria-label="Session history"
        >
          <History size={18} aria-hidden="true" />
          <span>History</span>
        </button>
        <button
          type="button"
          onClick={onPrivacy}
          aria-label="Privacy and product boundary"
        >
          <LockKeyhole size={18} aria-hidden="true" />
          <span>Privacy</span>
        </button>
      </nav>
    </header>
  );
}

function PrivacyPromise() {
  return (
    <div className="privacy-promise">
      <ShieldCheck size={18} aria-hidden="true" />
      <span>
        No account. Camera video is processed on this device and isn’t saved.
      </span>
    </div>
  );
}

function HomeScreen({
  input,
  setInput,
  inputSource,
  onInterpret,
  onDemo,
  onManual,
  onUpload,
  uploadState,
  onSpeak,
  listening,
  history,
  onOpenHistory,
  onPrivacy,
}: {
  input: string;
  setInput: (value: string) => void;
  inputSource: "typed" | "spoken";
  onInterpret: (source?: "typed" | "spoken" | "upload") => void;
  onDemo: () => void;
  onManual: () => void;
  onUpload: (file: File) => void;
  uploadState: string;
  onSpeak: () => void;
  listening: boolean;
  history: SessionRunResult[];
  onOpenHistory: () => void;
  onPrivacy: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  return (
    <main className="home-screen">
      <AppHeader
        onHome={() => undefined}
        onHistory={onOpenHistory}
        onPrivacy={onPrivacy}
      />
      <section className="home-hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <Sparkles size={16} aria-hidden="true" />
            Camera-guided home movement
          </div>
          <h1>
            Your session,
            <br />
            <em>understood.</em>
          </h1>
          <p>
            Tell Relay the exercise plan you already have. It turns the plan
            into a calm, hands-free session you can review before you move.
          </p>
          <PrivacyPromise />
        </div>

        <div className="composer-card">
          <div className="composer-label">
            <span>What do you need to complete?</span>
            <span className="supported-pill">
              Full-body beta · 3 exercises
            </span>
          </div>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Try: Right heel slides, 3 sets of 10, hold for 2 seconds, and do not bend beyond 90 degrees."
            aria-label="Exercise instructions"
          />
          <div className="composer-toolbar">
            <div className="input-actions">
              <button
                type="button"
                className={listening ? "listening" : ""}
                onClick={onSpeak}
              >
                <Mic size={19} aria-hidden="true" />
                {listening ? "Listening…" : "Speak"}
              </button>
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                disabled={Boolean(uploadState)}
              >
                <Upload size={19} aria-hidden="true" />
                {uploadState || "Upload plan"}
              </button>
              <input
                ref={fileRef}
                hidden
                type="file"
                accept=".txt,.md,.pdf,image/*"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onUpload(file);
                  event.currentTarget.value = "";
                }}
                aria-label="Upload a plan"
              />
            </div>
            <button
              type="button"
              className="primary-action"
              onClick={() => onInterpret(inputSource)}
              disabled={!input.trim()}
            >
              Create my session
              <ArrowRight size={20} aria-hidden="true" />
            </button>
          </div>
          <div className="local-processing-note">
            <LockKeyhole size={15} aria-hidden="true" />
            PDFs and photos are read locally, then discarded after
            interpretation.
          </div>
        </div>
      </section>

      <section className="start-options" aria-label="Other ways to start">
        <a className="start-option" href="/research-capture">
          <span className="option-icon research">
            <FlaskConical size={21} aria-hidden="true" />
          </span>
          <span>
            <strong>Build the movement dataset</strong>
            <small>Paired pose + optional RGB · local export</small>
          </span>
          <ChevronRight size={21} aria-hidden="true" />
        </a>
        <button type="button" className="start-option" onClick={onDemo}>
          <span className="option-icon demo">
            <Play size={21} fill="currentColor" aria-hidden="true" />
          </span>
          <span>
            <strong>Try the guided demo</strong>
            <small>All three exercises · about 2 minutes</small>
          </span>
          <ChevronRight size={21} aria-hidden="true" />
        </button>
        <button type="button" className="start-option" onClick={onManual}>
          <span className="option-icon manual">
            <SlidersHorizontal size={21} aria-hidden="true" />
          </span>
          <span>
            <strong>Build it manually</strong>
            <small>Choose the details yourself</small>
          </span>
          <ChevronRight size={21} aria-hidden="true" />
        </button>
        <button
          type="button"
          className="start-option"
          onClick={() => {
            setInput(REQUIRED_SCENARIO_TEXT);
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        >
          <span className="option-icon proof">
            <ShieldCheck size={21} aria-hidden="true" />
          </span>
          <span>
            <strong>Load the 90° limit proof</strong>
            <small>Target and restriction intelligence</small>
          </span>
          <ChevronRight size={21} aria-hidden="true" />
        </button>
      </section>

      <section className="home-lower">
        <div>
          <div className="eyebrow subtle">Designed around uncertainty</div>
          <h2>Relay knows when it can’t see enough.</h2>
          <p>
            Relay pauses any body feature it cannot estimate reliably—whether
            that is a joint angle, leg alignment, lift, or posture
            relationship. It never fills a tracking gap with a guess.
          </p>
        </div>
        <div className="confidence-demo" aria-label="Tracking states">
          <div className="confidence-card good">
            <CheckCircle2 size={22} aria-hidden="true" />
            <span>
              <strong>Good</strong>
              Body features active
            </span>
          </div>
          <div className="confidence-card limited">
            <CircleAlert size={22} aria-hidden="true" />
            <span>
              <strong>Reposition</strong>
              Some checks paused
            </span>
          </div>
          <div className="confidence-card paused">
            <Pause size={22} aria-hidden="true" />
            <span>
              <strong>Not measuring</strong>
              Assessment paused
            </span>
          </div>
        </div>
      </section>

      {history.length > 0 && (
        <section className="recent-section">
          <div className="section-heading">
            <div>
              <div className="eyebrow subtle">Stored on this device</div>
              <h2>Recent sessions</h2>
            </div>
            <button type="button" className="text-button" onClick={onOpenHistory}>
              View all <ArrowRight size={17} aria-hidden="true" />
            </button>
          </div>
          <div className="recent-grid">
            {history.slice(0, 3).map((run) => (
              <article key={run.id} className="recent-card">
                <div className="recent-icon">
                  <Activity size={21} aria-hidden="true" />
                </div>
                <div>
                  <strong>{run.title}</strong>
                  <span>
                    {new Date(run.completedAt).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                    })}
                    {" · "}
                    {run.exercises.reduce(
                      (sum, exercise) =>
                        sum +
                        exercise.sets.reduce(
                          (repSum, set) => repSum + set.reps.length,
                          0,
                        ),
                      0,
                    )}{" "}
                    reps
                  </span>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
      <footer className="home-footer">
        <span>Relay MVP · movement guidance, not medical advice</span>
        <button type="button" onClick={onPrivacy}>
          Product boundary & privacy
        </button>
      </footer>
    </main>
  );
}

function FlowShell({
  children,
  step,
  onBack,
  onHome,
}: {
  children: React.ReactNode;
  step: Parameters<typeof ProgressRail>[0]["step"];
  onBack: () => void;
  onHome: () => void;
}) {
  return (
    <main className="flow-screen">
      <header className="flow-header">
        <IconButton label="Go back" onClick={onBack}>
          <ArrowLeft size={22} />
        </IconButton>
        <button className="brand-button" type="button" onClick={onHome}>
          <Brand />
        </button>
        <div className="flow-spacer" />
      </header>
      <ProgressRail step={step} />
      {children}
    </main>
  );
}

function ClarifyScreen({
  session,
  onResolve,
  onBack,
  onHome,
}: {
  session: ConfirmedSession;
  onResolve: (clarification: Clarification, optionId: string) => void;
  onBack: () => void;
  onHome: () => void;
}) {
  const clarification = session.clarifications[0];
  if (!clarification) return null;
  return (
    <FlowShell step="check" onBack={onBack} onHome={onHome}>
      <section className="clarify-screen">
        <div className="flow-eyebrow">
          <Sparkles size={17} aria-hidden="true" />
          One detail changes the guidance
        </div>
        <h1>{clarification.question}</h1>
        <p>{clarification.detail}</p>
        <div className="source-quote">
          <span>From your instructions</span>
          <blockquote>“{session.sourceText}”</blockquote>
        </div>
        <div className="clarify-options">
          {clarification.options.map((option) => (
            <button
              type="button"
              key={option.id}
              onClick={() => onResolve(clarification, option.id)}
            >
              <span className="option-radio" aria-hidden="true" />
              <span>
                <strong>{option.label}</strong>
                <small>{option.description}</small>
              </span>
              <ChevronRight size={22} aria-hidden="true" />
            </button>
          ))}
        </div>
        {clarification.reason === "angle-meaning" && (
          <div className="trust-note">
            <ShieldCheck size={21} aria-hidden="true" />
            <span>
              Relay never silently turns a goal into a safety restriction—or a
              restriction into a goal.
            </span>
          </div>
        )}
      </section>
    </FlowShell>
  );
}

function RuleEditor({
  rule,
  onChange,
  onDelete,
}: {
  rule: MovementRule;
  onChange: (rule: MovementRule) => void;
  onDelete: () => void;
}) {
  const typeLabel =
    rule.kind === "maximum"
      ? "Maximum"
      : rule.kind === "target"
        ? "Target"
        : rule.kind === "minimum"
          ? "Minimum"
          : rule.kind === "range"
            ? "Range"
            : "Visible cue";
  return (
    <div className={`rule-editor rule-${rule.kind}`}>
      <span className="rule-type">{typeLabel}</span>
      <label>
        <span>{rule.metric.replaceAll("-", " ")}</span>
        {rule.kind === "range" ? (
          <span className="range-inputs">
            <input
              type="number"
              value={rule.min ?? ""}
              aria-label="Minimum"
              onChange={(event) =>
                onChange({ ...rule, min: Number(event.target.value) })
              }
            />
            <span>to</span>
            <input
              type="number"
              value={rule.max ?? ""}
              aria-label="Maximum"
              onChange={(event) =>
                onChange({ ...rule, max: Number(event.target.value) })
              }
            />
          </span>
        ) : (
          <span className="number-suffix">
            <input
              type="number"
              value={rule.value ?? ""}
              onChange={(event) =>
                onChange({ ...rule, value: Number(event.target.value) })
              }
              aria-label={`${typeLabel} value`}
            />
            <span>{rule.unit === "degrees" ? "°" : rule.unit}</span>
          </span>
        )}
      </label>
      <input
        className="rule-label-input"
        value={rule.label}
        aria-label="Rule wording"
        onChange={(event) => onChange({ ...rule, label: event.target.value })}
      />
      <IconButton label="Remove rule" onClick={onDelete}>
        <X size={17} />
      </IconButton>
      {rule.kind === "maximum" && rule.hard && (
        <span className="maximum-explainer">
          A limit, not a goal. Relay will never encourage movement toward it.
        </span>
      )}
      <span className="scope-explainer">
        {rule.scope.kind.replace("-", " ")} · {rule.scope.description}
      </span>
    </div>
  );
}

function ExerciseEditor({
  exercise,
  index,
  onChange,
}: {
  exercise: ExercisePrescription;
  index: number;
  onChange: (exercise: ExercisePrescription) => void;
}) {
  const [expanded, setExpanded] = useState(index === 0);
  const updateDose = (
    key: "sets" | "reps" | "holdSeconds",
    value: number,
  ) =>
    onChange({
      ...exercise,
      dose: { ...exercise.dose, [key]: Math.max(key === "holdSeconds" ? 0 : 1, value) },
    });
  const addRule = (kind: "target" | "maximum") => {
    const metric =
      exercise.exerciseId === "seated-knee-extension"
        ? "extension-progress"
        : exercise.exerciseId === "straight-leg-raise"
          ? "leg-lift"
          : "knee-flexion";
    const unit = metric === "extension-progress" ? "percent" : "degrees";
    const value = metric === "extension-progress" ? 80 : 90;
    const rule: MovementRule = {
      id: createId("rule"),
      scope: constraintScopeForMetric(metric),
      metric,
      kind,
      unit,
      value,
      hard: kind === "maximum",
      label:
        kind === "maximum"
          ? `Do not go beyond ${value}${unit === "degrees" ? "°" : "%"}`
          : `Aim for about ${value}${unit === "degrees" ? "°" : "%"}`,
    };
    onChange({ ...exercise, rules: [...exercise.rules, rule] });
  };

  return (
    <article className="exercise-editor">
      <button
        type="button"
        className="exercise-editor-heading"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        <span className="exercise-number">{index + 1}</span>
        <span className="exercise-title">
          <strong>{exercise.name}</strong>
          <small>
            {exercise.side} · {exercise.dose.sets} × {exercise.dose.reps}
            {exercise.dose.holdSeconds
              ? ` · ${exercise.dose.holdSeconds}s hold`
              : ""}
          </small>
          {!expanded && exercise.rules.length > 0 && (
            <span className="collapsed-rule-summary">
              {exercise.rules.slice(0, 2).map((rule) => (
                <span key={rule.id}>{liveRuleLabel(rule)}</span>
              ))}
              {exercise.rules.length > 2 && (
                <span>+{exercise.rules.length - 2} more</span>
              )}
            </span>
          )}
        </span>
        <ChevronRight
          size={22}
          className={expanded ? "rotated" : ""}
          aria-hidden="true"
        />
      </button>
      {expanded && (
        <div className="exercise-editor-body">
          <div className="form-row exercise-basics">
            <label>
              <span>Exercise</span>
              <select
                value={exercise.exerciseId}
                onChange={(event) => {
                  const exerciseId =
                    event.target.value as ExercisePrescription["exerciseId"];
                  onChange({
                    ...exercise,
                    exerciseId,
                    name: EXERCISE_LABELS[exerciseId],
                  });
                }}
              >
                <option value="heel-slide">Heel slide</option>
                <option value="straight-leg-raise">Straight-leg raise</option>
                <option value="seated-knee-extension">
                  Seated knee extension
                </option>
              </select>
            </label>
            <label>
              <span>Working side</span>
              <select
                value={exercise.side}
                onChange={(event) =>
                  onChange({
                    ...exercise,
                    side: event.target.value as "left" | "right",
                  })
                }
              >
                <option value="right">Right leg</option>
                <option value="left">Left leg</option>
              </select>
            </label>
          </div>
          <div className="dose-grid">
            <label>
              <span>Sets</span>
              <input
                type="number"
                min="1"
                max="20"
                value={exercise.dose.sets}
                onChange={(event) =>
                  updateDose("sets", Number(event.target.value))
                }
              />
            </label>
            <span className="dose-times">×</span>
            <label>
              <span>Reps</span>
              <input
                type="number"
                min="1"
                max="100"
                value={exercise.dose.reps}
                onChange={(event) =>
                  updateDose("reps", Number(event.target.value))
                }
              />
            </label>
            <label>
              <span>Hold</span>
              <span className="number-suffix">
                <input
                  type="number"
                  min="0"
                  max="120"
                  value={exercise.dose.holdSeconds}
                  onChange={(event) =>
                    updateDose("holdSeconds", Number(event.target.value))
                  }
                />
                <span>sec</span>
              </span>
            </label>
            <label>
              <span>Tempo</span>
              <select
                value={exercise.dose.tempo.cue}
                onChange={(event) =>
                  onChange({
                    ...exercise,
                    dose: {
                      ...exercise.dose,
                      tempo: {
                        ...exercise.dose.tempo,
                        cue: event.target.value as
                          | "comfortable"
                          | "slow"
                          | "controlled",
                      },
                    },
                  })
                }
              >
                <option value="comfortable">Comfortable</option>
                <option value="controlled">Controlled</option>
                <option value="slow">Slow</option>
              </select>
            </label>
          </div>
          <div className="rules-section">
            <div className="rules-heading">
              <div>
                <strong>Targets & restrictions</strong>
                <span>They are evaluated differently.</span>
              </div>
              <div>
                <button type="button" onClick={() => addRule("target")}>
                  + Target
                </button>
                <button type="button" onClick={() => addRule("maximum")}>
                  + Maximum
                </button>
              </div>
            </div>
            {exercise.rules.length === 0 ? (
              <div className="empty-rules">
                No numeric target or restriction. Relay will still count
                movement, holds, and tempo.
              </div>
            ) : (
              exercise.rules.map((rule) => (
                <RuleEditor
                  key={rule.id}
                  rule={rule}
                  onChange={(next) =>
                    onChange({
                      ...exercise,
                      rules: exercise.rules.map((item) =>
                        item.id === rule.id ? next : item,
                      ),
                    })
                  }
                  onDelete={() =>
                    onChange({
                      ...exercise,
                      rules: exercise.rules.filter(
                        (item) => item.id !== rule.id,
                      ),
                    })
                  }
                />
              ))
            )}
          </div>
        </div>
      )}
    </article>
  );
}

function ReviewScreen({
  session,
  onChange,
  onConfirm,
  onBack,
  onHome,
  error,
}: {
  session: ConfirmedSession;
  onChange: (session: ConfirmedSession) => void;
  onConfirm: () => void;
  onBack: () => void;
  onHome: () => void;
  error: string;
}) {
  return (
    <FlowShell step="check" onBack={onBack} onHome={onHome}>
      <section className="review-screen">
        <div className="review-intro">
          <div>
            <div className="flow-eyebrow">
              <CheckCircle2 size={17} aria-hidden="true" />
              Relay found {session.exercises.length} exercise
              {session.exercises.length === 1 ? "" : "s"}
            </div>
            <h1>Check what Relay understood.</h1>
            <p>
              Every value remains yours to correct. A target is encouragement;
              a maximum is a hard instruction.
            </p>
          </div>
          <div className="interpretation-source">
            <span>{session.source}</span>
            <p>{session.sourceText}</p>
          </div>
        </div>
        <div className="exercise-list">
          {session.exercises.map((exercise, index) => (
            <ExerciseEditor
              key={exercise.id}
              exercise={exercise}
              index={index}
              onChange={(next) =>
                onChange({
                  ...session,
                  exercises: session.exercises.map((item) =>
                    item.id === exercise.id ? next : item,
                  ),
                  updatedAt: new Date().toISOString(),
                })
              }
            />
          ))}
        </div>
        {error && (
          <div className="form-error" role="alert">
            <CircleAlert size={20} aria-hidden="true" />
            {error}
          </div>
        )}
        <div className="boundary-card">
          <ShieldCheck size={24} aria-hidden="true" />
          <div>
            <strong>You stay in control.</strong>
            <p>
              Relay follows the session you confirm. It does not decide whether
              an exercise is medically appropriate. Camera-derived movement
              values are estimates, not clinical measurements.
            </p>
          </div>
        </div>
        <div className="flow-actions sticky-actions">
          <button type="button" className="secondary-action" onClick={onBack}>
            Edit instructions
          </button>
          <button type="button" className="primary-action" onClick={onConfirm}>
            Confirm & set up camera
            <ArrowRight size={20} aria-hidden="true" />
          </button>
        </div>
      </section>
    </FlowShell>
  );
}

type PoseConnection = readonly [JointName, JointName];

const POSE_CONNECTIONS: PoseConnection[] = [
  ["nose", "left-eye-inner"],
  ["left-eye-inner", "left-eye"],
  ["left-eye", "left-eye-outer"],
  ["left-eye-outer", "left-ear"],
  ["nose", "right-eye-inner"],
  ["right-eye-inner", "right-eye"],
  ["right-eye", "right-eye-outer"],
  ["right-eye-outer", "right-ear"],
  ["mouth-left", "mouth-right"],
  ["left-ear", "left-shoulder"],
  ["right-ear", "right-shoulder"],
  ["left-shoulder", "right-shoulder"],
  ["left-shoulder", "left-elbow"],
  ["left-elbow", "left-wrist"],
  ["left-wrist", "left-pinky"],
  ["left-wrist", "left-index"],
  ["left-wrist", "left-thumb"],
  ["left-pinky", "left-index"],
  ["right-shoulder", "right-elbow"],
  ["right-elbow", "right-wrist"],
  ["right-wrist", "right-pinky"],
  ["right-wrist", "right-index"],
  ["right-wrist", "right-thumb"],
  ["right-pinky", "right-index"],
  ["left-shoulder", "left-hip"],
  ["right-shoulder", "right-hip"],
  ["left-hip", "right-hip"],
  ["left-hip", "left-knee"],
  ["left-knee", "left-ankle"],
  ["left-ankle", "left-heel"],
  ["left-ankle", "left-foot-index"],
  ["left-heel", "left-foot-index"],
  ["right-hip", "right-knee"],
  ["right-knee", "right-ankle"],
  ["right-ankle", "right-heel"],
  ["right-ankle", "right-foot-index"],
  ["right-heel", "right-foot-index"],
];

const PROMINENT_JOINTS = new Set<JointName>([
  "nose",
  "left-shoulder",
  "right-shoulder",
  "left-elbow",
  "right-elbow",
  "left-wrist",
  "right-wrist",
  "left-hip",
  "right-hip",
  "left-knee",
  "right-knee",
  "left-ankle",
  "right-ankle",
  "left-heel",
  "right-heel",
  "left-foot-index",
  "right-foot-index",
]);

function exerciseConnections(
  exerciseId: ExerciseId,
  side: "left" | "right",
): PoseConnection[] {
  const shoulder = `${side}-shoulder` as JointName;
  const hip = `${side}-hip` as JointName;
  const knee = `${side}-knee` as JointName;
  const ankle = `${side}-ankle` as JointName;
  const heel = `${side}-heel` as JointName;
  const foot = `${side}-foot-index` as JointName;
  if (exerciseId === "straight-leg-raise") {
    return [
      [shoulder, hip],
      [hip, knee],
      [knee, ankle],
      [hip, ankle],
    ];
  }
  if (exerciseId === "seated-knee-extension") {
    return [
      [hip, knee],
      [knee, ankle],
      [ankle, foot],
    ];
  }
  return [
    [hip, knee],
    [knee, ankle],
    [ankle, heel],
  ];
}

function drawPose(
  canvas: HTMLCanvasElement | null,
  frame: PoseFrame,
  side: "left" | "right",
  exerciseId: ExerciseId,
  fit: "cover" | "contain" = "cover",
) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(rect.width * ratio));
  const pixelHeight = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  // Landmarks are normalised against the source frame, so they must be mapped
  // through the same fit the viewer sees. With a live <video> behind the canvas
  // that is object-fit: cover, and the skeleton has to be cropped identically or
  // it drifts off the body. With simulated landmarks there is no video to match,
  // and cover would push most of a landscape body outside a narrower panel, so
  // contain is used to keep the whole body visible.
  const sourceWidth = Math.max(1, frame.frameWidth);
  const sourceHeight = Math.max(1, frame.frameHeight);
  const fitScale =
    fit === "cover"
      ? Math.max(rect.width / sourceWidth, rect.height / sourceHeight)
      : Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
  const offsetX = (rect.width - sourceWidth * fitScale) / 2;
  const offsetY = (rect.height - sourceHeight * fitScale) / 2;
  const toPoint = (joint: JointName) => {
    const point = frame.image[joint];
    return point
      ? {
          x: point.x * sourceWidth * fitScale + offsetX,
          y: point.y * sourceHeight * fitScale + offsetY,
          good: Math.min(point.visibility, point.presence) >= 0.65,
        }
      : undefined;
  };

  const highlightedConnections = exerciseConnections(exerciseId, side);
  const highlightedJoints = new Set<JointName>(
    highlightedConnections.flatMap(([start, end]) => [start, end]),
  );

  context.lineCap = "round";
  POSE_CONNECTIONS.forEach(([start, end]) => {
    const a = toPoint(start);
    const b = toPoint(end);
    if (!a || !b) return;
    context.strokeStyle =
      a.good && b.good
        ? "rgba(214, 241, 231, .46)"
        : "rgba(255, 191, 105, .48)";
    context.lineWidth = 2.25;
    context.setLineDash([]);
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  });

  highlightedConnections.forEach(([start, end], index) => {
    const a = toPoint(start);
    const b = toPoint(end);
    if (!a || !b) return;
    context.strokeStyle =
      a.good && b.good
        ? "rgba(148, 255, 198, .96)"
        : "rgba(255, 191, 105, .9)";
    context.lineWidth =
      index === highlightedConnections.length - 1 &&
      exerciseId === "straight-leg-raise"
        ? 2.5
        : 5;
    context.setLineDash(
      exerciseId === "straight-leg-raise" &&
        index === highlightedConnections.length - 1
        ? [8, 7]
        : [],
    );
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  });
  context.setLineDash([]);

  (Object.keys(frame.image) as JointName[]).forEach((joint) => {
    const point = toPoint(joint);
    if (!point) return;
    const highlighted = highlightedJoints.has(joint);
    const prominent = PROMINENT_JOINTS.has(joint);
    const radius = highlighted ? 7.5 : prominent ? 4.25 : 2.25;
    context.fillStyle = highlighted
      ? point.good
        ? "#f5fff8"
        : "#ffd9a8"
      : point.good
        ? "rgba(223, 245, 236, .72)"
        : "rgba(255, 217, 168, .68)";
    context.strokeStyle = highlighted
      ? "#163b34"
      : "rgba(16, 46, 40, .72)";
    context.lineWidth = highlighted ? 2.5 : 1.25;
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });
}

function CameraSetup({
  session,
  mode,
  onMode,
  cameraFacing,
  onCameraFacing,
  scenario,
  onScenario,
  onStart,
  onBack,
  onHome,
}: {
  session: ConfirmedSession;
  mode: SessionMode;
  onMode: (mode: SessionMode) => void;
  cameraFacing: CameraFacing;
  onCameraFacing: (facing: CameraFacing) => void;
  scenario: ReplayScenarioId;
  onScenario: (scenario: ReplayScenarioId) => void;
  onStart: () => void;
  onBack: () => void;
  onHome: () => void;
}) {
  const exercise = session.exercises[0];
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const providerRef = useRef<MediaPipeMotionProvider | null>(null);
  const stabilitySamplesRef = useRef<
    Array<{ timestampMs: number; value: number }>
  >([]);
  const [status, setStatus] = useState<MotionProviderStatus>({
    state: "idle",
    label: "Camera check not started",
  });
  const [measurements, setMeasurements] =
    useState<MotionMeasurements | null>(null);
  const [stableMs, setStableMs] = useState(0);
  const [placementConfirmed, setPlacementConfirmed] = useState(false);
  const [sideConfirmed, setSideConfirmed] = useState(false);
  const [checking, setChecking] = useState(false);
  const { muted, setMuted, speak } = useSpokenCoach();

  const runCheck = async () => {
    setChecking(true);
    setMeasurements(null);
    setStableMs(0);
    stabilitySamplesRef.current = [];
    if (mode === "replay") {
      const frames = createReplayFrames({
        exerciseId: exercise.exerciseId,
        side: exercise.side,
        scenario,
        reps: 1,
        holdSeconds: exercise.dose.holdSeconds,
      }).slice(0, 18);
      const measured = frames.map((frame) =>
        measurePose(frame, exercise.exerciseId, exercise.side),
      );
      const frame = frames[frames.length - 1];
      const next = measured[measured.length - 1];
      setMeasurements(next);
      setStableMs(
        measured.every(
          (measurement) =>
            evaluateSetupFeatureReadiness(exercise, measurement).status ===
            "good",
        )
          ? frame.timestampMs - frames[0].timestampMs
          : 0,
      );
      drawPose(
        canvasRef.current,
        frame,
        exercise.side,
        exercise.exerciseId,
        "contain",
      );
      setStatus({
        state: "ready",
        label: "Replay landmarks ready",
        detail: "Simulated data will pass through Relay’s production pipeline.",
      });
      setChecking(false);
      return;
    }
    try {
      providerRef.current?.dispose();
      const provider = new MediaPipeMotionProvider(cameraFacing);
      providerRef.current = provider;
      await provider.start(
        videoRef.current,
        (frame) => {
          const next = measurePose(frame, exercise.exerciseId, exercise.side);
          setMeasurements(next);
          const setupReadiness = evaluateSetupFeatureReadiness(exercise, next);
          const primary =
            exercise.exerciseId === "straight-leg-raise"
              ? next.legLift
              : exercise.exerciseId === "seated-knee-extension"
                ? next.extensionProgress
                : next.kneeFlexion;
          if (setupReadiness.status !== "good" || primary === undefined) {
            stabilitySamplesRef.current = [];
            setStableMs(0);
          } else {
            const samples = [
              ...stabilitySamplesRef.current,
              { timestampMs: frame.timestampMs, value: primary },
            ].filter(
              (sample) => sample.timestampMs >= frame.timestampMs - 2_000,
            );
            const values = samples.map((sample) => sample.value);
            const stable =
              values.length > 1 &&
              Math.max(...values) - Math.min(...values) <= 5;
            stabilitySamplesRef.current = stable ? samples : samples.slice(-2);
            setStableMs(
              stable
                ? frame.timestampMs -
                    (stabilitySamplesRef.current[0]?.timestampMs ??
                      frame.timestampMs)
                : 0,
            );
          }
          drawPose(
            canvasRef.current,
            frame,
            exercise.side,
            exercise.exerciseId,
            "cover",
          );
        },
        setStatus,
      );
    } catch {
      // User-facing provider status already contains the recovery path.
    } finally {
      setChecking(false);
    }
  };

  useEffect(
    () => () => {
      providerRef.current?.dispose();
    },
    [],
  );

  const setupReadiness = measurements
    ? evaluateSetupFeatureReadiness(exercise, measurements)
    : null;
  const quality = setupReadiness?.status;
  const geometryReady = quality === "good" && stableMs >= 1_500;
  const ready = geometryReady && placementConfirmed && sideConfirmed;
  const requiredJoints =
    exercise.exerciseId === "straight-leg-raise"
      ? "shoulder, hip, knee, and ankle"
      : "hip, knee, and ankle";
  return (
    <FlowShell step="position" onBack={onBack} onHome={onHome}>
      <section className="camera-screen">
        <div className="camera-heading">
          <div>
            <div className="flow-eyebrow">
              <Camera size={17} aria-hidden="true" />
              Camera setup
            </div>
            <h1>Let’s find a view Relay can trust.</h1>
            <p>
              Place the phone in landscape, about 6–8 feet away near the
              moving joints. Turn your {exercise.side} side toward the camera.
            </p>
          </div>
          <div className="mode-switch" aria-label="Movement source">
            <button
              type="button"
              className={mode === "live" ? "selected" : ""}
              onClick={() => {
                onMode("live");
                setMeasurements(null);
                setStableMs(0);
                stabilitySamplesRef.current = [];
                setStatus({ state: "idle", label: "Camera check not started" });
              }}
            >
              <Camera size={17} aria-hidden="true" /> Live camera
            </button>
            <button
              type="button"
              className={mode === "replay" ? "selected" : ""}
              onClick={() => {
                providerRef.current?.dispose();
                onMode("replay");
                setMeasurements(null);
                setStableMs(0);
                stabilitySamplesRef.current = [];
                setStatus({ state: "idle", label: "Replay check not started" });
              }}
            >
              <FlaskConical size={17} aria-hidden="true" /> Guided replay
            </button>
          </div>
        </div>
        <div className="camera-layout">
          <div
            className={`camera-preview ${mode} ${
              mode === "live" && cameraFacing === "user" ? "front" : ""
            }`}
          >
            <video
              ref={videoRef}
              className={mode === "live" ? "" : "hidden"}
              playsInline
              muted
            />
            <canvas ref={canvasRef} aria-hidden="true" />
            {mode === "replay" && !measurements && (
              <div className="replay-placeholder">
                <span className="replay-orbit">
                  <span />
                  <Activity size={40} aria-hidden="true" />
                </span>
                <strong>Repeatable movement replay</strong>
                <p>Simulated landmarks · no video</p>
              </div>
            )}
            <div className="frame-guide" aria-hidden="true">
              <span className="corner top-left" />
              <span className="corner top-right" />
              <span className="corner bottom-left" />
              <span className="corner bottom-right" />
              <span className="side-label">
                {exercise.side === "right" ? "Right side" : "Left side"}
              </span>
            </div>
            <div
              className={`preview-status status-${quality ?? "idle"}`}
              role="status"
            >
              {quality === "good" ? (
                <CheckCircle2 size={18} aria-hidden="true" />
              ) : quality ? (
                <CircleAlert size={18} aria-hidden="true" />
              ) : (
                <MoreHorizontal size={18} aria-hidden="true" />
              )}
              {quality === "good"
                ? geometryReady
                  ? "Good · body features active"
                  : "Hold still · checking stability"
                : quality === "limited"
                  ? "Reposition · some checks paused"
                  : quality === "lost"
                    ? "Not measuring · joints missing"
                    : status.label}
            </div>
          </div>
          <div className="setup-panel">
            <div className="setup-exercise">
              <span>Setting up</span>
              <strong>
                {exercise.side === "right" ? "Right" : "Left"} {exercise.name}
              </strong>
              <small>Relay needs the {requiredJoints} in view.</small>
            </div>
            {mode === "live" && (
              <label className="scenario-select">
                <span>Camera lens</span>
                <select
                  value={cameraFacing}
                  onChange={(event) => {
                    providerRef.current?.dispose();
                    onCameraFacing(event.target.value as CameraFacing);
                    setMeasurements(null);
                    setStableMs(0);
                    stabilitySamplesRef.current = [];
                    setStatus({
                      state: "idle",
                      label: "Camera check not started",
                    });
                  }}
                >
                  <option value="user">Front · see Relay’s coaching</option>
                  <option value="environment">Rear · usually sharper</option>
                </select>
              </label>
            )}
            {mode === "replay" && (
              <label className="scenario-select">
                <span>Replay scenario</span>
                <select
                  value={scenario}
                  onChange={(event) =>
                    onScenario(event.target.value as ReplayScenarioId)
                  }
                >
                  {REPLAY_SCENARIOS.filter((item) =>
                    item.id === "posture-variation"
                      ? exercise.exerciseId === "straight-leg-raise"
                      : exercise.exerciseId === "heel-slide"
                        ? true
                        : !["approach-limit", "exceed-limit"].includes(item.id),
                  ).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <ol className="setup-checklist">
              <li className={placementConfirmed ? "done" : ""}>
                <span>
                  {placementConfirmed ? (
                    <Check size={16} aria-hidden="true" />
                  ) : (
                    "1"
                  )}
                </span>
                <div>
                  <label>
                    <strong>Phone supported in landscape</strong>
                    <small>Lean it against something secure.</small>
                    <input
                      type="checkbox"
                      checked={placementConfirmed}
                      onChange={(event) =>
                        setPlacementConfirmed(event.target.checked)
                      }
                    />
                  </label>
                </div>
              </li>
              <li className={sideConfirmed ? "done" : ""}>
                <span>
                  {sideConfirmed ? (
                    <Check size={16} aria-hidden="true" />
                  ) : (
                    "2"
                  )}
                </span>
                <div>
                  <label>
                    <strong>
                      {exercise.side === "right" ? "Right" : "Left"} side faces
                      the camera
                    </strong>
                    <small>Relay never infers body side from screen position.</small>
                    <input
                      type="checkbox"
                      checked={sideConfirmed}
                      onChange={(event) =>
                        setSideConfirmed(event.target.checked)
                      }
                    />
                  </label>
                </div>
              </li>
              <li className={quality ? (quality === "lost" ? "" : "done") : ""}>
                <span>
                  {quality && quality !== "lost" ? (
                    <Check size={16} aria-hidden="true" />
                  ) : (
                    "3"
                  )}
                </span>
                <div>
                  <strong>Required joints visible</strong>
                  <small>
                    {setupReadiness?.reasons[0] ??
                      "Run the check to inspect the frame."}
                  </small>
                </div>
              </li>
              <li className={geometryReady ? "done" : ""}>
                <span>
                  {geometryReady ? (
                    <Check size={16} aria-hidden="true" />
                  ) : (
                    "4"
                  )}
                </span>
                <div>
                  <strong>Stable neutral pose and body geometry agree</strong>
                  <small>
                    Relay requires 1.5 seconds of stable visible joint
                    relationships before applying movement rules.
                  </small>
                </div>
              </li>
            </ol>
            <div className="model-note">
              <span className="model-dot" />
              <span>
                <strong>On-device movement model</strong>
                MediaPipe Pose Full + Relay Temporal · video is not saved
              </span>
            </div>
            <div className="audio-row">
              <button
                type="button"
                onClick={() => {
                  setMuted(false);
                  speak(
                    "Relay audio is ready. You will hear the same guidance shown on screen.",
                    true,
                  );
                }}
              >
                <Volume2 size={19} aria-hidden="true" /> Test audio
              </button>
              <button
                type="button"
                onClick={() => setMuted(!muted)}
                aria-pressed={muted}
              >
                {muted ? (
                  <VolumeX size={19} aria-hidden="true" />
                ) : (
                  <Volume2 size={19} aria-hidden="true" />
                )}
                {muted ? "Sound off" : "Sound on"}
              </button>
            </div>
            <button
              type="button"
              className="camera-check-button"
              onClick={runCheck}
              disabled={checking}
            >
              {checking ? (
                <>
                  <span className="spinner" /> Checking…
                </>
              ) : (
                <>
                  <Camera size={20} aria-hidden="true" />
                  {measurements ? "Check again" : "Run camera check"}
                </>
              )}
            </button>
            <button
              type="button"
              className="primary-action start-session-button"
              onClick={async () => {
                await providerRef.current?.stop();
                onStart();
              }}
              disabled={!ready}
            >
              Start session
              <ArrowRight size={20} aria-hidden="true" />
            </button>
            {!ready && (
              <p className="start-help">
                Body-feature targets and restrictions stay unavailable until
                the setup check is good. You can choose guided replay to
                explore safely.
              </p>
            )}
          </div>
        </div>
      </section>
    </FlowShell>
  );
}

function metricDisplay(output: EngineOutput | null) {
  if (!output)
    return { value: "—", label: "Waiting for movement", numeric: 0 };
  if (!output.measurements.confidence.canAssessBodyFeatures) {
    return { value: "—", label: "Estimate paused", numeric: 0 };
  }
  if (output.exerciseId === "straight-leg-raise") {
    const numeric = output.measurements.legLift ?? 0;
    return {
      value:
        output.measurements.legLift !== undefined
          ? `${Math.round(output.measurements.legLift)}°`
          : "—",
      label: "estimated lift",
      numeric,
    };
  }
  if (output.exerciseId === "seated-knee-extension") {
    const numeric = output.measurements.extensionProgress ?? 0;
    return {
      value:
        output.measurements.extensionProgress !== undefined
          ? `${Math.round(output.measurements.extensionProgress)}%`
          : "—",
      label: "estimated extension",
      numeric,
    };
  }
  const numeric = output.measurements.kneeFlexion ?? 0;
  return {
    value:
      output.measurements.kneeFlexion !== undefined
        ? `${Math.round(output.measurements.kneeFlexion)}°`
        : "—",
    label: "estimated knee flexion",
    numeric,
  };
}

function liveRuleLabel(rule: MovementRule) {
  if (rule.metric === "knee-bend") return "Knee stays straight";
  if (rule.metric === "trunk-compensation")
    return "Shoulders & hip stay steady";
  return rule.label;
}

function liveRuleState(
  rule: MovementRule,
  output: EngineOutput | null,
): {
  label: string;
  state: "active" | "paused" | "watch" | "stop";
} {
  const relatedEvents =
    output?.events.filter(
      (event) => event.ruleId === rule.id || event.metric === rule.metric,
    ) ?? [];
  if (relatedEvents.some((event) => event.type === "restriction-reached")) {
    return { label: "Stop", state: "stop" };
  }
  if (
    relatedEvents.some((event) =>
      ["restriction-approaching", "posture-observation"].includes(event.type),
    )
  ) {
    return { label: "Watch", state: "watch" };
  }
  const feature = output?.measurements.features.find(
    (item) => item.metric === rule.metric,
  );
  if (!output || !feature || feature.quality !== "good") {
    return { label: "Paused", state: "paused" };
  }
  return {
    label: rule.kind === "maximum" && rule.hard ? "Limit active" : "Measuring",
    state: "active",
  };
}

function ActiveSession({
  session,
  mode,
  cameraFacing,
  scenario,
  onComplete,
  onExit,
}: {
  session: ConfirmedSession;
  mode: SessionMode;
  cameraFacing: CameraFacing;
  scenario: ReplayScenarioId;
  onComplete: (result: SessionRunResult) => void;
  onExit: () => void;
}) {
  const [exerciseIndex, setExerciseIndex] = useState(0);
  const [setIndex, setSetIndex] = useState(0);
  const [output, setOutput] = useState<EngineOutput | null>(null);
  const [providerStatus, setProviderStatus] = useState<MotionProviderStatus>({
    state: "loading",
    label: "Preparing movement intelligence",
  });
  const [paused, setPaused] = useState(false);
  const [setComplete, setSetComplete] = useState(false);
  const [completionStats, setCompletionStats] = useState({
    repCount: 0,
    trackingPercent: 0,
  });
  const [starting, setStarting] = useState(true);
  const [temporalPrediction, setTemporalPrediction] =
    useState<TemporalPrediction | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const providerRef = useRef<MotionProvider | null>(null);
  const engineRef = useRef<ExerciseEngine | null>(null);
  const eventsRef = useRef<MotionEvent[]>([]);
  const resultsRef = useRef<ExerciseRunResult[]>([]);
  const startedAtRef = useRef(new Date().toISOString());
  const completionLatchedRef = useRef(false);
  const temporalModelRef = useRef<TemporalMovementModel | null>(null);
  const temporalPredictionRef = useRef<TemporalPrediction | null>(null);
  const nextTemporalSampleAtRef = useRef<number | null>(null);
  const frameHandlerRef = useRef<(frame: PoseFrame) => void>(() => undefined);
  const { muted, setMuted, speak } = useSpokenCoach();

  const exercise = session.exercises[exerciseIndex];
  const metric = metricDisplay(output);
  const displayedMetric =
    exercise.exerciseId === "straight-leg-raise"
      ? "leg-lift"
      : exercise.exerciseId === "seated-knee-extension"
        ? "extension-progress"
        : "knee-flexion";
  const hardLimits = exercise.rules.filter(
    (rule) => rule.kind === "maximum" && rule.hard,
  );
  const maximum = exercise.rules.find(
    (rule) =>
      rule.kind === "maximum" &&
      rule.hard &&
      rule.metric === displayedMetric,
  );
  const target = exercise.rules.find(
    (rule) =>
      (rule.kind === "target" || rule.kind === "minimum") &&
      rule.metric === displayedMetric,
  );

  const startProvider = useCallback(async () => {
    setStarting(true);
    completionLatchedRef.current = false;
    eventsRef.current = [];
    await providerRef.current?.stop();
    const engine = new ExerciseEngine(exercise);
    engineRef.current = engine;
    try {
      temporalModelRef.current ??= await loadTemporalMovementModel();
      temporalModelRef.current.reset();
      temporalPredictionRef.current = null;
      setTemporalPrediction(null);
      nextTemporalSampleAtRef.current = null;
    } catch {
      temporalModelRef.current = null;
      temporalPredictionRef.current = null;
      nextTemporalSampleAtRef.current = null;
    }
    let provider = providerRef.current;
    if (mode === "replay") {
      provider?.dispose?.();
      provider = new ReplayMotionProvider(
        createReplayFrames({
          exerciseId: exercise.exerciseId,
          side: exercise.side,
          scenario,
          reps: exercise.dose.reps,
          holdSeconds: exercise.dose.holdSeconds,
          intervalMs: 100,
        }),
        5,
      );
    } else if (provider?.id !== "mediapipe") {
      provider?.dispose?.();
      provider = new MediaPipeMotionProvider(cameraFacing);
    }
    if (!provider) throw new Error("Movement provider did not initialize.");
    const activeProvider = provider;
    providerRef.current = activeProvider;
    frameHandlerRef.current = (frame) => {
      if (temporalModelRef.current) {
        const sampling = advanceTemporalSampleSchedule(
          frame.timestampMs,
          nextTemporalSampleAtRef.current,
          temporalModelRef.current.artifact.input.sampleRateHz,
        );
        nextTemporalSampleAtRef.current = sampling.nextSampleAtMs;
        if (sampling.shouldSample) {
          const prediction = temporalModelRef.current.observe(
            frame,
            exercise.side,
            exercise.exerciseId,
          );
          temporalPredictionRef.current = prediction;
          setTemporalPrediction(prediction);
        }
      }
      const prediction = temporalPredictionRef.current;
      const next = engine.process(
        frame,
        prediction
          ? {
              phase: prediction.phase,
              confidence: prediction.confidence,
              margin: prediction.margin,
              applicable: prediction.shouldApplyPhase,
              modelId: prediction.modelId,
            }
          : undefined,
      );
      drawPose(
        canvasRef.current,
        frame,
        exercise.side,
        exercise.exerciseId,
        mode === "live" ? "cover" : "contain",
      );
      setOutput(next);
      if (next.events.length) {
        eventsRef.current.push(...next.events);
        const urgent = next.events.some((item) =>
          [
            "restriction-reached",
            "restriction-approaching",
            "tracking-lost",
          ].includes(item.type),
        );
        speak(next.coachMessage, urgent);
      }
      if (
        next.repsCompleted >= exercise.dose.reps &&
        !completionLatchedRef.current
      ) {
        completionLatchedRef.current = true;
        const currentRecords = next.repRecords;
        const exerciseFinished = setIndex + 1 >= exercise.dose.sets;
        const sessionFinished =
          exerciseFinished &&
          exerciseIndex + 1 >= session.exercises.length;
        const committedExerciseRecords =
          resultsRef.current
            .find((item) => item.exerciseId === exercise.id)
            ?.sets.flatMap((set) => set.reps) ?? [];
        const committedSessionRecords = resultsRef.current.flatMap((item) =>
          item.sets.flatMap((set) => set.reps),
        );
        const completedRecords = sessionFinished
          ? [...committedSessionRecords, ...currentRecords]
          : exerciseFinished
            ? [...committedExerciseRecords, ...currentRecords]
            : currentRecords;
        setCompletionStats({
          repCount: completedRecords.length,
          trackingPercent: Math.round(
            (completedRecords.reduce(
              (sum, rep) => sum + rep.trackingGoodRatio,
              0,
            ) /
              Math.max(1, completedRecords.length)) *
              100,
          ),
        });
        setSetComplete(true);
        void activeProvider.stop();
        speak(
          setIndex + 1 >= exercise.dose.sets &&
            exerciseIndex + 1 >= session.exercises.length
            ? "Session complete. Your summary is ready."
            : `Set ${setIndex + 1} complete.`,
          true,
        );
      }
    };
    try {
      await activeProvider.initialize();
      await activeProvider.start(
        mode === "live" ? videoRef.current : null,
        frameHandlerRef.current,
        setProviderStatus,
      );
      setStarting(false);
      speak(
        `${exercise.side} ${exercise.name}. ${
          hardLimits.length > 0
            ? `${hardLimits.map((rule) => rule.label).join(". ")}. ${
                hardLimits.length === 1
                  ? "That is a maximum, not a goal."
                  : "Those are maximums, not goals."
              }`
            : "Begin when you are ready."
        }`,
        true,
      );
    } catch {
      setStarting(false);
    }
  }, [
    exercise,
    exerciseIndex,
    cameraFacing,
    hardLimits,
    mode,
    scenario,
    session.exercises.length,
    setIndex,
    speak,
  ]);

  useEffect(() => {
    // This effect owns the external camera/replay provider lifecycle.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void startProvider();
    return () => {
      void providerRef.current?.stop();
    };
    // startProvider is intentionally tied to the current set/exercise.
  }, [exerciseIndex, setIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(
    () => () => {
      providerRef.current?.dispose?.();
    },
    [],
  );

  const commitCurrentSet = () => {
    const nextSet = {
      set: setIndex + 1,
      plannedReps: exercise.dose.reps,
      reps: output?.repRecords ?? [],
      events: [...eventsRef.current],
    };
    const existing = resultsRef.current.find(
      (item) => item.exerciseId === exercise.id,
    );
    if (existing) {
      existing.sets = [
        ...existing.sets.filter((item) => item.set !== nextSet.set),
        nextSet,
      ];
    } else {
      resultsRef.current.push({
        exerciseId: exercise.id,
        kind: exercise.exerciseId,
        name: exercise.name,
        side: exercise.side,
        plannedSets: exercise.dose.sets,
        sets: [nextSet],
      });
    }
  };

  const continueAfterSet = () => {
    commitCurrentSet();
    setSetComplete(false);
    setOutput(null);
    if (setIndex + 1 < exercise.dose.sets) {
      setSetIndex((value) => value + 1);
      return;
    }
    if (exerciseIndex + 1 < session.exercises.length) {
      setSetIndex(0);
      setExerciseIndex((value) => value + 1);
      return;
    }
    const result = createRunResult(
      session,
      startedAtRef.current,
      mode,
      resultsRef.current,
      true,
    );
    onComplete(result);
  };

  const stopSession = () => {
    void providerRef.current?.stop();
    commitCurrentSet();
    const result = createRunResult(
      session,
      startedAtRef.current,
      mode,
      resultsRef.current,
      false,
    );
    onComplete(result);
  };

  const togglePause = async () => {
    if (paused) {
      setPaused(false);
      await providerRef.current?.start(
        mode === "live" ? videoRef.current : null,
        frameHandlerRef.current,
        setProviderStatus,
      );
      speak("Session resumed. Continue when ready.", true);
    } else {
      setPaused(true);
      engineRef.current?.pause();
      await providerRef.current?.stop();
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    }
  };

  const tracking = output?.measurements.confidence.status ?? "lost";
  const restrictionReached = output?.events.some(
    (item) => item.type === "restriction-reached",
  );
  const restrictionApproaching = output?.events.some(
    (item) => item.type === "restriction-approaching",
  );
  const progress =
    ((exerciseIndex +
      (setIndex + (output?.repsCompleted ?? 0) / exercise.dose.reps) /
        exercise.dose.sets) /
      session.exercises.length) *
    100;
  return (
    <main className="active-screen">
      <header className="active-header">
        <Brand inverse />
        <div className="active-progress">
          <span style={{ width: `${Math.min(100, progress)}%` }} />
        </div>
        <button type="button" onClick={onExit}>
          Exit
        </button>
      </header>
      <div
        className={`active-camera ${
          mode === "live" && cameraFacing === "user" ? "front" : ""
        }`}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          className={mode === "live" ? "" : "hidden"}
        />
        <canvas ref={canvasRef} aria-hidden="true" />
        {mode === "replay" && (
          <div className="replay-watermark">
            <FlaskConical size={17} aria-hidden="true" />
            Guided replay
            <span className="replay-detail">· simulated landmarks</span>
          </div>
        )}
        <div className="camera-grid" aria-hidden="true" />
        <div className={`tracking-chip tracking-${tracking}`}>
          {tracking === "good" ? (
            <CheckCircle2 size={18} aria-hidden="true" />
          ) : (
            <CircleAlert size={18} aria-hidden="true" />
          )}
          {tracking === "good"
            ? "Good · measuring"
            : tracking === "limited"
              ? "Reposition · some checks paused"
              : "Not measuring"}
        </div>
        <div
          className={`temporal-chip temporal-${
            temporalPrediction?.state ?? "unavailable"
          }`}
        >
          <Activity size={17} aria-hidden="true" />
          {temporalPrediction?.shouldApplyPhase
            ? `Movement phase · ${temporalPrediction.phase}`
            : temporalPrediction?.state === "caution"
              ? "Movement phase · checking"
              : "Movement history · warming up"}
        </div>
      </div>
      <section className="active-content">
        <div className="active-title-row">
          <div>
            <span>
              Exercise {exerciseIndex + 1} of {session.exercises.length}
            </span>
            <h1>
              {exercise.side === "right" ? "Right" : "Left"} {exercise.name}
            </h1>
          </div>
          <div className="set-count">
            <span>Set</span>
            <strong>
              {setIndex + 1}
              <small>/{exercise.dose.sets}</small>
            </strong>
          </div>
        </div>
        <div className="active-metrics">
          <div className="rep-counter">
            <span>Rep</span>
            <strong>{output?.repsCompleted ?? 0}</strong>
            <small>of {exercise.dose.reps}</small>
          </div>
          <div className={`angle-dial angle-${tracking}`}>
            <div
              className="angle-ring"
              style={
                {
                  "--angle-progress": `${Math.min(
                    100,
                    maximum?.value
                      ? (metric.numeric / maximum.value) * 100
                      : exercise.exerciseId === "seated-knee-extension"
                        ? metric.numeric
                        : (metric.numeric / 90) * 100,
                  )}%`,
                } as React.CSSProperties
              }
            >
              <div>
                <strong>{metric.value}</strong>
                <span>{metric.label}</span>
              </div>
            </div>
            {maximum?.value !== undefined && (
              <div className="limit-legend">
                <span />
                <strong>
                  Maximum {maximum.value}
                  {maximum.unit === "degrees"
                    ? "°"
                    : maximum.unit === "percent"
                      ? "%"
                      : ` ${maximum.unit}`}
                </strong>
              </div>
            )}
            {target?.value !== undefined && (
              <div className="target-legend">
                <span />
                <strong>
                  {target.kind === "minimum" ? "Minimum" : "Target"}{" "}
                  {target.value}
                  {target.unit === "degrees" ? "°" : "%"}
                </strong>
              </div>
            )}
          </div>
          <div className="hold-counter">
            <Clock3 size={23} aria-hidden="true" />
            <span>Hold</span>
            <strong>
              {(output?.holdElapsedMs ?? 0) > 0
                ? `${Math.min(
                    exercise.dose.holdSeconds,
                    (output?.holdElapsedMs ?? 0) / 1000,
                  ).toFixed(1)}s`
                : "—"}
            </strong>
            <small>
              {exercise.dose.holdSeconds
                ? `goal ${exercise.dose.holdSeconds}s`
                : "no timed hold"}
            </small>
          </div>
        </div>
        <div
          className={`coach-card coach-${
            restrictionReached
              ? "stop"
              : restrictionApproaching
                ? "warning"
                : tracking !== "good"
                  ? "paused"
                  : "normal"
          }`}
          role={restrictionReached ? "alert" : "status"}
          aria-live={restrictionReached ? "assertive" : "polite"}
        >
          <span className="coach-pulse" aria-hidden="true">
            <Volume2 size={22} />
          </span>
          <div>
            <span>Relay says</span>
            <strong>
              {starting
                ? "Preparing movement tracking…"
                : paused
                  ? "Session paused. Take your time."
                  : output?.coachMessage ?? providerStatus.label}
            </strong>
          </div>
        </div>
        {exercise.rules.length > 0 && (
          <div className="active-rule-strip" aria-label="Confirmed movement rules">
            {exercise.rules.map((rule) => {
              const ruleState = liveRuleState(rule, output);
              return (
                <span
                  key={rule.id}
                  className={`${rule.hard ? "hard " : ""}rule-${ruleState.state}`}
                  title={rule.scope.description}
                >
                  {ruleState.state === "stop" ||
                  ruleState.state === "watch" ||
                  ruleState.state === "paused" ? (
                    <CircleAlert size={16} aria-hidden="true" />
                  ) : (
                    <Activity size={16} aria-hidden="true" />
                  )}
                  {ruleState.label} · {liveRuleLabel(rule)}
                </span>
              );
            })}
          </div>
        )}
      </section>
      {/* Outside the scrolling readings so pause and stop are always on screen. */}
      <div className="active-controls">
        <button type="button" className="pause-button" onClick={togglePause}>
          {paused ? (
            <Play size={22} fill="currentColor" aria-hidden="true" />
          ) : (
            <Pause size={22} fill="currentColor" aria-hidden="true" />
          )}
          {paused ? "Resume" : "Pause"}
        </button>
        <button type="button" className="mute-button" onClick={() => setMuted(!muted)}>
          {muted ? (
            <VolumeX size={22} aria-hidden="true" />
          ) : (
            <Volume2 size={22} aria-hidden="true" />
          )}
          {muted ? "Sound off" : "Sound on"}
        </button>
        <button type="button" className="stop-button" onClick={stopSession}>
          <Square size={20} fill="currentColor" aria-hidden="true" />
          Stop session
        </button>
      </div>
      {setComplete && (
        <div
          className="set-complete-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="set-complete-title"
        >
          <div className="set-complete-card">
            <span className="success-orbit">
              <Check size={34} aria-hidden="true" />
            </span>
            <div className="flow-eyebrow">
              {exerciseIndex + 1}/{session.exercises.length} exercises
            </div>
            <h2 id="set-complete-title">
              {setIndex + 1 < exercise.dose.sets
                ? `Set ${setIndex + 1} complete`
                : exerciseIndex + 1 < session.exercises.length
                  ? `${exercise.name} complete`
                  : "Session complete"}
            </h2>
            <p>
              {completionStats.repCount} repetition
              {completionStats.repCount === 1 ? "" : "s"} recorded with{" "}
              {completionStats.trackingPercent}% assessable tracking.
            </p>
            {setIndex + 1 >= exercise.dose.sets &&
              exerciseIndex + 1 < session.exercises.length && (
                <div className="transition-guidance">
                  <strong>
                    Position for{" "}
                    {session.exercises[exerciseIndex + 1].name}
                  </strong>
                  <span>
                    {session.exercises[exerciseIndex + 1].notes[0]}
                  </span>
                  <span>
                    Relay will check the required joints and recalibrate your
                    starting position before counting.
                  </span>
                </div>
              )}
            <button
              type="button"
              className="primary-action"
              onClick={continueAfterSet}
              autoFocus
            >
              {setIndex + 1 < exercise.dose.sets
                ? `Start set ${setIndex + 2}`
                : exerciseIndex + 1 < session.exercises.length
                  ? `I’m positioned · Start ${session.exercises[exerciseIndex + 1].name}`
                  : "See session summary"}
              <ArrowRight size={20} aria-hidden="true" />
            </button>
          </div>
        </div>
      )}
    </main>
  );
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function constraintPresentation(
  constraint: ExerciseSummary["constraints"][number],
  prescription: ExercisePrescription,
) {
  const configuredRule = prescription.rules.find(
    (rule) => rule.id === constraint.ruleId,
  );
  const isHardMaximum =
    constraint.kind === "maximum" && configuredRule?.hard;
  const isPosture =
    constraint.kind === "visible-condition" ||
    (constraint.kind === "maximum" && !isHardMaximum);
  const tone =
    constraint.reached > 0
      ? "stop"
      : constraint.approached > 0 ||
          constraint.missed > 0 ||
          constraint.observed > 0
        ? "warning"
        : constraint.assessableReps === 0
          ? "paused"
          : "good";

  if (constraint.assessableReps === 0) {
    return {
      detail: "No repetition had enough reliable feature tracking.",
      headline: "Not assessed",
      tone,
    };
  }
  if (isHardMaximum) {
    if (constraint.reached > 0) {
      return {
        detail: `Configured limit reached on ${constraint.reached} rep${constraint.reached === 1 ? "" : "s"}.`,
        headline: `${constraint.reached} reached`,
        tone,
      };
    }
    if (constraint.approached > 0) {
      return {
        detail: `Approached on ${constraint.approached} rep${constraint.approached === 1 ? "" : "s"}; not reached.`,
        headline: `${constraint.approached} approached`,
        tone,
      };
    }
    return {
      detail: `Not approached in ${constraint.assessableReps} assessable rep${constraint.assessableReps === 1 ? "" : "s"}.`,
      headline: `${constraint.assessableReps} clear`,
      tone,
    };
  }
  if (isPosture) {
    if (constraint.observed > 0) {
      return {
        detail: `Broad visible variation on ${constraint.observed} rep${constraint.observed === 1 ? "" : "s"}.`,
        headline: `${constraint.observed} watch`,
        tone,
      };
    }
    return {
      detail: `No gross variation in ${constraint.assessableReps} assessable rep${constraint.assessableReps === 1 ? "" : "s"}.`,
      headline: `${constraint.assessableReps}/${constraint.assessableReps} steady`,
      tone,
    };
  }
  return {
    detail:
      constraint.missed > 0
        ? `${constraint.met} met; ${constraint.missed} missed.`
        : `${constraint.met} of ${constraint.assessableReps} assessable reps met.`,
    headline: `${constraint.met}/${constraint.assessableReps} met`,
    tone,
  };
}

function SummaryScreen({
  result,
  storageNotice,
  onHome,
  onHistory,
  onPrivacy,
}: {
  result: SessionRunResult;
  storageNotice: string;
  onHome: () => void;
  onHistory: () => void;
  onPrivacy: () => void;
}) {
  const summaries = useMemo(() => summarizeRun(result), [result]);
  const totalDone = summaries.reduce(
    (sum, summary) => sum + summary.completedReps,
    0,
  );
  const totalPlanned = summaries.reduce(
    (sum, summary) => sum + summary.plannedReps,
    0,
  );
  const completion = Math.round((totalDone / Math.max(1, totalPlanned)) * 100);
  const [shared, setShared] = useState(false);

  const share = async () => {
    const text = exportRunJson(result);
    if (navigator.share && typeof File !== "undefined") {
      const file = new File([text], `relay-${result.id}.json`, {
        type: "application/json",
      });
      if (navigator.canShare?.({ files: [file] })) {
        try {
          await navigator.share({
            title: `Relay · ${result.title}`,
            text: "Structured Relay session summary. No camera video included.",
            files: [file],
          });
          setShared(true);
          return;
        } catch (error) {
          if (error instanceof DOMException && error.name === "AbortError") {
            return;
          }
          // Fall through to local download if file sharing fails.
        }
      }
    }
    downloadText(`relay-${result.id}.json`, text);
    setShared(true);
  };

  return (
    <main className="summary-screen">
      <AppHeader
        onHome={onHome}
        onHistory={onHistory}
        onPrivacy={onPrivacy}
        compact
      />
      <ProgressRail step="review" />
      <section className="summary-hero">
        <span className="summary-check">
          <Check size={36} aria-hidden="true" />
        </span>
        <div className="flow-eyebrow">
          {result.completed ? "Session complete" : "Session ended early"}
        </div>
        <h1>Your movement, made visible.</h1>
        <p>
          {totalDone} of {totalPlanned} planned repetitions recorded. Camera
          values below are estimates, not clinical measurements.
        </p>
        {result.source === "replay" && (
          <div className="replay-summary-badge">
            <FlaskConical size={17} aria-hidden="true" />
            Guided replay result · simulated landmarks
          </div>
        )}
        <div className="completion-ring">
          <div
            style={
              {
                "--completion": `${completion}%`,
              } as React.CSSProperties
            }
          >
            <strong>{completion}%</strong>
            <span>complete</span>
          </div>
        </div>
      </section>
      <section className="summary-content">
        {storageNotice && (
          <div className="form-error" role="status">
            <CircleAlert size={20} aria-hidden="true" />
            {storageNotice}
          </div>
        )}
        <div className="summary-heading">
          <div>
            <div className="eyebrow subtle">Exercise details</div>
            <h2>{result.title}</h2>
          </div>
          <span>
            {new Date(result.completedAt).toLocaleString(undefined, {
              month: "short",
              day: "numeric",
              hour: "numeric",
              minute: "2-digit",
            })}
          </span>
        </div>
        <div className="summary-grid">
          {summaries.map((summary, index) => {
            const prescription = result.session.exercises[index];
            const exerciseComplete =
              summary.completedReps >= summary.plannedReps;
            const peakUnit =
              prescription.exerciseId === "seated-knee-extension" ? "%" : "°";
            const peakLabel =
              prescription.exerciseId === "heel-slide"
                ? "estimated peak flexion"
                : prescription.exerciseId === "straight-leg-raise"
                  ? "estimated peak lift"
                  : "best extension progress";
            return (
              <article className="summary-card" key={summary.exerciseId}>
                <div className="summary-card-top">
                  <span className="exercise-number">{index + 1}</span>
                  <div>
                    <strong>
                      {summary.side === "right" ? "Right" : "Left"}{" "}
                      {summary.name}
                    </strong>
                    <span>
                      {summary.completedSets}/{summary.plannedSets} sets ·{" "}
                      {summary.completedReps}/{summary.plannedReps} reps
                    </span>
                  </div>
                  {exerciseComplete ? (
                    <CheckCircle2 size={22} aria-hidden="true" />
                  ) : (
                    <CircleAlert size={22} aria-hidden="true" />
                  )}
                </div>
                <div className="summary-stats">
                  <div>
                    <strong>
                      {summary.medianPeak !== undefined
                        ? `${Math.round(summary.medianPeak)}${peakUnit}`
                        : "—"}
                    </strong>
                    <span>median {peakLabel}</span>
                  </div>
                  <div>
                    <strong>
                      {summary.peakRange
                        ? `${Math.round(summary.peakRange[0])}–${Math.round(
                            summary.peakRange[1],
                          )}${peakUnit}`
                        : "—"}
                    </strong>
                    <span>assessable range</span>
                  </div>
                  <div>
                    <strong>
                      {Math.round(summary.trackingCoverage * 100)}%
                    </strong>
                    <span>tracking coverage</span>
                  </div>
                </div>
                {summary.constraints.length > 0 && (
                  <div
                    className="summary-constraint-grid"
                    aria-label="Body-feature results"
                  >
                    {summary.constraints.map((constraint) => {
                      const presentation = constraintPresentation(
                        constraint,
                        prescription,
                      );
                      const configuredRule = prescription.rules.find(
                        (rule) => rule.id === constraint.ruleId,
                      );
                      return (
                        <div
                          key={constraint.ruleId}
                          className={`constraint-result ${presentation.tone}`}
                        >
                          <span>
                            {presentation.tone === "good" ? (
                              <CheckCircle2 size={17} aria-hidden="true" />
                            ) : (
                              <CircleAlert size={17} aria-hidden="true" />
                            )}
                            {configuredRule
                              ? liveRuleLabel(configuredRule)
                              : constraint.label}
                          </span>
                          <strong>{presentation.headline}</strong>
                          <small>{presentation.detail}</small>
                        </div>
                      );
                    })}
                  </div>
                )}
                <ul className="summary-observations">
                  {summary.holdAssessable > 0 && (
                    <li>
                      <Check size={16} aria-hidden="true" />
                      {prescription.dose.holdSeconds}-second hold met on{" "}
                      {summary.holdMet} of {summary.holdAssessable} assessable
                      reps
                    </li>
                  )}
                  {summary.tempoMet + summary.tempoMissed > 0 && (
                    <li className={summary.tempoMissed > 0 ? "warning" : ""}>
                      <Clock3 size={16} aria-hidden="true" />
                      Tempo met on {summary.tempoMet} rep
                      {summary.tempoMet === 1 ? "" : "s"}; missed on{" "}
                      {summary.tempoMissed}
                    </li>
                  )}
                  {summary.limitedPrimaryPeaks > 0 && (
                    <li className="warning">
                      <CircleAlert size={16} aria-hidden="true" />
                      {summary.limitedPrimaryPeaks} peak estimate
                      {summary.limitedPrimaryPeaks === 1 ? " has" : "s have"}{" "}
                      incomplete tracking coverage
                    </li>
                  )}
                  {summary.observations.map((observation) => (
                    <li key={observation}>
                      <Activity size={16} aria-hidden="true" />
                      {observation} — broad camera observation
                    </li>
                  ))}
                </ul>
              </article>
            );
          })}
        </div>
        <div className="summary-trust">
          <ShieldCheck size={23} aria-hidden="true" />
          <div>
            <strong>What this summary means</strong>
            <p>
              Relay stores structured repetition, timing, event, and quality
              data on this device. It does not store camera video. Estimates may
              vary with phone placement, clothing, lighting, or occlusion.
            </p>
          </div>
        </div>
        <div className="summary-actions">
          <button type="button" className="primary-action" onClick={share}>
            {shared ? (
              <Check size={20} aria-hidden="true" />
            ) : (
              <Download size={20} aria-hidden="true" />
            )}
            {shared ? "Export ready" : "Export or share"}
          </button>
          <button type="button" className="secondary-action" onClick={onHistory}>
            <History size={20} aria-hidden="true" />
            View local history
          </button>
          <button type="button" className="text-button" onClick={onHome}>
            Return home
          </button>
        </div>
      </section>
    </main>
  );
}

function HistoryScreen({
  history,
  onChange,
  onBack,
  onOpen,
  onPrivacy,
}: {
  history: SessionRunResult[];
  onChange: (history: SessionRunResult[]) => void;
  onBack: () => void;
  onOpen: (result: SessionRunResult) => void;
  onPrivacy: () => void;
}) {
  return (
    <main className="history-screen">
      <AppHeader
        onHome={onBack}
        onHistory={() => undefined}
        onPrivacy={onPrivacy}
      />
      <section className="history-content">
        <div className="history-heading">
          <div>
            <div className="eyebrow subtle">Only on this device</div>
            <h1>Your session history</h1>
            <p>
              Structured movement results, never continuous camera video.
            </p>
          </div>
          {history.length > 0 && (
            <button
              type="button"
              className="danger-text-button"
              onClick={() => {
                if (
                  window.confirm(
                    "Delete every locally stored Relay session? This cannot be undone.",
                  )
                ) {
                  clearHistory();
                  onChange([]);
                }
              }}
            >
              <Trash2 size={17} aria-hidden="true" /> Clear all
            </button>
          )}
        </div>
        {history.length === 0 ? (
          <div className="empty-history">
            <span>
              <History size={36} aria-hidden="true" />
            </span>
            <h2>No saved sessions yet</h2>
            <p>
              Complete a live session or guided replay and Relay will keep the
              structured summary here.
            </p>
            <button type="button" className="primary-action" onClick={onBack}>
              Create a session
            </button>
          </div>
        ) : (
          <div className="history-list">
            {history.map((result) => {
              const summaries = summarizeRun(result);
              const reps = summaries.reduce(
                (sum, summary) => sum + summary.completedReps,
                0,
              );
              const planned = summaries.reduce(
                (sum, summary) => sum + summary.plannedReps,
                0,
              );
              return (
                <article key={result.id} className="history-card">
                  <button type="button" onClick={() => onOpen(result)}>
                    <span className="history-date">
                      <strong>
                        {new Date(result.completedAt).toLocaleDateString(
                          undefined,
                          { day: "2-digit" },
                        )}
                      </strong>
                      <small>
                        {new Date(result.completedAt)
                          .toLocaleDateString(undefined, { month: "short" })
                          .toUpperCase()}
                      </small>
                    </span>
                    <span className="history-main">
                      <span>
                        <strong>{result.title}</strong>
                        {result.source === "replay" && (
                          <small className="mini-replay">Replay</small>
                        )}
                      </span>
                      <small>
                        {result.exercises.length} exercise
                        {result.exercises.length === 1 ? "" : "s"} · {reps}/
                        {planned} reps ·{" "}
                        {new Date(result.completedAt).toLocaleTimeString(
                          undefined,
                          { hour: "numeric", minute: "2-digit" },
                        )}
                      </small>
                    </span>
                    <ChevronRight size={22} aria-hidden="true" />
                  </button>
                  <IconButton
                    label={`Delete ${result.title}`}
                    onClick={() => onChange(deleteRun(result.id))}
                  >
                    <Trash2 size={18} />
                  </IconButton>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </main>
  );
}

function PrivacyScreen({ onBack }: { onBack: () => void }) {
  return (
    <main className="privacy-screen">
      <AppHeader
        onHome={onBack}
        onHistory={() => undefined}
        onPrivacy={() => undefined}
      />
      <section className="privacy-content">
        <button type="button" className="text-button" onClick={onBack}>
          <ArrowLeft size={18} aria-hidden="true" /> Back
        </button>
        <div className="privacy-hero">
          <span>
            <LockKeyhole size={34} aria-hidden="true" />
          </span>
          <div className="eyebrow subtle">Local-first by design</div>
          <h1>Your camera is for your session—not our servers.</h1>
          <p>
            Relay processes pose landmarks on your device. It does not save or
            upload continuous camera video. You decide whether structured
            results leave this device.
          </p>
        </div>
        <div className="privacy-grid">
          <article>
            <Camera size={24} aria-hidden="true" />
            <h2>Camera processing</h2>
            <p>
              Frames are analyzed in memory by the local pose model. Relay keeps
              landmarks only long enough to drive the current session and does
              not persist raw frames.
            </p>
            <p>
              The upstream MediaPipe runtime may report performance and
              utilization metrics to Google under its privacy notice. Relay
              does not send raw frames.
            </p>
          </article>
          <article>
            <FileText size={24} aria-hidden="true" />
            <h2>Plan uploads</h2>
            <p>
              Text and PDFs are read in your browser. Photo text recognition
              runs on the device. The original file is discarded after
              interpretation by default.
            </p>
          </article>
          <article>
            <History size={24} aria-hidden="true" />
            <h2>Local history</h2>
            <p>
              Confirmed plans, repetition records, timing, estimate summaries,
              and tracking quality stay in browser storage until you delete
              them.
            </p>
          </article>
          <article>
            <Download size={24} aria-hidden="true" />
            <h2>Export and sharing</h2>
            <p>
              Relay creates a structured JSON record only when you choose to
              export or share. It contains no camera video.
            </p>
          </article>
        </div>
        <div className="product-boundary">
          <HeartPulse size={26} aria-hidden="true" />
          <div>
            <h2>Relay’s product boundary</h2>
            <p>
              Relay helps you follow a plan you already have. It does not
              diagnose injury, interpret pain, assess tissue healing, decide
              whether an exercise is appropriate, or decide whether you are
              ready to progress.
            </p>
            <p>
              Camera-derived values estimate visible movement. They are not
              clinical measurements, and a warning cannot guarantee an
              anatomical restriction was not crossed. Follow the original
              instructions you were given and stop if you feel unsafe or
              experience new or worsening symptoms.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}

export default function RelayApp() {
  const [screen, setScreen] = useState<Screen>("home");
  const [input, setInput] = useState(INITIAL_TEXT);
  const [session, setSession] = useState<ConfirmedSession | null>(null);
  const [mode, setMode] = useState<SessionMode>("live");
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>("user");
  const [scenario, setScenario] = useState<ReplayScenarioId>("correct");
  const [result, setResult] = useState<SessionRunResult | null>(null);
  const [history, setHistory] = useState<SessionRunResult[]>([]);
  const [uploadState, setUploadState] = useState("");
  const [error, setError] = useState("");
  const [listening, setListening] = useState(false);
  const [inputSource, setInputSource] = useState<"typed" | "spoken">("typed");
  const [storageNotice, setStorageNotice] = useState("");

  useEffect(() => {
    // Browser storage is intentionally hydrated only after the client mounts.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHistory(loadHistory());
    if ("serviceWorker" in navigator) {
      void navigator.serviceWorker.register("/sw.js").catch(() => undefined);
    }
  }, []);

  useLayoutEffect(() => {
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }, [screen]);

  const goHome = () => {
    setScreen("home");
    setError("");
    setStorageNotice("");
  };

  const interpret = (source: "typed" | "spoken" | "upload" = "typed") => {
    const interpreted = interpretInstructions(input, source);
    if (interpreted.session.exercises.length === 0) {
      setError(
        interpreted.notices[0] ??
          "Relay could not map these instructions to a supported exercise.",
      );
      return;
    }
    setSession(interpreted.session);
    setMode("live");
    setError("");
    setScreen(
      interpreted.session.clarifications.length > 0 ? "clarify" : "review",
    );
  };

  const handleUpload = async (file: File) => {
    setUploadState("Reading…");
    setError("");
    try {
      const extraction = await extractInstructionsFromFile(file);
      setInput(extraction.text);
      const interpreted = interpretInstructions(extraction.text, "upload");
      if (interpreted.session.exercises.length === 0) {
        setError(
          "Relay read the file but could not find a supported exercise. You can correct the extracted text before trying again.",
        );
        return;
      }
      interpreted.notices.push(extraction.notice);
      setSession(interpreted.session);
      setMode("live");
      setScreen(
        interpreted.session.clarifications.length > 0 ? "clarify" : "review",
      );
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? `Relay could not read this file: ${uploadError.message}`
          : "Relay could not read this file.",
      );
    } finally {
      setUploadState("");
    }
  };

  const startSpeech = () => {
    const SpeechRecognition =
      (
        window as unknown as {
          SpeechRecognition?: new () => {
            continuous: boolean;
            interimResults: boolean;
            lang: string;
            start(): void;
            stop(): void;
            onresult: ((event: {
              results: ArrayLike<{
                0: { transcript: string };
                isFinal: boolean;
              }>;
            }) => void) | null;
            onend: (() => void) | null;
            onerror: (() => void) | null;
          };
          webkitSpeechRecognition?: new () => {
            continuous: boolean;
            interimResults: boolean;
            lang: string;
            start(): void;
            stop(): void;
            onresult: ((event: {
              results: ArrayLike<{
                0: { transcript: string };
                isFinal: boolean;
              }>;
            }) => void) | null;
            onend: (() => void) | null;
            onerror: (() => void) | null;
          };
        }
      ).SpeechRecognition ??
      (
        window as unknown as {
          webkitSpeechRecognition?: new () => {
            continuous: boolean;
            interimResults: boolean;
            lang: string;
            start(): void;
            stop(): void;
            onresult: ((event: {
              results: ArrayLike<{
                0: { transcript: string };
                isFinal: boolean;
              }>;
            }) => void) | null;
            onend: (() => void) | null;
            onerror: (() => void) | null;
          };
        }
      ).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError(
        "Speech input is not supported by this browser. Type or paste the same instructions instead.",
      );
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    setListening(true);
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((resultItem) => resultItem[0].transcript)
        .join(" ");
      setInput(transcript);
      setInputSource("spoken");
    };
    recognition.onend = () => setListening(false);
    recognition.onerror = () => {
      setListening(false);
      setError(
        "Relay could not hear that clearly. Try again or type the instructions.",
      );
    };
    recognition.start();
  };

  const resolveClarification = (
    clarification: Clarification,
    optionId: string,
  ) => {
    if (!session) return;
    const next = applyClarification(session, clarification.id, optionId);
    setSession(next);
    setScreen(next.clarifications.length > 0 ? "clarify" : "review");
  };

  const confirm = () => {
    if (!session) return;
    const conflict = session.exercises.find((exercise) => {
      const targets = exercise.rules.filter(
        (rule) => rule.kind === "target" && rule.value !== undefined,
      );
      const maximums = exercise.rules.filter(
        (rule) => rule.kind === "maximum" && rule.value !== undefined,
      );
      return targets.some((target) =>
        maximums.some(
          (maximum) =>
            maximum.metric === target.metric &&
            (target.value ?? 0) > (maximum.value ?? Infinity),
        ),
      );
    });
    if (conflict) {
      setError(
        `${conflict.name} has a target beyond its maximum. Change one value before confirming.`,
      );
      return;
    }
    const next = {
      ...session,
      confirmed: true,
      estimatesAcknowledged: true,
      updatedAt: new Date().toISOString(),
    };
    const validation = confirmedSessionSchema.safeParse(next);
    if (!validation.success) {
      setError(validation.error.issues[0]?.message ?? "Check the session.");
      return;
    }
    setSession(next);
    setScreen("camera");
    setError("");
  };

  const completeSession = (nextResult: SessionRunResult) => {
    setResult(nextResult);
    const saved = trySaveRun(nextResult);
    setHistory(saved.history);
    setStorageNotice(
      saved.persisted
        ? ""
        : "Your summary is ready, but this browser did not allow Relay to save it locally. Export it now if you want to keep a copy.",
    );
    setScreen("summary");
  };

  if (screen === "home") {
    return (
      <>
        <HomeScreen
          input={input}
          setInput={(value) => {
            setInput(value);
            setInputSource("typed");
          }}
          inputSource={inputSource}
          onInterpret={interpret}
          onDemo={() => {
            setSession(createDemoSession());
            setMode("replay");
            setScreen("review");
            setError("");
          }}
          onManual={() => {
            setSession(createManualSession());
            setMode("live");
            setScreen("review");
            setError("");
          }}
          onUpload={handleUpload}
          uploadState={uploadState}
          onSpeak={startSpeech}
          listening={listening}
          history={history}
          onOpenHistory={() => setScreen("history")}
          onPrivacy={() => setScreen("privacy")}
        />
        {error && (
          <div className="global-toast" role="alert">
            <CircleAlert size={19} aria-hidden="true" />
            <span>{error}</span>
            <IconButton label="Dismiss" onClick={() => setError("")}>
              <X size={16} />
            </IconButton>
          </div>
        )}
      </>
    );
  }

  if (screen === "clarify" && session) {
    return (
      <ClarifyScreen
        session={session}
        onResolve={resolveClarification}
        onBack={goHome}
        onHome={goHome}
      />
    );
  }

  if (screen === "review" && session) {
    return (
      <ReviewScreen
        session={session}
        onChange={setSession}
        onConfirm={confirm}
        onBack={goHome}
        onHome={goHome}
        error={error}
      />
    );
  }

  if (screen === "camera" && session) {
    return (
      <CameraSetup
        session={session}
        mode={mode}
        onMode={setMode}
        cameraFacing={cameraFacing}
        onCameraFacing={setCameraFacing}
        scenario={scenario}
        onScenario={setScenario}
        onStart={() => setScreen("active")}
        onBack={() => setScreen("review")}
        onHome={goHome}
      />
    );
  }

  if (screen === "active" && session) {
    return (
      <ActiveSession
        session={session}
        mode={mode}
        cameraFacing={cameraFacing}
        scenario={scenario}
        onComplete={completeSession}
        onExit={() => setScreen("camera")}
      />
    );
  }

  if (screen === "summary" && result) {
    return (
      <SummaryScreen
        result={result}
        storageNotice={storageNotice}
        onHome={goHome}
        onHistory={() => setScreen("history")}
        onPrivacy={() => setScreen("privacy")}
      />
    );
  }

  if (screen === "history") {
    return (
      <HistoryScreen
        history={history}
        onChange={setHistory}
        onBack={goHome}
        onOpen={(next) => {
          setResult(next);
          setStorageNotice("");
          setScreen("summary");
        }}
        onPrivacy={() => setScreen("privacy")}
      />
    );
  }

  if (screen === "privacy") {
    return <PrivacyScreen onBack={goHome} />;
  }

  return null;
}
