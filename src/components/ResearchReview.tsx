"use client";

import {
  ArrowLeft,
  CheckCircle2,
  Download,
  FlaskConical,
  LockKeyhole,
  Play,
  ShieldCheck,
  Video,
} from "lucide-react";
import { unzipSync } from "fflate";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import type { JointName } from "../motion/types";
import type {
  ObservableAssessment,
  ResearchAnnotation,
  ResearchPoseFrame,
} from "../research/capture-bundle";
import {
  observeDecodedContinuousVideo,
  selectLatestCausalPoseForReplayTime,
  validateDecodedContinuousVideoEvidence,
  type ReplayPoseAttachment,
} from "../research/capture-replay";
import {
  RESEARCH_REVIEW_JOINTS,
  buildBlindedResearchReviewView,
  buildIndependentReviewSidecarExport,
  revealPostSubmissionCollectorComparison,
  verifyResearchReviewArchive,
  type BlindedResearchReviewView,
  type IndependentReviewSidecarExport,
  type PostSubmissionCollectorComparison,
  type VerifiedResearchReviewArchive,
} from "../research/review-sidecar";

type ReviewStep = "load" | "replay" | "review" | "export";

const STEPS: readonly ReviewStep[] = ["load", "replay", "review", "export"];
const CONNECTIONS: readonly (readonly [JointName, JointName])[] = [
  ["left-shoulder", "right-shoulder"],
  ["left-shoulder", "left-elbow"],
  ["left-elbow", "left-wrist"],
  ["right-shoulder", "right-elbow"],
  ["right-elbow", "right-wrist"],
  ["left-shoulder", "left-hip"],
  ["right-shoulder", "right-hip"],
  ["left-hip", "right-hip"],
  ["left-hip", "left-knee"],
  ["left-knee", "left-ankle"],
  ["right-hip", "right-knee"],
  ["right-knee", "right-ankle"],
];

interface ReplayPoseRecord {
  readonly sequence: number;
  readonly callbackMonotonicTimeMs: number;
  readonly frame: ResearchPoseFrame;
}

interface SubmittedReview {
  readonly reviewExport: IndependentReviewSidecarExport;
  readonly comparison: PostSubmissionCollectorComparison;
}

function createCode(prefix: string): string {
  return (
    prefix +
    "-" +
    Date.now().toString(36) +
    "-" +
    Math.random().toString(36).slice(2, 8)
  );
}

function formatTime(value: number): string {
  return (value / 1000).toFixed(3) + " s";
}

function annotationSummary(annotation: ResearchAnnotation): string {
  const values = [
    annotation.metric,
    annotation.observedValue === undefined
      ? undefined
      : "observed " + annotation.observedValue,
    annotation.targetValue === undefined
      ? undefined
      : "target " + annotation.targetValue,
    annotation.delta === undefined ? undefined : "delta " + annotation.delta,
    annotation.unit,
  ].filter(Boolean);
  return values.length > 0 ? values.join(" · ") : "No numeric claim supplied";
}

function replayAttachments(
  frames: readonly ResearchPoseFrame[],
): ReplayPoseAttachment<ReplayPoseRecord>[] {
  return frames.flatMap((frame) =>
    frame.videoTiming
      ? [
          {
            strategy: "nearest-latest-causal-video-frame" as const,
            pose: {
              sequence: frame.sequenceNumber,
              callbackMonotonicTimeMs: frame.callbackMonotonicTimeMs,
              frame,
            },
            videoFrame: {
              sequence: frame.videoTiming.sequence,
              callbackMonotonicTimeMs:
                frame.videoTiming.callbackMonotonicTimeMs,
              mediaTimeMs: frame.videoTiming.mediaTimeMs,
              presentedFrames: frame.videoTiming.presentedFrames,
            },
          },
        ]
      : [],
  );
}

function drawLandmarks(
  canvas: HTMLCanvasElement | null,
  frame: ResearchPoseFrame | null,
): void {
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!frame) {
    context?.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const width = frame.frameWidth || 1280;
  const height = frame.frameHeight || 720;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  if (!context) return;
  context.clearRect(0, 0, width, height);
  context.lineWidth = Math.max(3, width / 320);
  context.strokeStyle = "rgba(168, 255, 207, 0.9)";
  context.fillStyle = "#f28a67";
  for (const [startName, endName] of CONNECTIONS) {
    const start = frame.image[startName];
    const end = frame.image[endName];
    if (!start || !end || start.visibility < 0.35 || end.visibility < 0.35) {
      continue;
    }
    context.beginPath();
    context.moveTo(start.x * width, start.y * height);
    context.lineTo(end.x * width, end.y * height);
    context.stroke();
  }
  for (const landmark of Object.values(frame.image)) {
    if (!landmark || landmark.visibility < 0.35) continue;
    context.beginPath();
    context.arc(
      landmark.x * width,
      landmark.y * height,
      Math.max(3, width / 240),
      0,
      Math.PI * 2,
    );
    context.fill();
  }
}

function downloadLocal(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export default function ResearchReview() {
  const replayVideoRef = useRef<HTMLVideoElement>(null);
  const replayCanvasRef = useRef<HTMLCanvasElement>(null);
  const replayAnimationRef = useRef<number | null>(null);

  const [step, setStep] = useState<ReviewStep>("load");
  const [verified, setVerified] =
    useState<VerifiedResearchReviewArchive | null>(null);
  const [blindedView, setBlindedView] =
    useState<BlindedResearchReviewView | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [watchedReplay, setWatchedReplay] = useState(false);
  const [overlaysInspected, setOverlaysInspected] = useState(false);
  const [referenceInspected, setReferenceInspected] = useState(false);
  const [attemptInspected, setAttemptInspected] = useState(false);
  const [reviewDecodeConfirmed, setReviewDecodeConfirmed] = useState(false);
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState<SubmittedReview | null>(null);

  const [raterCode, setRaterCode] = useState("");
  const [blindedDeclared, setBlindedDeclared] = useState(false);
  const [independentDeclared, setIndependentDeclared] = useState(false);
  const [movementIdentity, setMovementIdentity] = useState("");
  const [assessment, setAssessment] =
    useState<ObservableAssessment>("uncertain");
  const [realizedDifference, setRealizedDifference] = useState("");
  const [evidenceStartMs, setEvidenceStartMs] = useState("");
  const [evidenceEndMs, setEvidenceEndMs] = useState("");
  const [evidenceJoints, setEvidenceJoints] = useState<JointName[]>([]);
  const [confidence, setConfidence] = useState("0.5");
  const [abstainReason, setAbstainReason] = useState("");

  const attachments = useMemo(
    () => replayAttachments(blindedView?.session.poseFrames ?? []),
    [blindedView],
  );

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    return () => {
      if (replayAnimationRef.current !== null) {
        cancelAnimationFrame(replayAnimationRef.current);
      }
    };
  }, []);

  function drawReplayFrame(): void {
    const video = replayVideoRef.current;
    if (!video) return;
    const selected = selectLatestCausalPoseForReplayTime(
      video.currentTime * 1000,
      attachments,
      blindedView?.session.videoFrameTimings,
    );
    drawLandmarks(replayCanvasRef.current, selected?.pose.frame ?? null);
  }

  function animateReplay(): void {
    drawReplayFrame();
    const video = replayVideoRef.current;
    if (video && !video.paused && !video.ended) {
      replayAnimationRef.current = requestAnimationFrame(animateReplay);
    } else {
      replayAnimationRef.current = null;
    }
  }

  function beginReplayAnimation(): void {
    if (replayAnimationRef.current !== null) {
      cancelAnimationFrame(replayAnimationRef.current);
    }
    replayAnimationRef.current = requestAnimationFrame(animateReplay);
  }

  function stopReplayAnimation(): void {
    if (replayAnimationRef.current !== null) {
      cancelAnimationFrame(replayAnimationRef.current);
      replayAnimationRef.current = null;
    }
    drawReplayFrame();
  }

  function seekTo(mediaTimeMs: number): void {
    const video = replayVideoRef.current;
    if (!video) return;
    video.currentTime = mediaTimeMs / 1000;
    drawReplayFrame();
  }

  async function confirmReviewVideoDecode(): Promise<void> {
    const video = replayVideoRef.current;
    if (!video || !blindedView) return;
    setNotice("");
    try {
      const observed = await observeDecodedContinuousVideo(video);
      const validation = validateDecodedContinuousVideoEvidence(
        observed,
        blindedView.session.videoFrameTimings,
      );
      if (!validation.valid) {
        throw new Error(validation.issues.map((entry) => entry.message).join(" "));
      }
      setReviewDecodeConfirmed(true);
      drawReplayFrame();
    } catch (error) {
      setReviewDecodeConfirmed(false);
      setNotice(
        error instanceof Error
          ? error.message
          : "This browser did not decode the continuous video.",
      );
    }
  }

  async function loadArchive(file: File | undefined): Promise<void> {
    if (!file) return;
    setLoading(true);
    setNotice("");
    setSubmitted(null);
    setWatchedReplay(false);
    setOverlaysInspected(false);
    setReferenceInspected(false);
    setAttemptInspected(false);
    setReviewDecodeConfirmed(false);
    try {
      const compressed = new Uint8Array(await file.arrayBuffer());
      const members = unzipSync(compressed);
      const nextVerified = await verifyResearchReviewArchive(file.name, members);
      const nextView = buildBlindedResearchReviewView(nextVerified);
      const nextUrl = URL.createObjectURL(
        new Blob([Uint8Array.from(nextVerified.videoBytes)], {
          type: nextView.session.video.mimeType,
        }),
      );
      setVerified(nextVerified);
      setBlindedView(nextView);
      setVideoUrl(nextUrl);
      setStep("replay");
    } catch (error) {
      setVerified(null);
      setBlindedView(null);
      setVideoUrl(null);
      setStep("load");
      setNotice(
        error instanceof Error
          ? error.message
          : "The archive could not be verified locally.",
      );
    } finally {
      setLoading(false);
    }
  }

  function toggleJoint(joint: JointName, checked: boolean): void {
    setEvidenceJoints((current) =>
      checked
        ? current.includes(joint)
          ? current
          : [...current, joint]
        : current.filter((value) => value !== joint),
    );
  }

  async function commitIndependentReview(): Promise<void> {
    if (!verified) return;
    setLoading(true);
    setNotice("");
    const createdAt = new Date().toISOString();
    const isUnobservable = assessment === "unobservable";
    const annotation: ResearchAnnotation = {
      annotationId: createCode("independent-annotation"),
      raterCode,
      raterRole: "independent-reviewer",
      reviewStatus: "draft-unreviewed",
      movementIdentity,
      assessment,
      realizedErrorType:
        isUnobservable || realizedDifference.trim() === ""
          ? undefined
          : realizedDifference,
      evidenceJoints: [...evidenceJoints],
      evidenceStartMs:
        isUnobservable || evidenceStartMs.trim() === ""
          ? undefined
          : Number(evidenceStartMs),
      evidenceEndMs:
        isUnobservable || evidenceEndMs.trim() === ""
          ? undefined
          : Number(evidenceEndMs),
      evidenceClock: isUnobservable
        ? undefined
        : "encoded-replay-approximate-ms",
      desiredTargetSource: isUnobservable ? "unknown" : "captured-reference",
      confidence: Number(confidence),
      abstainReason: abstainReason.trim() === "" ? undefined : abstainReason,
      createdAt,
    };
    try {
      const reviewExport = await buildIndependentReviewSidecarExport(
        verified,
        annotation,
        {
          blinded: blindedDeclared,
          independentlyProduced: independentDeclared,
          replay: {
            watchedContinuousVideo: watchedReplay,
            overlaysVisible: overlaysInspected,
            referenceMarkerInspected: referenceInspected,
            attemptMarkerInspected: attemptInspected,
            encodedOverlayAlignment: "APPROXIMATE_UNCALIBRATED",
          },
        },
      );
      const comparison = revealPostSubmissionCollectorComparison(
        verified,
        reviewExport.sidecar,
      );
      setSubmitted({ reviewExport, comparison });
      setStep("export");
    } catch (error) {
      setNotice(
        error instanceof Error
          ? error.message
          : "The independent review is incomplete.",
      );
    } finally {
      setLoading(false);
    }
  }

  function downloadReview(): void {
    if (!submitted) return;
    downloadLocal(
      new Blob([submitted.reviewExport.json], {
        type: "application/json;charset=utf-8",
      }),
      submitted.reviewExport.fileName,
    );
    downloadLocal(
      new Blob([submitted.reviewExport.sha256Sidecar], {
        type: "text/plain;charset=utf-8",
      }),
      submitted.reviewExport.fileName + ".sha256",
    );
  }

  const comparison = submitted?.comparison;
  const sourceIntervals = blindedView?.session.intervals ?? [];

  return (
    <main className="research-capture-screen">
      <header className="research-header">
        <Link href="/" className="research-back">
          <ArrowLeft size={19} /> Relay
        </Link>
        <div className="research-step-label">
          Procedurally blinded review · {STEPS.indexOf(step) + 1} of {STEPS.length}
        </div>
      </header>

      <section className="research-hero">
        <div>
          <div className="flow-eyebrow">
            <FlaskConical size={17} /> Independent evidence review
          </div>
          <h1>See the motion. Commit your judgment. Then compare.</h1>
          <p>
            Verify one quarantined episode locally, replay its uninterrupted
            video with causal pose overlays, and write a hash-bound independent
            observation before any collector judgment is revealed.
          </p>
        </div>
        <div className="research-boundary-card">
          <ShieldCheck size={22} />
          <div>
            <strong>Observable motion only.</strong>
            <span>
              This workflow does not establish clinical correctness, safety,
              diagnosis, pain, force, load, contact, or intent.
            </span>
          </div>
        </div>
      </section>

      <section className="research-workspace">
        <div className="research-progress">
          {STEPS.map((item, index) => (
            <div
              key={item}
              className={index <= STEPS.indexOf(step) ? "active" : ""}
            >
              <span>{index + 1}</span>
              <small>{item}</small>
            </div>
          ))}
        </div>

        {step === "load" && (
          <div className="research-panel">
            <div className="research-panel-heading">
              <div>
                <span>Fail-closed intake</span>
                <h2>Load one complete v2 episode.</h2>
              </div>
              <LockKeyhole size={22} />
            </div>
            <p className="research-helper">
              The archive never leaves this browser. Every uncompressed member
              is re-hashed, the continuous video is matched to its exact episode
              declaration, and any missing, changed, extra, non-v2, or
              non-quarantined content is rejected.
            </p>
            <div className="research-form-grid two">
              <label>
                Quarantined episode archive
                <input
                  type="file"
                  accept=".relay-episode-v2.zip,application/zip"
                  disabled={loading}
                  onChange={(event) =>
                    void loadArchive(event.currentTarget.files?.[0])
                  }
                />
              </label>
            </div>
            <div className="research-review-warning">
              <ShieldCheck size={19} />
              <span>
                Collector labels, frozen protocol targets, executor values, and
                prior cues remain hidden in this UI through annotation. The
                source ZIP itself contains those records and is not a sealed
                blind package; independence is a reviewer self-attestation.
              </span>
            </div>
          </div>
        )}

        {(step === "replay" || step === "review") && blindedView && videoUrl && (
          <div className="research-camera-strip">
            <div
              className={
                "research-camera " +
                (blindedView.session.cameraFacing === "user" ? "front" : "")
              }
            >
              <video
                ref={replayVideoRef}
                src={videoUrl}
                playsInline
                controls
                onLoadedMetadata={() => void confirmReviewVideoDecode()}
                onTimeUpdate={drawReplayFrame}
                onSeeked={drawReplayFrame}
                onPlay={beginReplayAnimation}
                onPause={stopReplayAnimation}
                onEnded={stopReplayAnimation}
              />
              <canvas ref={replayCanvasRef} aria-hidden="true" />
              <div className="research-camera-status">
                <span className="live" /> Verified continuous local replay
              </div>
            </div>
            <div className="research-signal-contract">
              <strong>Pre-commit redacted UI surface</strong>
              <span>{blindedView.session.frameCount} retained pose frames</span>
              <span>{sourceIntervals.length} marked capture intervals</span>
              <span>Source-clock latest-past pose; encoded alignment uncalibrated</span>
              <span>Episode bytes and continuous video hash-bound locally</span>
              <small>Prior movement judgment, targets, values, and cue are hidden.</small>
            </div>
          </div>
        )}

        {step === "replay" && blindedView && (
          <div className="research-panel">
            <div className="research-panel-heading">
              <div>
                <span>Independent replay</span>
                <h2>Inspect the continuous evidence first.</h2>
              </div>
              <Video size={22} />
            </div>
            <p className="research-helper">
              Watch neutral time, transitions, the captured reference, and the
              attempt. Interval names are capture markers—not judgments that
              either motion is correct, ideal, safe, or clinically appropriate.
              Their placement over the encoded video and the visible pose overlay
              are approximate because the source-to-Blob clock mapping is not calibrated.
            </p>
            <div className="research-review-warning">
              {reviewDecodeConfirmed ? (
                <CheckCircle2 size={19} />
              ) : (
                <ShieldCheck size={19} />
              )}
              <span>
                {reviewDecodeConfirmed
                  ? "This review browser decoded the local video with finite duration and nonzero dimensions. Source-to-Blob alignment remains uncalibrated."
                  : "Waiting for this review browser to decode and validate the local video."}
              </span>
            </div>
            <div className="research-record-grid">
              {sourceIntervals.map((interval, index) => (
                <article
                  className="research-record-card"
                  key={interval.kind + "-" + index}
                >
                  <div className="research-record-heading">
                    <span className="research-record-number">{index + 1}</span>
                    <div>
                      <strong>Approximate {interval.kind} marker</strong>
                      <small>
                        {formatTime(interval.begin.mediaTimeMs)} to{" "}
                        {formatTime(interval.end.mediaTimeMs)}
                      </small>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="secondary-action"
                    onClick={() => seekTo(interval.begin.mediaTimeMs)}
                  >
                    <Play size={17} /> Seek to marker
                  </button>
                </article>
              ))}
            </div>
            <div className="research-consent-block">
              <label>
                <input
                  type="checkbox"
                  checked={watchedReplay}
                  onChange={(event) => setWatchedReplay(event.target.checked)}
                />
                I watched the continuous replay and inspected the approximate
                reference and attempt markers with the approximate pose overlay.
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={overlaysInspected}
                  onChange={(event) => setOverlaysInspected(event.target.checked)}
                />
                The approximate pose overlay was visible while I inspected the replay.
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={referenceInspected}
                  onChange={(event) => setReferenceInspected(event.target.checked)}
                />
                I inspected the approximate reference interval marker.
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={attemptInspected}
                  onChange={(event) => setAttemptInspected(event.target.checked)}
                />
                I inspected the approximate attempt interval marker.
              </label>
            </div>
            <div className="research-actions">
              <button
                type="button"
                className="primary-action"
                disabled={
                  !watchedReplay ||
                  !reviewDecodeConfirmed ||
                  !overlaysInspected ||
                  !referenceInspected ||
                  !attemptInspected
                }
                onClick={() => setStep("review")}
              >
                Continue to independent review
              </button>
            </div>
          </div>
        )}

        {step === "review" && blindedView && (
          <div className="research-panel">
            <div className="research-panel-heading">
              <div>
                <span>Independent commitment</span>
                <h2>Describe only what you can support.</h2>
              </div>
              <LockKeyhole size={22} />
            </div>
            <p className="research-helper">
              Use approximate encoded-replay milliseconds for evidence timing;
              they are not source-clock-exact until timeline calibration exists. If the
              observable signal cannot support the requested distinction,
              choose unobservable and state why; more certainty is not a better
              label.
            </p>

            <div className="research-form-grid three">
              <label>
                Distinct reviewer code
                <input
                  value={raterCode}
                  onChange={(event) => setRaterCode(event.target.value)}
                  placeholder="R-REVIEW-002"
                />
              </label>
              <label>
                Movement identity
                <input
                  value={movementIdentity}
                  onChange={(event) => setMovementIdentity(event.target.value)}
                  placeholder="What movement did you observe?"
                />
              </label>
              <label>
                Observability / assessment
                <select
                  value={assessment}
                  onChange={(event) =>
                    setAssessment(event.target.value as ObservableAssessment)
                  }
                >
                  <option value="uncertain">Observable — qualitative reference comparison</option>
                  <option value="unobservable">Unobservable — abstain</option>
                </select>
              </label>
            </div>

            {assessment !== "unobservable" && (
              <>
                <label className="research-wide-field">
                  Independently observed difference
                  <input
                    value={realizedDifference}
                    onChange={(event) => setRealizedDifference(event.target.value)}
                    placeholder="Describe the observable difference without diagnosing it"
                  />
                </label>
                <div className="research-form-grid two">
                  <label>
                    Evidence start (approx. encoded replay ms)
                    <input
                      type="number"
                      step="any"
                      value={evidenceStartMs}
                      onChange={(event) => setEvidenceStartMs(event.target.value)}
                    />
                  </label>
                  <label>
                    Evidence end (approx. encoded replay ms)
                    <input
                      type="number"
                      step="any"
                      value={evidenceEndMs}
                      onChange={(event) => setEvidenceEndMs(event.target.value)}
                    />
                  </label>
                </div>
              </>
            )}

            <fieldset className="research-joint-fieldset">
              <legend>Evidence joints</legend>
              <div>
                {RESEARCH_REVIEW_JOINTS.map((joint) => (
                  <label key={joint}>
                    <input
                      type="checkbox"
                      checked={evidenceJoints.includes(joint)}
                      onChange={(event) => toggleJoint(joint, event.target.checked)}
                    />
                    {joint.replaceAll("-", " ")}
                  </label>
                ))}
              </div>
            </fieldset>

            {assessment !== "unobservable" && (
              <div className="research-review-warning">
                <ShieldCheck size={19} />
                <span>
                  This review form intentionally does not ask you to invent exact
                  angles, target values, deltas, or a correction cue. Those require
                  a separately validated measurement tool and declared source.
                </span>
              </div>
            )}

            <div className="research-form-grid two">
              <label>
                Confidence from 0 to 1
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={confidence}
                  onChange={(event) => setConfidence(event.target.value)}
                />
              </label>
              <label>
                Abstention or uncertainty reason
                <input
                  value={abstainReason}
                  onChange={(event) => setAbstainReason(event.target.value)}
                  placeholder={
                    assessment === "unobservable"
                      ? "Required: what is missing or ambiguous?"
                      : "Optional"
                  }
                />
              </label>
            </div>

            <div className="research-consent-block">
              <label>
                <input
                  type="checkbox"
                  checked={blindedDeclared}
                  onChange={(event) => setBlindedDeclared(event.target.checked)}
                />
                I did not see the collector annotation, protocol-execution
                target values, or prior cue before producing this review.
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={independentDeclared}
                  onChange={(event) => setIndependentDeclared(event.target.checked)}
                />
                I produced this annotation independently from the replayed raw
                evidence and did not copy another rater&apos;s judgment.
              </label>
            </div>
            <div className="research-actions">
              <button
                type="button"
                className="primary-action"
                disabled={loading || !blindedDeclared || !independentDeclared}
                onClick={() => void commitIndependentReview()}
              >
                <LockKeyhole size={17} /> Commit immutable review
              </button>
            </div>
          </div>
        )}

        {step === "export" && submitted && comparison && (
          <div className="research-panel research-export-panel">
            <div className="research-export-icon">
              <CheckCircle2 size={31} />
            </div>
            <span>Post-commit comparison</span>
            <h2>The independent label is now append-only.</h2>
            <p>
              Collector evidence is revealed only after commitment. The exact
              comparison below is not consensus or adjudication, and the source
              episode remains quarantined. The derived state is{" "}
              <strong>{comparison.derivedReviewState.state}</strong>.
            </p>

            <div className="research-record-grid">
              <article className="research-record-card">
                <div className="research-record-heading">
                  <span className="research-record-number">C</span>
                  <div>
                    <strong>Collector annotation</strong>
                    <small>Revealed after independent commitment</small>
                  </div>
                </div>
                <div className="research-record-stats">
                  <span>{comparison.collectorAnnotation.movementIdentity}</span>
                  <span>{comparison.collectorAnnotation.assessment}</span>
                </div>
                <p className="research-helper">
                  {annotationSummary(comparison.collectorAnnotation)}
                </p>
                <p className="research-helper">
                  {comparison.collectorAnnotation.cue ??
                    comparison.collectorAnnotation.abstainReason ??
                    "No collector cue or abstention reason."}
                </p>
              </article>
              <article className="research-record-card complete">
                <div className="research-record-heading">
                  <span className="research-record-number">I</span>
                  <div>
                    <strong>Independent annotation</strong>
                    <small>Hash-bound in the export below</small>
                  </div>
                </div>
                <div className="research-record-stats">
                  <span>{comparison.independentAnnotation.movementIdentity}</span>
                  <span>{comparison.independentAnnotation.assessment}</span>
                </div>
                <p className="research-helper">
                  {annotationSummary(comparison.independentAnnotation)}
                </p>
                <p className="research-helper">
                  {comparison.independentAnnotation.cue ??
                    comparison.independentAnnotation.abstainReason ??
                    "No independent cue or abstention reason."}
                </p>
              </article>
            </div>

            <div className="research-hash-card">
              <small>Exact disagreement fields</small>
              <code>
                {comparison.disagreements.length > 0
                  ? comparison.disagreements.map(({ field }) => field).join(", ")
                  : "No claim-bearing field disagreements under exact comparison"}
              </code>
            </div>
            <div className="research-hash-card">
              <small>Source episode.json SHA-256</small>
              <code>
                {submitted.reviewExport.sidecar.sourceEpisode.episodeJsonSha256}
              </code>
            </div>
            <div className="research-hash-card">
              <small>Independent review sidecar SHA-256</small>
              <code>{submitted.reviewExport.sha256}</code>
            </div>
            <div className="research-export-actions">
              <button type="button" className="primary-action" onClick={downloadReview}>
                <Download size={18} /> Export review JSON + SHA
              </button>
            </div>
            <div className="research-review-warning">
              <ShieldCheck size={19} />
              <span>
                A second procedurally blinded independent review is still
                required. Any disagreement requires a distinct third-rater
                adjudication; none of these steps alone establishes clinical truth.
              </span>
            </div>
          </div>
        )}
      </section>

      {notice && <div className="research-notice">{notice}</div>}
    </main>
  );
}
