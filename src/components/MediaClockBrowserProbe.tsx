"use client";

import {
  ArrowLeft,
  CheckCircle2,
  Clock3,
  Download,
  FlaskConical,
  Play,
  ShieldOff,
  TriangleAlert,
} from "lucide-react";
import Link from "next/link";
import { useRef, useState } from "react";
import {
  runSyntheticMediaClockProbe,
  SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG,
  type SyntheticMediaClockProbeProgress,
  type SyntheticMediaClockProbeRun,
} from "../research/media-clock-browser-probe";
import styles from "./MediaClockBrowserProbe.module.css";

const PROGRESS_LABELS: Record<SyntheticMediaClockProbeProgress, string> = {
  preparing: "Preparing a private synthetic stream…",
  recording: "Recording generated frame identities…",
  hashing: "Hashing the exact encoded Blob…",
  replaying: "Decoding the recorded video frame by frame…",
  fitting: "Fitting the source-to-encoded clock map…",
};

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function formatNumber(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : value.toFixed(digits);
}

export default function MediaClockBrowserProbe() {
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const sourceVideoRef = useRef<HTMLVideoElement>(null);
  const sourceDecodeCanvasRef = useRef<HTMLCanvasElement>(null);
  const replayVideoRef = useRef<HTMLVideoElement>(null);
  const replayDecodeCanvasRef = useRef<HTMLCanvasElement>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] =
    useState<SyntheticMediaClockProbeProgress | null>(null);
  const [run, setRun] = useState<SyntheticMediaClockProbeRun | null>(null);
  const [notice, setNotice] = useState("");

  async function runOneTrial(): Promise<void> {
    if (running) return;
    const sourceCanvas = sourceCanvasRef.current;
    const sourceVideo = sourceVideoRef.current;
    const sourceDecodeCanvas = sourceDecodeCanvasRef.current;
    const replayVideo = replayVideoRef.current;
    const replayDecodeCanvas = replayDecodeCanvasRef.current;
    if (
      !sourceCanvas ||
      !sourceVideo ||
      !sourceDecodeCanvas ||
      !replayVideo ||
      !replayDecodeCanvas
    ) {
      setNotice("The local probe surface is not ready. Reload this page and retry.");
      return;
    }

    setRunning(true);
    setRun(null);
    setNotice("");
    setProgress("preparing");
    try {
      const nextRun = await runSyntheticMediaClockProbe(
        {
          sourceCanvas,
          sourceVideo,
          sourceDecodeCanvas,
          replayVideo,
          replayDecodeCanvas,
        },
        { onProgress: setProgress },
      );
      setRun(nextRun);
      if (nextRun.report.status !== "COMPLETED") {
        setNotice(nextRun.report.issues.join(" "));
      }
    } catch (error) {
      setNotice(
        error instanceof Error ? error.message : "The synthetic trial failed.",
      );
    } finally {
      setRunning(false);
      setProgress(null);
    }
  }

  function downloadReport(): void {
    if (!run) return;
    downloadBlob(
      new Blob([JSON.stringify(run.report, null, 2)], {
        type: "application/json",
      }),
      `relay-synthetic-media-clock-${Date.now()}.json`,
    );
  }

  function downloadVideo(): void {
    if (!run?.recordedBlob) return;
    const extension = run.recordedBlob.type.includes("mp4") ? "mp4" : "webm";
    downloadBlob(
      run.recordedBlob,
      `relay-synthetic-media-clock-${Date.now()}.${extension}`,
    );
  }

  const report = run?.report ?? null;
  const fit = report?.fit ?? null;
  const validSource = report
    ? report.rawObservations.source.filter((entry) => entry.decoded.valid).length
    : 0;
  const validEncoded = report
    ? report.rawObservations.encoded.filter((entry) => entry.decoded.valid).length
    : 0;

  return (
    <main className={styles.screen}>
      <header className={styles.header}>
        <Link className={styles.back} href="/research-capture">
          <ArrowLeft size={17} aria-hidden="true" />
          Research capture
        </Link>
        <span>Timing mechanics probe</span>
      </header>

      <section className={styles.hero}>
        <div>
          <span className={styles.eyebrow}>
            <FlaskConical size={15} aria-hidden="true" />
            Synthetic browser measurement
          </span>
          <h1>Measure the clock gap without filming anyone.</h1>
          <p>
            One click generates numbered frames, records them through this
            browser, decodes the resulting video, and reports the fitted timing
            map with every raw callback observation.
          </p>
        </div>
        <aside className={styles.boundary}>
          <ShieldOff size={23} aria-hidden="true" />
          <div>
            <strong>Not camera calibration</strong>
            <span>
              This never requests camera access. It cannot admit a human capture
              or prove that the result transfers to a physical camera path.
            </span>
          </div>
        </aside>
      </section>

      <section className={styles.workspace}>
        <div className={styles.intro}>
          <div>
            <span>Bounded protocol</span>
            <h2>One trial, one exact artifact</h2>
          </div>
          <p>
            {SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.frameCount} frames at an
            exploratory {SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.targetFrameIntervalMs}
            -ms target interval; automatic stop by {(
              SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.maximumWallTimeMs / 1000
            ).toFixed(0)} seconds.
          </p>
        </div>

        <div className={styles.previewGrid} aria-label="Synthetic probe video path">
          <article className={styles.previewCard}>
            <div>
              <span>01</span>
              <strong>Generated pixels</strong>
            </div>
            <canvas
              ref={sourceCanvasRef}
              width={SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.width}
              height={SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.height}
            />
          </article>
          <article className={styles.previewCard}>
            <div>
              <span>02</span>
              <strong>Source presentation</strong>
            </div>
            <video ref={sourceVideoRef} muted playsInline />
          </article>
          <article className={styles.previewCard}>
            <div>
              <span>03</span>
              <strong>Encoded replay</strong>
            </div>
            <video ref={replayVideoRef} muted playsInline />
          </article>
        </div>

        <canvas ref={sourceDecodeCanvasRef} className={styles.decodeCanvas} />
        <canvas ref={replayDecodeCanvasRef} className={styles.decodeCanvas} />

        <div className={styles.runBar}>
          <button type="button" onClick={runOneTrial} disabled={running}>
            {running ? <Clock3 size={18} /> : <Play size={18} />}
            {running ? "Trial running" : "Run one synthetic trial"}
          </button>
          <div aria-live="polite">
            <strong>{running && progress ? PROGRESS_LABELS[progress] : "Local only"}</strong>
            <span>
              No upload, camera permission, participant data, or admission flag.
            </span>
          </div>
        </div>

        {notice ? (
          <div className={styles.notice} role="alert">
            <TriangleAlert size={18} aria-hidden="true" />
            <span>{notice}</span>
          </div>
        ) : null}

        {report ? (
          <section className={styles.results} aria-labelledby="clock-results-title">
            <div className={styles.resultsHeading}>
              <div>
                {report.status === "COMPLETED" ? (
                  <CheckCircle2 size={20} aria-hidden="true" />
                ) : (
                  <TriangleAlert size={20} aria-hidden="true" />
                )}
                <span>{report.evidenceClass}</span>
                <h2 id="clock-results-title">
                  {fit?.status ?? report.status}
                </h2>
              </div>
              <span className={styles.scopePill}>{report.scope}</span>
            </div>

            <div className={styles.metricGrid}>
              <article>
                <span>Decoded IDs</span>
                <strong>{validSource} source</strong>
                <small>{validEncoded} replay</small>
              </article>
              <article>
                <span>Matched frames</span>
                <strong>{fit?.matchedFrameCount ?? 0}</strong>
                <small>{fit?.sourceUnmatchedCount ?? 0} source unmatched</small>
              </article>
              <article>
                <span>Affine slope β</span>
                <strong>{formatNumber(fit?.affineMap?.slope, 6)}</strong>
                <small>
                  intercept {formatNumber(fit?.affineMap?.interceptMs)} ms
                </small>
              </article>
              <article>
                <span>Absolute residual</span>
                <strong>{formatNumber(fit?.residuals?.p95AbsoluteMs)} ms p95</strong>
                <small>
                  {formatNumber(fit?.residuals?.maximumAbsoluteMs)} ms maximum
                </small>
              </article>
            </div>

            <div className={styles.hashCard}>
              <span>Encoded Blob SHA-256</span>
              <code>{report.recorder.blobSha256 ?? "Unavailable"}</code>
              <small>
                {report.recorder.blobByteLength?.toLocaleString() ?? "—"} bytes · {report.recorder.realizedMimeType ?? "unknown codec"}
              </small>
            </div>

            <div className={styles.claimBoundary}>
              <strong>Claim boundary retained in the report</strong>
              <ul>
                {report.claimBoundary.map((claim) => (
                  <li key={claim}>{claim}</li>
                ))}
              </ul>
            </div>

            <div className={styles.downloads}>
              <button type="button" onClick={downloadReport}>
                <Download size={17} aria-hidden="true" />
                Download report + raw observations
              </button>
              <button
                type="button"
                onClick={downloadVideo}
                disabled={!run?.recordedBlob}
              >
                <Download size={17} aria-hidden="true" />
                Download encoded test video
              </button>
            </div>
          </section>
        ) : null}
      </section>
    </main>
  );
}
