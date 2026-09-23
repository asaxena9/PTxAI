#!/usr/bin/env python3
"""Build a self-contained replay.html: skeleton animation + recorded model predictions.

REPLAYS already-recorded evaluation rows; no model is loaded and no inference
runs here. Steps: (1) read the frozen task plan (which sealed_test task saw
which UI-PRMD Kinect Positions/Angles file, and which 12-frame window),
(2) read the recorded prediction rows for those tasks, (3) read the raw
Kinect joint data straight out of the official UI-PRMD archive zips (no
extraction to disk, no copy outside this sandbox), (4) reconstruct absolute
2D joint positions per frame using the same hierarchical position+angle
composition as the dataset authors' own `source/Animation.m` (see
`reconstruct_episode` for the citation and caveat), (5) emit one static
replay.html with joint trajectories + predictions embedded as JSON, plus
vanilla JS/CSS to animate and browse them. No external CDNs, no network.

Dependencies: Python 3 stdlib + numpy. No torch. No live model.
Run: python3 replay.py   ->   writes replay.html next to this script.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Paths (read-only; nothing outside this script's own directory is written)
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

LANE = REPO_ROOT / "experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0"
TASK_PLAN_PATH = LANE / "TASK_PLAN.json"
JOB_PATH = LANE / "results/lfm1p2b-science-lora8-a100-002/job.json"
SNAPSHOT_DIR = LANE / "results/lfm1p2b-science-lora8-a100-002/emergency-readonly-snapshots-2026-08-13"
PREDICTIONS_PATH = SNAPSHOT_DIR / "post-sealed-complete-1200.jsonl"

ADMISSION = REPO_ROOT / "experiments/public-data-audit/ui-prmd-admission-2026-08-12"
ORDINARY_ZIP = ADMISSION / "source/full-archives/Segmented_Movements.zip"
INCORRECT_ZIP = ADMISSION / "source/full-archives/Incorrect_Segmented_Movements.zip"
ZIP_BY_BRANCH = {"ordinary": ORDINARY_ZIP, "acted_nonoptimal": INCORRECT_ZIP}

OUTPUT_HTML = HERE / "replay.html"

# UI-PRMD 22-joint Kinect topology: joint order and bone connectivity are taken
# verbatim from the dataset authors' own visualization script (the `J` matrix
# and joint-name comment in ui-prmd-admission-2026-08-12/source/Animation.m),
# independently re-declared as SOURCE_JOINT_NAMES in this lane's task_plan.py.
JOINT_NAMES = [
    "waist", "spine", "chest", "neck", "head", "head_tip",
    "left_collar", "left_upper_arm", "left_forearm", "left_hand",
    "right_collar", "right_upper_arm", "right_forearm", "right_hand",
    "left_upper_leg", "left_lower_leg", "left_foot", "left_toes",
    "right_upper_leg", "right_lower_leg", "right_foot", "right_toes",
]
assert len(JOINT_NAMES) == 22

# (parent, child) 0-indexed, converted from Animation.m's 1-indexed `J` matrix.
BONES = [
    (3, 2), (5, 4), (4, 2), (2, 1), (1, 0),
    (2, 6), (6, 7), (7, 8), (8, 9),
    (2, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16), (16, 17),
    (0, 18), (18, 19), (19, 20), (20, 21),
]
assert len(BONES) == 21

# Mirrors GRAMMAR.json's "activities" and task_plan.py's MOVEMENTS (lane root).
MOVEMENTS = {
    "m01": "deep squat", "m02": "hurdle step", "m03": "inline lunge",
    "m04": "side lunge", "m05": "sit-to-stand",
    "m06": "standing active straight-leg raise",
    "m07": "standing shoulder abduction", "m08": "standing shoulder extension",
    "m09": "standing shoulder internal-external rotation",
    "m10": "standing shoulder scaption",
}
NOMINAL_RATE_HZ = 30.0

# Skeleton reconstruction
def euler_to_R(xyz_deg: np.ndarray) -> np.ndarray:
    """Rz(z) @ Ry(y) @ Rx(x), matching Animation.m's `eulers_2_rot_matrix`."""
    x, y, z = np.radians(xyz_deg)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx

def reconstruct_episode(offsets: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Absolute 3D joint positions per frame from raw parent-relative offsets.

    Raw Kinect `..._positions.txt` rows are NOT absolute joint coordinates:
    joint 0 (waist) is absolute, every other joint is a bone-length offset in
    its parent's rotated local frame. Recovering a plausible skeleton means
    composing each offset with the cumulative rotation from the matching
    `..._angles.txt` row, exactly as UI-PRMD's own `source/Animation.m` does
    (see that file for the full derivation). This is a frame-independent
    port of its "relative to absolute coordinates" section, including its
    documented quirk: the four limb branches (L/R arm, L/R leg) seed their
    rotation from the branch root's OWN angle triplet, not the fully
    compounded parent rotation. Animation.m's own comment says this "is not
    entirely correct for all lower body joints" (named offenders: m03, m04,
    m05) -- carried over unmodified rather than silently "fixed": a
    different convention would be a guess, not a citation.

    offsets, angles: (n_frames, 22, 3) arrays. Returns (n_frames, 22, 3).
    """
    n = offsets.shape[0]
    out = np.zeros((n, 22, 3))
    for i in range(n):
        off, ang = offsets[i], angles[i]
        pos = np.zeros((22, 3))
        rot: dict[int, np.ndarray] = {0: np.eye(3)}
        pos[0] = off[0]

        def place(child: int, parent: int, r: np.ndarray) -> None:
            pos[child] = r @ off[child] + pos[parent]
            rot[child] = r

        # spine -> chest -> neck -> head -> head tip
        place(1, 0, euler_to_R(ang[0]))
        place(2, 1, rot[1] @ euler_to_R(ang[1]))
        place(3, 2, rot[2] @ euler_to_R(ang[2]))
        place(4, 3, rot[3] @ euler_to_R(ang[3]))
        place(5, 4, rot[4] @ euler_to_R(ang[4]))
        # left arm, seeded from chest's own angle (Animation.m parity, see docstring)
        place(6, 2, euler_to_R(ang[2]))
        place(7, 6, rot[6] @ euler_to_R(ang[6]))
        place(8, 7, rot[7] @ euler_to_R(ang[7]))
        place(9, 8, rot[8] @ euler_to_R(ang[8]))
        # right arm
        place(10, 2, euler_to_R(ang[2]))
        place(11, 10, rot[10] @ euler_to_R(ang[10]))
        place(12, 11, rot[11] @ euler_to_R(ang[11]))
        place(13, 12, rot[12] @ euler_to_R(ang[12]))
        # left leg, seeded from waist's own angle
        place(14, 0, euler_to_R(ang[0]))
        place(15, 14, rot[14] @ euler_to_R(ang[14]))
        place(16, 15, rot[15] @ euler_to_R(ang[15]))
        place(17, 16, rot[16] @ euler_to_R(ang[16]))
        # right leg
        place(18, 0, euler_to_R(ang[0]))
        place(19, 18, rot[18] @ euler_to_R(ang[18]))
        place(20, 19, rot[19] @ euler_to_R(ang[19]))
        place(21, 20, rot[20] @ euler_to_R(ang[20]))
        out[i] = pos
    return out

# Data loading
def load_task_plan() -> dict:
    return json.loads(TASK_PLAN_PATH.read_text())

def load_predictions() -> dict:
    rows = {}
    with PREDICTIONS_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["task_id"]] = row
    return rows

def read_zip_member(zip_path: Path, member: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(member).decode("utf-8")

def parse_frames(text: str) -> np.ndarray:
    rows = [line for line in text.strip().splitlines() if line.strip()]
    arr = np.array([[float(x) for x in line.split(",")] for line in rows])
    if arr.shape[1] != 66:
        raise ValueError(f"expected 66 columns (22 joints x xyz), got {arr.shape[1]}")
    return arr.reshape(arr.shape[0], 22, 3)

def angles_member_for(positions_member: str) -> str:
    return positions_member.replace("/Positions/", "/Angles/").replace(
        "_positions", "_angles"
    )

# Episode selection. Deterministic, documented, outcome-blind rule: walk the 10
# UI-PRMD movements in order, alternate the held-out subject (s09/s10) and the
# preferred collection branch (ordinary/acted_nonoptimal) by movement index,
# and take the lowest-numbered episode available for that (movement, subject)
# pair. Nothing about correctness feeds the rule -- the resulting mix of
# correct/wrong rows is whatever the recorded model actually produced. See
# INVENTORY.md / README.md for the measured outcome.
def select_episodes(by_ancestor: dict) -> list:
    subjects = ["s09", "s10"]
    branch_pref = [("ordinary", "acted_nonoptimal"), ("acted_nonoptimal", "ordinary")]
    chosen = []
    for i, mov in enumerate([f"m{n:02d}" for n in range(1, 11)]):
        subject = subjects[i % 2]
        preferred, fallback = branch_pref[i % 2]
        candidates = [
            anc for anc in by_ancestor
            if anc.split("|")[1] == mov and anc.split("|")[2] == subject
        ]

        def sort_key(anc: str) -> tuple:
            branch, _, _, ep = anc.split("|")
            order = 0 if branch == preferred else 1
            return (ep, order)

        candidates.sort(key=sort_key)
        if candidates:
            chosen.append(candidates[0])
    return chosen

# Payload construction
def build_episode(ancestor: str, tasks: list, preds: dict) -> dict:
    branch, movement_id, subject, episode_id = ancestor.split("|")
    any_task = tasks[0]
    identity = any_task["audit_only_identity"]
    positions_member = identity["archive_member"]
    angles_member = angles_member_for(positions_member)
    zip_path = ZIP_BY_BRANCH[branch]

    pos_text = read_zip_member(zip_path, positions_member)
    sha = hashlib.sha256(pos_text.encode("utf-8")).hexdigest()
    if sha != identity["member_sha256"]:
        raise ValueError(f"positions sha256 mismatch for {positions_member}")
    ang_text = read_zip_member(zip_path, angles_member)

    pos3 = parse_frames(pos_text)
    ang3 = parse_frames(ang_text)
    if pos3.shape[0] != identity["source_frame_count"]:
        raise ValueError(f"frame count mismatch for {positions_member}")
    if ang3.shape[0] != pos3.shape[0]:
        raise ValueError(f"positions/angles frame count differ for {ancestor}")

    absolute = reconstruct_episode(pos3, ang3)  # (n, 22, 3)
    # Front view: screen-x = data-x, screen-y = -data-y (data is +Y up).
    xy = np.stack([absolute[:, :, 0], -absolute[:, :, 1]], axis=-1)  # (n, 22, 2)
    pad = 0.15
    xmin, ymin = xy.reshape(-1, 2).min(axis=0)
    xmax, ymax = xy.reshape(-1, 2).max(axis=0)
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    bbox = [
        round(float(xmin - pad * dx), 2), round(float(ymin - pad * dy), 2),
        round(float(xmax + pad * dx), 2), round(float(ymax + pad * dy), 2),
    ]

    phases = {}
    for task in tasks:
        phase = task["target"]["phase"]
        row = preds[task["task_id"]]
        task_identity = task["audit_only_identity"]
        phases[phase] = {
            "task_id": task["task_id"],
            "window": [task_identity["window_start_inclusive"], task_identity["window_stop_exclusive"]],
            "target_text": task["target"]["text"].rstrip("\n"),
            "predicted_text": row["generated_text"].rstrip("\n"),
            "parsed_activity": row.get("parsed_activity"),
            "parsed_phase": row.get("parsed_phase"),
            "activity_correct": bool(row["activity_correct"]),
            "phase_correct": bool(row["phase_correct"]),
            "joint_exact": bool(row["joint_exact"]),
            "grammar_valid": bool(row.get("grammar_valid", False)),
        }

    return {
        "id": ancestor,
        "branch": branch,
        "movement_id": movement_id,
        "movement_name": MOVEMENTS[movement_id],
        "subject": subject,
        "episode": episode_id,
        "archive_member": positions_member,
        "frame_count": int(pos3.shape[0]),
        "nominal_fps": NOMINAL_RATE_HZ,
        "bbox": bbox,
        "frames": [[[round(float(v), 2) for v in joint] for joint in frame] for frame in xy],
        "phases": phases,
    }

def compute_global_stats(sealed_tasks: list, preds: dict) -> dict:
    rows = [preds[t["task_id"]] for t in sealed_tasks]
    n = len(rows)
    activity_dist = Counter(r["parsed_activity"] for r in rows)
    phase_dist = Counter(r["parsed_phase"] for r in rows)
    return {
        "n_sealed_tasks": n,
        "n_sealed_episodes": len({t["physical_ancestor"] for t in sealed_tasks}),
        "activity_correct_rate": round(sum(r["activity_correct"] for r in rows) / n, 4),
        "phase_correct_rate": round(sum(r["phase_correct"] for r in rows) / n, 4),
        "joint_exact_rate": round(sum(r["joint_exact"] for r in rows) / n, 4),
        "predicted_activity_distribution": dict(activity_dist.most_common()),
        "predicted_phase_distribution": dict(phase_dist.most_common()),
    }

# Validation (cannot open a browser here, so validate the payload directly)
def validate_payload(payload: dict) -> list:
    problems = []
    if not payload["episodes"]:
        problems.append("no episodes in payload")
    for ep in payload["episodes"]:
        frames = ep["frames"]
        if len(frames) != ep["frame_count"]:
            problems.append(f"{ep['id']}: frame_count mismatch")
        for fi, frame in enumerate(frames):
            if len(frame) != 22:
                problems.append(f"{ep['id']}: frame {fi} has {len(frame)} joints, expected 22")
                break
            for joint in frame:
                if len(joint) != 2:
                    problems.append(f"{ep['id']}: frame {fi} joint has {len(joint)} coords, expected 2")
                    break
                for v in joint:
                    if not np.isfinite(v):
                        problems.append(f"{ep['id']}: frame {fi} has non-finite coordinate")
                    if abs(v) > 100000:
                        problems.append(f"{ep['id']}: frame {fi} coordinate out of range: {v}")
        x0, y0, x1, y1 = ep["bbox"]
        if not (np.isfinite([x0, y0, x1, y1]).all() and x1 > x0 and y1 > y0):
            problems.append(f"{ep['id']}: bad bbox {ep['bbox']}")
        for b0, b1 in BONES:
            if not (0 <= b0 < 22 and 0 <= b1 < 22):
                problems.append(f"bone index out of range: {(b0, b1)}")
        for phase_name in ("early", "middle", "late"):
            if phase_name not in ep["phases"]:
                problems.append(f"{ep['id']}: missing phase {phase_name}")
                continue
            p = ep["phases"][phase_name]
            if not p["target_text"] or not p["predicted_text"]:
                problems.append(f"{ep['id']}/{phase_name}: empty text field")
    return problems

# HTML rendering
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PTxAI replay demo v0 -- UI-PRMD held-person canary</title>
<style>
  :root { --bg:#14161c; --panel:#1c1f28; --panel2:#22262f; --border:#2f3441; --text:#e8eaf0; --muted:#9aa1b4; --accent:#6ea8fe; --good:#35c46b; --bad:#e2574c; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  header { padding: 18px 24px 8px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0 0 6px; font-size: 20px; }
  header p { margin: 4px 0; color: var(--muted); font-size: 13.5px; }
  .banner { margin: 10px 0 0; padding: 10px 14px; background: #2a2110; border: 1px solid #5a4415; border-radius: 8px; color: #ecd08a; font-size: 13px; }
  .stats { margin: 8px 0 0; padding: 10px 14px; background: var(--panel2); border: 1px solid var(--border); border-radius: 8px; font-size: 13px; color: var(--muted); display: flex; gap: 22px; flex-wrap: wrap; }
  .stats b { color: var(--text); }
  main { display: flex; gap: 16px; padding: 16px 24px 24px; align-items: flex-start; flex-wrap: wrap; }
  .col-left { flex: 1 1 480px; min-width: 380px; }
  .col-right { flex: 1 1 420px; min-width: 340px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 14px; }
  .card h2 { margin: 0 0 10px; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  #stage { width: 100%; height: 420px; background: #0f1116; border-radius: 8px; display: block; }
  .controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
  button { background: var(--panel2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 12px; cursor: pointer; font-size: 13px; }
  button:hover { border-color: var(--accent); }
  button.active { background: var(--accent); color: #0b0f19; border-color: var(--accent); }
  input[type=range] { flex: 1; }
  .timeline { position: relative; height: 22px; margin-top: 10px; }
  .timeline .track { position: absolute; top: 9px; left: 0; right: 0; height: 4px; background: var(--panel2); border-radius: 2px; }
  .timeline .win { position: absolute; top: 6px; height: 10px; border-radius: 3px; opacity: .55; }
  .timeline .win.early { background: #6ea8fe; } .timeline .win.middle { background: #b98af8; } .timeline .win.late { background: #f2a65a; }
  .timeline .playhead { position: absolute; top: 2px; width: 2px; height: 18px; background: #fff; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); margin-top: 6px; }
  .legend span::before { content: "\25A0"; margin-right: 5px; }
  .legend .early::before { color: #6ea8fe; } .legend .middle::before { color: #b98af8; } .legend .late::before { color: #f2a65a; }
  .ep-list { display: flex; flex-direction: column; gap: 6px; max-height: 190px; overflow-y: auto; }
  .ep-btn { text-align: left; display: flex; justify-content: space-between; gap: 8px; }
  .ep-btn .tag { color: var(--muted); font-size: 11.5px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; letter-spacing: .03em; }
  .badge.correct { background: rgba(53,196,107,.18); color: var(--good); }
  .badge.wrong { background: rgba(226,87,76,.18); color: var(--bad); }
  .phase-row { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; background: var(--panel2); }
  .phase-row .head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .phase-row .name { font-weight: 600; text-transform: capitalize; }
  .phase-row .text-row { font-size: 13px; margin: 3px 0; }
  .phase-row .label { color: var(--muted); display: inline-block; width: 78px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .pass8 { font-size: 12px; color: var(--muted); margin-top: 6px; border-top: 1px dashed var(--border); padding-top: 6px; }
  footer { padding: 10px 24px 28px; color: var(--muted); font-size: 12px; }
  code { background: var(--panel2); padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<header>
  <h1>PTxAI replay demo v0 &mdash; recorded predictions vs. ground truth, UI-PRMD held-person canary</h1>
  <p>Run <code>__RUN_ID__</code> &middot; <code>__MODEL_ID__</code>, LoRA rank 8, __TRAINABLE_PCT__% of parameters trainable
     &middot; evidence class <b>MEASURED-PILOT</b> &middot; scientific decision <b>__SCI_DECISION__</b></p>
  <div class="banner">
    This replays <b>recorded</b> evaluation rows (no live model runs in this page). Held-person N=2
    (subjects s09, s10 of 10). This is an indication of current model behavior, not a capability claim.
    See README.md in this folder for what this demo does and does not show.
  </div>
  <div class="stats">
    <div>held-out sealed tasks: <b>__N_TASKS__</b> (__N_EPISODES__ episodes &times; 3 phase probes)</div>
    <div>joint-exact rate: <b>__JOINT_EXACT_RATE__%</b></div>
    <div>activity-correct rate: <b>__ACTIVITY_RATE__%</b></div>
    <div>phase-correct rate: <b>__PHASE_RATE__%</b></div>
    <div>most-predicted activity: <b>__TOP_ACTIVITY__</b> (__TOP_ACTIVITY_PCT__% of all 1200 rows)</div>
    <div>predicted phase is always: <b>__TOP_PHASE__</b> (__TOP_PHASE_PCT__% of all 1200 rows)</div>
  </div>
</header>
<main>
  <div class="col-left">
    <div class="card">
      <h2>Skeleton replay &mdash; front view (Kinect X/Y, depth dropped)</h2>
      <canvas id="stage" width="900" height="700"></canvas>
      <div class="timeline"><div class="track"></div><div id="win-early" class="win early"></div>
        <div id="win-middle" class="win middle"></div><div id="win-late" class="win late"></div>
        <div id="playhead" class="playhead"></div></div>
      <div class="legend"><span class="early">early window</span><span class="middle">middle window</span><span class="late">late window</span></div>
      <div class="controls">
        <button id="btn-play">Play</button>
        <span>speed</span>
        <button class="speed" data-speed="0.25">0.25x</button>
        <button class="speed" data-speed="0.5">0.5x</button>
        <button class="speed active" data-speed="1">1x</button>
        <button class="speed" data-speed="2">2x</button>
        <input id="scrub" type="range" min="0" max="1" step="1" value="0">
        <span id="frame-readout" class="mono muted">frame 0</span>
      </div>
    </div>
    <div class="card">
      <h2>Episode picker (__N_EPISODES__ episodes, both held subjects, 10 movements)</h2>
      <div id="ep-list" class="ep-list"></div>
    </div>
  </div>
  <div class="col-right">
    <div class="card">
      <h2 id="current-ep-title">Episode</h2>
      <div id="phases"></div>
      <div class="pass8">pass@8 sampled outcomes: not available locally for this run &mdash; the raw 9,600-row
        sample file lived only in the final GCS result archive, which was audited but not retained in this
        repository (see <code>RECOVERED_GCS_RESULT_AUDIT.md</code>). This page shows single-shot generations only.</div>
    </div>
  </div>
</main>
<footer>
  Generated __GENERATED_AT__ by <code>experiments/e0-native-fusion/replay_demo_v0/replay.py</code>.
  Predictions: <code>__PRED_SOURCE__</code>. Task/window plan: <code>__PLAN_SOURCE__</code>.
  Raw motion: official UI-PRMD archives (<code>Segmented_Movements.zip</code> / <code>Incorrect_Segmented_Movements.zip</code>),
  reconstructed to absolute joint positions using the dataset authors' own <code>source/Animation.m</code> hierarchy
  (documented lower-body caveat carried over, see README.md).
</footer>
<script>
const DATA = __PAYLOAD_JSON__, BONES = DATA.bones;
const canvas = document.getElementById('stage'), ctx = canvas.getContext('2d');
let epIndex = 0, frameIndex = 0, playing = false, speed = 1, lastT = 0, acc = 0;
const currentEp = () => DATA.episodes[epIndex];

function fitTransform(ep) {
  const [x0,y0,x1,y1] = ep.bbox;
  const s = Math.min(canvas.width/(x1-x0), canvas.height/(y1-y0));
  const ox = (canvas.width-(x1-x0)*s)/2 - x0*s, oy = (canvas.height-(y1-y0)*s)/2 - y0*s;
  return (pt) => [pt[0]*s+ox, pt[1]*s+oy];
}

function draw() {
  const ep = currentEp(), frame = ep.frames[frameIndex], tf = fitTransform(ep);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 4; ctx.strokeStyle = '#6ea8fe'; ctx.lineCap = 'round';
  for (const [a, b] of BONES) {
    const [ax, ay] = tf(frame[a]), [bx, by] = tf(frame[b]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  ctx.fillStyle = '#e8eaf0';
  for (const j of frame) { const [x, y] = tf(j); ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill(); }
  document.getElementById('frame-readout').textContent =
    `frame ${frameIndex+1}/${ep.frame_count}  (${(frameIndex/ep.nominal_fps).toFixed(2)}s)`;
  document.getElementById('scrub').max = ep.frame_count - 1;
  document.getElementById('scrub').value = frameIndex;
  const total = ep.frame_count;
  for (const name of ['early', 'middle', 'late']) {
    const [w0, w1] = ep.phases[name].window, el = document.getElementById('win-'+name);
    el.style.left = (w0/total*100)+'%'; el.style.width = Math.max((w1-w0)/total*100, 1)+'%';
  }
  document.getElementById('playhead').style.left = (frameIndex/Math.max(total-1,1)*100)+'%';
}

function renderEpisodeList() {
  const list = document.getElementById('ep-list');
  list.innerHTML = '';
  DATA.episodes.forEach((ep, i) => {
    const anyCorrect = Object.values(ep.phases).some(p => p.joint_exact);
    const btn = document.createElement('button');
    btn.className = 'ep-btn' + (i === epIndex ? ' active' : '');
    btn.innerHTML = `<span>${ep.movement_name} &mdash; ${ep.subject} / ${ep.episode} / ${ep.branch}</span>` +
      `<span class="tag">${anyCorrect ? 'has a correct probe' : 'all 3 probes wrong'}</span>`;
    btn.onclick = () => { epIndex = i; frameIndex = 0; renderEpisodeList(); renderPhases(); draw(); };
    list.appendChild(btn);
  });
}

function renderPhases() {
  const ep = currentEp();
  document.getElementById('current-ep-title').textContent =
    `${ep.movement_name} — ${ep.subject}, episode ${ep.episode}, ${ep.branch} branch (${ep.frame_count} frames @ ${ep.nominal_fps} Hz nominal)`;
  const el = document.getElementById('phases');
  el.innerHTML = '';
  for (const name of ['early', 'middle', 'late']) {
    const p = ep.phases[name], label = p.joint_exact ? 'CORRECT' : 'WRONG', cls = p.joint_exact ? 'correct' : 'wrong';
    const row = document.createElement('div');
    row.className = 'phase-row';
    row.innerHTML = `<div class="head"><span class="name">${name} probe</span><span class="badge ${cls}">${label}</span></div>
      <div class="text-row"><span class="label">ground truth</span>${p.target_text}</div>
      <div class="text-row"><span class="label">model said</span>${p.predicted_text}</div>
      <div class="text-row"><span class="label">window</span><span class="mono">frames ${p.window[0]}-${p.window[1]} (source-file indices, 12-frame model input)</span></div>`;
    el.appendChild(row);
  }
}

document.getElementById('btn-play').onclick = (e) => {
  playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';
  lastT = performance.now(); requestAnimationFrame(tick);
};
document.querySelectorAll('button.speed').forEach(b => b.onclick = () => {
  document.querySelectorAll('button.speed').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); speed = parseFloat(b.dataset.speed);
});
document.getElementById('scrub').oninput = (e) => { frameIndex = parseInt(e.target.value, 10); draw(); };

function tick(t) {
  if (!playing) return;
  const dt = (t - lastT) / 1000; lastT = t; acc += dt * speed;
  const stepEvery = 1 / currentEp().nominal_fps;
  while (acc >= stepEvery) { frameIndex = (frameIndex + 1) % currentEp().frame_count; acc -= stepEvery; }
  draw();
  requestAnimationFrame(tick);
}

renderEpisodeList(); renderPhases(); draw();
</script>
</body>
</html>
"""

def render_html(payload: dict, meta: dict) -> str:
    html = HTML_TEMPLATE
    stats = payload["global_stats"]
    top_activity, top_activity_n = next(iter(stats["predicted_activity_distribution"].items()))
    top_phase, top_phase_n = next(iter(stats["predicted_phase_distribution"].items()))
    subs = {
        "__RUN_ID__": meta["run_id"],
        "__MODEL_ID__": meta["model_id"],
        "__TRAINABLE_PCT__": f"{meta['trainable_pct']:.4f}",
        "__SCI_DECISION__": meta["scientific_decision"],
        "__N_TASKS__": str(stats["n_sealed_tasks"]),
        "__N_EPISODES__": str(len(payload["episodes"])),
        "__JOINT_EXACT_RATE__": f"{stats['joint_exact_rate']*100:.1f}",
        "__ACTIVITY_RATE__": f"{stats['activity_correct_rate']*100:.1f}",
        "__PHASE_RATE__": f"{stats['phase_correct_rate']*100:.1f}",
        "__TOP_ACTIVITY__": top_activity,
        "__TOP_ACTIVITY_PCT__": f"{100*top_activity_n/stats['n_sealed_tasks']:.1f}",
        "__TOP_PHASE__": top_phase,
        "__TOP_PHASE_PCT__": f"{100*top_phase_n/stats['n_sealed_tasks']:.1f}",
        "__GENERATED_AT__": meta["generated_at"],
        "__PRED_SOURCE__": meta["pred_source"],
        "__PLAN_SOURCE__": meta["plan_source"],
        "__PAYLOAD_JSON__": json.dumps(payload, separators=(",", ":")),
    }
    for key, val in subs.items():
        html = html.replace(key, val)
    return html

# Main
def main() -> int:
    if not ORDINARY_ZIP.exists() or not INCORRECT_ZIP.exists():
        print("FATAL: UI-PRMD source archives not found:", ORDINARY_ZIP, INCORRECT_ZIP, file=sys.stderr)
        return 1

    plan = load_task_plan()
    preds = load_predictions()
    sealed_tasks = [t for t in plan["tasks"] if t["split"] == "sealed_test"]

    by_ancestor = defaultdict(list)
    for t in sealed_tasks:
        by_ancestor[t["physical_ancestor"]].append(t)

    ancestors = select_episodes(by_ancestor)
    print(f"selected {len(ancestors)} episodes:")
    episodes = []
    skipped = []
    for anc in ancestors:
        try:
            ep = build_episode(anc, by_ancestor[anc], preds)
            episodes.append(ep)
            print(f"  OK   {anc:32s} frames={ep['frame_count']:<4d} member={ep['archive_member']}")
        except Exception as exc:  # degrade gracefully, do not fabricate
            skipped.append((anc, str(exc)))
            print(f"  SKIP {anc:32s} reason={exc}")

    if not episodes:
        print("FATAL: no episodes could be built", file=sys.stderr)
        return 1

    global_stats = compute_global_stats(sealed_tasks, preds)
    payload = {
        "schema": "ptxai.replay_demo_v0.payload.v1",
        "joint_names": JOINT_NAMES,
        "bones": BONES,
        "episodes": episodes,
        "global_stats": global_stats,
        "skipped_episodes": skipped,
    }

    problems = validate_payload(payload)
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"validation: OK ({len(episodes)} episodes, "
          f"{sum(e['frame_count'] for e in episodes)} total frames, all coordinates finite/in-range)")

    job = json.loads(JOB_PATH.read_text())
    trainable_pct = 100 * job["expected_trainable_parameters"] / job["expected_active_parameters"]
    meta = {
        "run_id": job["run_id"],
        "model_id": job["model_identity"]["repository"],
        "trainable_pct": trainable_pct,
        "scientific_decision": "FAIL_OR_INCONCLUSIVE_LOCAL_H1",
        "generated_at": "2026-08-14 (replay_demo_v0)",
        "pred_source": str(PREDICTIONS_PATH.relative_to(REPO_ROOT)),
        "plan_source": str(TASK_PLAN_PATH.relative_to(REPO_ROOT)),
    }
    html = render_html(payload, meta)
    OUTPUT_HTML.write_text(html)
    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"wrote {OUTPUT_HTML} ({size_kb:.1f} KiB)")
    print(f"global stats over all {global_stats['n_sealed_tasks']} sealed tasks: "
          f"joint_exact={global_stats['joint_exact_rate']*100:.1f}% "
          f"activity_correct={global_stats['activity_correct_rate']*100:.1f}% "
          f"phase_correct={global_stats['phase_correct_rate']*100:.1f}%")
    print("predicted-activity distribution:", global_stats["predicted_activity_distribution"])
    print("predicted-phase distribution:", global_stats["predicted_phase_distribution"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
