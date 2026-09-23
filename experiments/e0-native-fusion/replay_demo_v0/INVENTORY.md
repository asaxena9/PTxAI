# Inventory: what actually exists for the replay demo

Date: 2026-08-14. Scope: `experiments/e0-native-fusion/replay_demo_v0/` only (read-only
everywhere else). This is a factual inventory, not a design doc -- see `README.md` for
what the demo does/doesn't show.

## Bottom line

Both pieces the demo needs are present locally and were cross-verified against each
other: (1) recorded held-person prediction rows from the LoRA-rank-8 run, and (2) the
exact raw Kinect motion files those predictions were made about, reachable through a
frozen task plan that names the exact archive member per task. **No pose data is
missing for any row used in the demo.** The one real gap is pass@8 (see below).

## 1. Prediction rows that exist locally

Run `lfm1p2b-science-lora8-002` (LiquidAI/LFM2.5-1.2B-Base, rank-8 LoRA touching
1,936,768 / 1,172,277,376 parameters = 0.165214%). Its final GCS result archive
(`result.tar.gz`, 10,369,391 bytes, sha256 `4e8f236b...`) was independently audited in
`results/lfm1p2b-science-lora8-a100-002/RECOVERED_GCS_RESULT_AUDIT.md` but **the archive
itself was not retained in the repo** ("downloaded to an isolated temporary directory
... not retained"). What *is* retained locally is the emergency snapshot directory,
which the recovery audit confirms is byte-identical / an exact subset of that final
archive for the two files that matter here:

| file | rows | what it is | path |
|---|---:|---|---|
| `post-sealed-complete-1200.jsonl` | 1,200 | one row per held-person (`sealed_test`, subjects s09/s10) task: target activity/phase, free-generated `generated_text`, parsed activity/phase, `activity_correct`/`phase_correct`/`joint_exact` booleans, grammar-constrained diagnostic candidates | `experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/results/lfm1p2b-science-lora8-a100-002/emergency-readonly-snapshots-2026-08-13/post-sealed-complete-1200.jsonl` |
| `interventions-partial-15964.jsonl` | 15,964 / 16,800 planned | single-generation ablations (identity, query-only, pose-zero, characteristic-limb deletion, matched deletion, phase-swap control) over the same 1,200 tasks; only `phase_swap_original_body_control` is incomplete (364/1,200) | same directory |

These 1,200 sealed-test rows cover **400 distinct episodes**: 2 held subjects x 10
movements x 10 episodes x 2 collection branches (ordinary / acted_nonoptimal), each
episode probed at 3 derived-phase windows (early/middle/late) = 1,200. This is the full
held-person surface, not a sample -- `python3 replay.py` reads all 1,200 rows to compute
the honest global stats banner, then renders 10 of the 400 episodes (see part 4).

**Not present locally: pass@8.** The recovery audit reports pass@8 as 9,600 rows (1,200
tasks x 8 samples) inside the *un-retained* final archive. It was never copied into this
repo. The demo says so explicitly in the UI rather than omitting it silently.

## 2. Pose files for those exact rows: locally available, verified

`TASK_PLAN.json` (repo root of the lane, 6,000 tasks: 3,600 fit + 1,200 development +
1,200 sealed_test) names, for every task, the exact archive member, its sha256/crc32,
the source episode's frame count, and the 12-frame window
(`window_start_inclusive`/`window_stop_exclusive`) that was actually shown to the model
-- e.g. `"archive_member": "Segmented Movements/Kinect/Positions/m04_s07_e03_positions.txt"`.
Every sealed_test task's `task_id` matches a row in `post-sealed-complete-1200.jsonl` 1:1
(checked: 1,200/1,200 both directions, zero orphans either side).

The named archive members live in the two official UI-PRMD zips, present on disk and
never modified:

| archive | branch | bytes | member count |
|---|---|---:|---:|
| `experiments/public-data-audit/ui-prmd-admission-2026-08-12/source/full-archives/Segmented_Movements.zip` | ordinary | 206,962,745 | 3,987 (Kinect+Vicon x Angles+Positions x ~10x10x10) |
| `experiments/public-data-audit/ui-prmd-admission-2026-08-12/source/full-archives/Incorrect_Segmented_Movements.zip` | acted_nonoptimal (`_inc`) | 213,516,724 | 3,987 |

`replay.py` reads Positions/Angles members straight out of these zips with
`zipfile.ZipFile.read()` -- no extraction to disk, nothing copied outside this sandbox.
For every episode it renders it re-hashes the Positions text and asserts it matches
`TASK_PLAN.json`'s recorded sha256 before use (see `build_episode` in `replay.py`); all
10 rendered episodes passed. Kinect positions are the admitted asset here
(`full_corpus/ADMISSION_CONTRACT.md`: "Complete Kinect positions ... ALLOWED,
CLAIM-LIMITED"); Vicon was not needed and is not used.

**Coverage check for the held subjects specifically:** the smaller 24-file
"outcome-blind sample" under `ui-prmd-admission-2026-08-12/sample/` only covers subjects
s01/s05/s10 and movements m01/m05/m07 -- it does **not** cover s09 at all. The full zips
do: `unzip -l` confirms 200 Kinect-Positions members matching `_s09_|_s10_` in the
ordinary zip alone, and both branches have full m01-m10 x e01-e10 coverage for s09/s10.
So the demo reads from the **full archives**, not the small sample directory.

## 3. Skeleton reconstruction note (read before trusting the picture)

Raw `..._positions.txt` rows are **not** absolute joint coordinates -- only joint 0
(waist) is absolute; every other joint is a parent-relative bone offset that must be
composed with the matching `..._angles.txt` row's rotation to place it in world space.
`replay.py::reconstruct_episode` ports this composition directly from the dataset
authors' own `experiments/public-data-audit/ui-prmd-admission-2026-08-12/source/Animation.m`,
including that script's own documented caveat that the result "is not entirely correct
for all lower body joints" for m03/m04/m05. This is carried over unmodified rather than
silently "fixed." See `README.md` for what this means for trust in the picture.

## 4. Episodes rendered in this v0 demo

10 of the 400 available sealed-test episodes, chosen by a deterministic, outcome-blind
rule (documented in `replay.py::select_episodes`): walk the 10 movements in order,
alternate subject s09/s10 and preferred branch by movement index, take the
lowest-numbered episode available. Nothing about correctness fed the rule. Measured
result of applying it:

- All 10 UI-PRMD movements represented, both held subjects (5 episodes each), both
  collection branches present.
- 8 of 10 are clean failures on all 3 phase probes (deep squat, hurdle step, side
  lunge, sit-to-stand, straight-leg raise, shoulder abduction, internal-external
  rotation, scaption).
- 2 of 10 have one correct probe each (inline lunge/late, standing shoulder
  extension/late) -- and even those are wrong on their other 2 probes.

This is not cherry-picked toward either outcome; it is what the mechanical rule
produced, and it happens to match the base rate closely (see next section).

## 5. Measured finding worth flagging: near-total mode collapse

Computed directly from all 1,200 sealed rows (`compute_global_stats` in `replay.py`,
printed to stdout on every run, embedded in the page's stats banner):

- `joint_exact` (activity AND phase both correct): **5.7%** (68/1,200)
- `activity_correct`: **16.4%** (197/1,200)
- `phase_correct`: **33.3%** (400/1,200)
- Predicted phase is `"late"` on **100.0%** of all 1,200 rows -- never early or middle.
- Predicted activity is `"standing shoulder extension"` on **90.5%** (1,086/1,200) of
  all rows. Only 4 of the 10 target activities are ever predicted at all.

This means most of the rare "successes" are the model's near-constant output happening
to match the target by chance (target is m08 "standing shoulder extension" 1/10 of the
time, and "late" 1/3 of the time). This matches `RECOVERED_GCS_RESULT_AUDIT.md`'s
finding that the LoRA cell did not beat the frozen causal-TCN comparator
(`0.056667` vs `0.201667` teacher-forced joint-exact) -- the demo makes the same
conclusion visible and concrete rather than just citing the aggregate number.

## 6. What's NOT available locally (gaps, not fabricated)

- **pass@8 raw samples** (9,600 rows): only ever existed in the un-retained final GCS
  archive. The demo states this in the UI instead of omitting the section silently.
- **Full-fine-tune predictions**: checked all four full-FT run directories
  (`lfm1p2b-science-fullft-a100-001/002/003`, `lfm1p2b-fullft-calibration-a100-001`).
  None contain prediction rows. The three `-fullft-a100-00N` science runs have empty
  `PROGRESS_STREAM.jsonl` and `OPERATOR_FAILURE.json` recording
  `"streamed compact result was not verified before stop"` and
  `"remote credential deletion could not be verified"` -- these runs never reached a
  verified result, locally or (per their own lifecycle files) in GCS. The calibration
  run only has parameter-update-magnitude sanity checks
  (`result/CALIBRATION.json`, per-tensor `sample_update_l2`/`sample_update_rank`), no
  activity/phase predictions. **This is the one gap a completed full-FT run would fix**:
  a second model column to replay side-by-side with the LoRA column, to see whether
  full fine-tuning also collapses to a near-constant output or actually uses the body
  input.
- **Vicon data**: not used (Kinect positions are the admitted asset for this claim;
  Vicon angles/positions were not needed for this demo and were left untouched).

## 7. Key file paths (all relative to repo root)

```
experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/TASK_PLAN.json
experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/GRAMMAR.json
experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/results/lfm1p2b-science-lora8-a100-002/job.json
experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/results/lfm1p2b-science-lora8-a100-002/RECOVERED_GCS_RESULT_AUDIT.md
experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/results/lfm1p2b-science-lora8-a100-002/emergency-readonly-snapshots-2026-08-13/post-sealed-complete-1200.jsonl
experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/results/lfm1p2b-science-lora8-a100-002/emergency-readonly-snapshots-2026-08-13/interventions-partial-15964.jsonl
experiments/public-data-audit/ui-prmd-admission-2026-08-12/ADMISSION_CONTRACT.md
experiments/public-data-audit/ui-prmd-admission-2026-08-12/full_corpus/ADMISSION_CONTRACT.md
experiments/public-data-audit/ui-prmd-admission-2026-08-12/source/full-archives/Segmented_Movements.zip
experiments/public-data-audit/ui-prmd-admission-2026-08-12/source/full-archives/Incorrect_Segmented_Movements.zip
experiments/public-data-audit/ui-prmd-admission-2026-08-12/source/Animation.m
```
