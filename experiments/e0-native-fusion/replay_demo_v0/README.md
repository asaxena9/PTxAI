# Replay demo v0

The first visible artifact for the body-native decoder work: a page where you watch a
skeleton animation of a real UI-PRMD physical-therapy repetition while the model's
recorded typed-sentence prediction for that repetition sits next to it, marked
CORRECT or WRONG against the ground truth.

This **replays recorded evaluation rows**. No model runs when you open the page, and
no model ran when `replay.py` built it -- `replay.py` only reads already-written JSONL
prediction files and the raw UI-PRMD archives. Plain Python 3 + numpy, no torch.

## Regenerate

```
cd experiments/e0-native-fusion/replay_demo_v0
python3 replay.py
```

Writes `replay.html` next to the script (self-contained, ~275 KiB, no external CDNs or
network calls) and prints, to stdout: which episodes were selected and whether each
built cleanly or was skipped with a reason, a payload-validation pass (episode count,
total frames, "all coordinates finite/in-range"), and the measured global stats
(joint-exact / activity-correct / phase-correct rates, predicted-activity and
predicted-phase distributions over all 1,200 held-person rows). Requires read access to
this repo's `experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/` and
`experiments/public-data-audit/ui-prmd-admission-2026-08-12/` trees; writes nothing
outside this directory.

To change which episodes are shown, edit the documented rule in
`replay.py::select_episodes` (it is a plain function over `TASK_PLAN.json` + the
prediction rows, not a hardcoded list) and re-run.

## What this demo shows

- A 2D front-view stick-figure animation (canvas, 22-joint Kinect topology, standard
  UI-PRMD bone connectivity) of one full recorded repetition at a time, play/pause,
  0.25x-2x speed, and a scrubber.
- The three within-repetition points the model was actually asked about for that
  repetition (early/middle/late, each a real 12-frame / 0.4s window at the recorded
  30 Hz nominal rate -- shaded on the timeline), each showing the ground-truth
  sentence, the model's generated sentence, and a CORRECT/WRONG badge from the
  recorded `joint_exact` field (both activity and phase right).
- An episode picker covering all 10 UI-PRMD movements, both held-out subjects (s09,
  s10), and both collection branches, chosen by a documented mechanical rule (see
  `INVENTORY.md` part 4) -- 8 of the 10 are clean failures, 2 have one correct probe
  out of three. That mix is what the rule produced, not a curated split.
- A stats banner with the measured rates over **all 1,200** held-person rows (not just
  the 10 shown episodes): 5.7% joint-exact, 16.4% activity-correct, 33.3%
  phase-correct, and the finding that the model predicts "standing shoulder
  extension" 90.5% of the time and "late" 100.0% of the time regardless of input.

## What this demo does NOT show

- **A capability.** This is one LoRA adaptation cell (rank 8) of one 1.2B model on one
  small dataset, evidence class **MEASURED-PILOT**. `RECOVERED_GCS_RESULT_AUDIT.md`'s
  scientific decision for this run is **FAIL_OR_INCONCLUSIVE_LOCAL_H1**: the model
  learned some selective body-dependent signal (identity beats query-only and
  pose-zero controls) but did not beat a frozen causal-TCN comparator on teacher-forced
  joint-exact (0.057 vs 0.202). Nothing here should be read as "the model can do PT
  movement recognition."
- **Broad participant coverage.** UI-PRMD has 10 healthy university participants total.
  The two subjects in this demo (s09, s10) are the entire held-person test set --
  N=2. See `experiments/public-data-audit/ui-prmd-admission-2026-08-12/ADMISSION_CONTRACT.md`
  for the full construct-validity limits (no clinical, diagnostic, or correction claim).
- **Most of the model.** LoRA rank 8 trained 1,936,768 of 1,172,277,376 active
  parameters -- **0.165214%**. A negative or weak result here bounds the adapter, not
  necessarily the 1.2B backbone (see `RECOVERED_GCS_RESULT_AUDIT.md`'s own framing:
  "adapter-bottleneck ambiguous"). The planned full-fine-tune comparison run never
  produced a verified result (`INVENTORY.md` part 6) -- it is the natural next thing
  that would sharpen this picture, not a re-run of this same LoRA cell.
- **pass@8.** The sampled 8-generations-per-task outcomes exist only in a GCS archive
  that was audited but never retained in this repo. The page says so in place rather
  than hiding the row.
- **Anatomically validated motion.** The skeleton reconstruction ports the UI-PRMD
  authors' own `source/Animation.m` position+angle composition verbatim, including
  its own documented caveat that lower-body joints are "not entirely correct" for
  m03/m04/m05. Treat the animation as an illustrative replay of the recorded numbers,
  not a validated biomechanical reconstruction.
- **Natural patient error or correction.** The `acted_nonoptimal` branch shown for some
  episodes is healthy participants arbitrarily acting out nonoptimal movement on
  request, per the dataset's own admission contract -- not a labeled clinical error,
  and not paired with the `ordinary` branch as a before/after correction.

## Provenance

Predictions: `lfm1p2b-science-lora8-002` run, audited in
`experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/results/lfm1p2b-science-lora8-a100-002/RECOVERED_GCS_RESULT_AUDIT.md`.
Task/window plan: `experiments/autoresearch/body_native_ui_prmd_semantic_canary_v0/TASK_PLAN.json`.
Raw motion: official UI-PRMD archives under
`experiments/public-data-audit/ui-prmd-admission-2026-08-12/source/full-archives/`,
admitted for this use in that lane's `ADMISSION_CONTRACT.md` /
`full_corpus/ADMISSION_CONTRACT.md`. Full detail on what was found, what was verified,
and what's missing: `INVENTORY.md` in this directory.
