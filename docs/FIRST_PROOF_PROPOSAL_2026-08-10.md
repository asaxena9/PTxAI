# First proof: what an independent researcher can actually prove here

Date: 2026-08-10
Author: Claude (Fable), principal synthesis; external facts verified by four research agents on 2026-08-10 (zero-shot pose-language prior art; procedural datasets with body pose; structured-state baselines + sign-language streaming; procedure-understanding literature). Evidence classes per `AGENTS.md`. Numbers from agent reports carry their verification status: [SOURCED] = read at a primary source; [UNVERIFIED] = secondary-snippet only, re-check before citing in a paper. Gap claims are evidence-based absences after systematic multi-angle search, not proofs of non-existence.

---

## 0. The question you are asking, made precise

Your words: pose (maybe + signals) → a decoder fine-tuned to understand those cases → continuously verify what the person is doing → what the next step should be → real edge intelligence.

Formalized: given continuous body observations `O_≤t` and a task/protocol `Q` stated in natural language, emit a **streaming verification state** `V_t` (which step of Q is happening; started/completed/skipped/deviating; with timestamps and confidence) and **protocol-relative guidance** `G_t` (what Q says comes next), within edge budgets.

That decomposes into five separately provable claims:

- **C1 — semantics:** pose streams carry language-alignable activity meaning at compact scale.
- **C2 — state:** step state is trackable *online* with bounded decision latency and false-event rate.
- **C3 — composition:** conditioning transfers to **protocols never seen in training**, composed of seen primitive actions. ← this is the decoder's actual job, and the thesis's live claim
- **C4 — guidance:** once C2+C3 hold, "next step" is a lookup in the protocol state — **not a learned recommender** (see Q2 below).
- **C5 — edge:** the full loop runs on-device. Designed-for now, measured after a capability exists.

**The first proof is C1. The decisive proof is C3 (with C2's metrics). C4 falls out free. C5 waits.**

---

## 1. The questions you are not asking yet (each one decides success)

1. **The output-space question.** A decoder beats a classifier only when the output space is open and compositional. On fixed label sets, the external evidence says discriminative wins (+2.5%, 3× faster on action understanding [SOURCED, arXiv 2603.02546]; LLM backbones removable from continuous-signal models without loss [SOURCED, arXiv 2406.16964]). So the first task must be open **by construction** — protocols in natural language, unseen at test time. Otherwise you are re-running the experiment your own repo already failed.
2. **The supervision-provenance question.** "What should they do next" has no ground truth unless (a) the protocol defines it — then it's derivable lookup, not learning; or (b) outcomes are logged — you have none. Learned recommendation in v1 is an unfalsifiable trap. The dataset agents confirmed this empirically: next-step ground truth exists *naturally* in recipe/manual datasets (CaptainCook4D via WikiHow steps, IKEA ASM via real manuals, Assembly101, IndustReal) and must be *labeled* elsewhere. Verification is evaluable; recommendation quality is counterfactual. Build the verifier; emit guidance as protocol-state lookup.
3. **The real-null question.** Your competitor is not a softmax classifier. It is **keystep-classifier + learned symbolic task graph**: Differentiable Task Graph Learning (NeurIPS 2024 spotlight, arXiv 2406.01486 + 2502.17753, code public) — SOTA online mistake detection with double-digit gains (Assembly101-O +19.8%, CaptainCook4D +14.5–16.7% deltas [SOURCED abstracts; absolute tables unverified]). Your decoder must beat *that*, configured from the same NL protocol, at matched compute — or lose honestly to it.
4. **The generalization-axis question.** Pick exactly one axis for the claim: **unseen protocols composed of seen primitives**. Not new-person SOTA (saturated, big-lab territory), not closed-set accuracy (classifiers own it: NTU-60 ≈ 92–93% at 2–3M params [SOURCED]).
5. **The observability question.** From pose alone: activity identity — yes; protocol conformance — partially; effort/pain/load/force — no. `UNSURE` must be a first-class output. Your repo's own admission discipline already encodes this; keep it.
6. **The streaming-metrics question.** Decision latency, false events per hour, boundary tolerance: *no benchmark in any adjacent literature reports these for pose* — even sign-language streaming papers use ad-hoc wall-clock, never Average Lagging [agent-verified absence]. Defining these metrics is itself a citable contribution.
7. **The calibration question.** In any deployment, a wrong "verified ✓" is worse than "unsure." Report calibration (ECE, risk–coverage curves), not just accuracy. Nobody in the surveyed literature does; cheap differentiation.
8. **The identity/privacy question.** Pose is not anonymous — your own audit re-identified the participant from skeleton proportions in 64/64 recordings [MEASURED]. Claim *data minimization vs RGB*, never anonymity.
9. **The deployment-shift question.** Mocap-trained ≠ phone-deployable: your repo measured MediaPipe error (2.71% bbox-diagonal mean; elbow p95 43.1°) [MEASURED]. Evaluate on estimated pose from day one, mocap as ground truth only.
10. **The protocol-chaos question.** The zero-shot skeleton field runs three incompatible evaluation protocols on identically-named splits; the same baseline is reported at 60.72% and 77.61% by two papers [SOURCED disagreement]. Pre-register your splits and protocol publicly on day one — both scientific hygiene and a wedge.
11. **The indie-moat question.** What survives a big lab replicating you: the task formulation, the benchmark and metrics, the compact/efficiency point, released weights, and speed. Not scale. Design every deliverable so that being copied *cites you*.

---

## 2. What you can prove — because it is verifiably unproven today

Four independent gap-checks (2026-08-10), each an evidence-based absence after systematic search:

| Gap | Evidence |
|---|---|
| No **open-vocabulary skeleton recognition in streaming mode** at compact scale | Top zero-shot skeleton systems are transductive by their own papers' admission (PGFA 93.17%, Skeleton-Cache 89.41% — need the whole test batch; cannot classify one incoming clip) [SOURCED]. Inductive frontier: Neuron (CVPR'25) 86.9/62.7, InfoCPL 85.91/53.32 on NTU-60 55/5 [SOURCED]. The only open-vocab *online* detector, OV-OAD (NeurIPS'24), is RGB-only [SOURCED]. |
| No **cross-dataset zero-shot skeleton transfer** anywhere | Every "cross-dataset" claim in the literature means per-dataset splits repeated, never train-A/test-B [agent-verified absence]. |
| No **pose + NL-protocol + streaming + structured-events on unseen procedures** system | Closest each miss ≥1 axis: PREGO/TI-PREGO (streaming, but closed-vocab RGB); zero-shot VLM mistake detection (open, beats supervised on EgoPER +4.4 EDA [SOURCED], streaming unconfirmed, RGB); ProVideLLM (ICCV'25, genuinely streaming 10 FPS/2 GB [SOURCED], RGB, unseen-protocol unconfirmed); JARVIS (AR, user-study only). |
| No **pose-only + streaming + natural-language output + quality** even in sign language | Offline skeleton-only SLT ≈ BLEU-4 27.8–28.5 (SIGNET, ~1.5M-param encoder) [SOURCED]; the one genuinely streaming system reports BLEU 15.9 at ~1.35 s latency [SOURCED]; no SLT paper reports Average Lagging [agent-verified absence]. |

And the field you'd enter is unsaturated: Ego-Exo4D keystep baseline ≈ 25.8% top-1 [UNVERIFIED], best reported ≈ 54.7–56.7% [UNVERIFIED]; mistake-detection F1s sit in the 40s–50s (CaptainCook4D 53.87 [SOURCED via snippet], Assembly101-O ≈ 45.6–48.4 [UNVERIFIED]); Ego-Exo4D's own task-graph leaderboard literally says "Baseline — Coming Soon" [SOURCED].

---

## 3. Proof-0 (weeks 1–6): the open-vocabulary streaming capability card

**Claim to prove (C1):** a ≤25M pose encoder, contrastively aligned to a frozen text encoder, does open-vocabulary activity recognition **frame-causally, in a stream**, including on classes and datasets never seen in training.

- **Train:** streaming pose windows ↔ step/action text, on BABEL — verified as the only drop-in substrate (dense, contiguous, multi-label, precisely time-aligned frame labels; ~43.5 h AMASS; 250+ categories; academic registration license) [SOURCED]. Add FLAG3D's instruction-language exercise data as a second training domain (form-gated license) [SOURCED]. HumanML3D is clip-level and Motion-X's "frame-level" labels are static pose descriptions — neither substitutes [SOURCED].
- **Evaluate:** (a) held-out-class zero-shot (pre-registered split, protocol published — see Q10); (b) **cross-dataset zero-shot** — train BABEL, test FLAG3D and REHAB24-6 exercise names untouched (the axis nobody has); (c) **streaming metrics** — decision latency, false events/hour, boundary tolerance (the metrics nobody reports); (d) on estimated pose, not just mocap (Q9).
- **Baselines:** closed-set classifier (cannot do held-out classes — that's the point, shown explicitly); the strongest inductive zero-shot numbers as context (Neuron/InfoCPL, non-streaming); frontier-LLM-on-rendered-pose (see week-1 experiment); trivial priors.
- **Components to reuse, not rebuild:** TMR (MIT-licensed text-motion embedding code) and MotionCLIP as inits/baselines [SOURCED]; your MotionBERT-Lite transfer result (pretrained > random ×6 on lifting; pooling destroys it [MEASURED]) directly motivates the token-level (non-pooled) interface.
- **What Proof-0 does and does not prove:** it proves the substrate (C1 + streaming feasibility) and stakes the benchmark. It does **not** prove decoder value — it is a dual encoder, and saying so plainly is what makes Proof-1 meaningful.

**Week-1 quick win (verified unrun in the literature):** "frontier LLM on raw pose coordinates" — serialize BABEL windows as text, prompt GPT/Gemini/local models for zero-shot action labels, measure. No published version of this exists [agent-verified absence; closest: Käs 2025, where Gemini "struggles" on gestures vs HD-GCN, input encoding unconfirmed]. Two or three days, ~$50 of API calls, and you own the missing feasibility anchor — a citable micro-result regardless of outcome, and your first external number.

## 4. Proof-1 (weeks 6–16): protocol-conditioned streaming step verification — where the decoder earns its job or loses it

**Claim to test (C3, the thesis):** given an *unseen* natural-language protocol composed of seen primitives, a compact decoder tracks protocol state online — emitting `STEP_START / STEP_DONE / STEP_SKIPPED / DEVIATION / UNSURE` events with timestamps and protocol-line pointers — better than the strongest structured-state pipeline at matched compute.

- **Data (verified and ranked):**
  1. **Ego-Exo4D** — the foundation: the only dataset with native 3D body pose + a 278-class keystep taxonomy explicitly shared across scenarios (the seen-primitive/unseen-protocol structure by construction); 1,286 h / 740 participants [SOURCED, discrepant larger figures traced to synthesis artifacts]; free signed license (~48 h). No native mistake labels — plan proficiency-as-proxy or light error injection.
  2. **CaptainCook4D** — the mistakes cell: 7-category error taxonomy, deliberate-deviation sessions, next-step ground truth = recipe lookup, Apache-2.0. Body pose must be estimated from the GoPro RGB (validate the estimator first — Q9).
  3. **IKEA ASM** — the same-day-start cell: direct CC BY-NC download, real IKEA manuals as protocols, three calibrated Kinect views + depth → derivable 3D pose.
  4. **EgoExo-Fitness** — the PT-domain cell: 86 ordered action-sequences (3–6 steps, annotated boundaries and order) = exercise protocols; pose derivability and scale need direct verification before committing [flagged].
  5. REHAB24-6 stays the in-repo PT eval cell; ATTACH (native Azure-Kinect 3D body pose, assembly) worth a direct check.
- **Splits:** hold out entire recipes / furniture types / routines; primitives shared. Protocol-only and cadence/ordinal priors run as mandatory shortcut nulls — port the D3Q shortcut-audit and matched-intervention machinery wholesale (it demonstrably catches real leaks [MEASURED]).
- **Arms (matched compute, ≥3 seeds, pre-registered):**
  1. Proof-0 encoder + compact causal decoder (protocol in context, typed event head);
  2. **the strongest null:** Differentiable Task Graph Learning with the same embeddings, graph parsed from the same NL protocol (their code is public);
  3. classifier + FSM (closed-vocab ceiling);
  4. shortcut nulls (protocol-only, cadence, duration priors).
- **Pre-registered decision rule:** the decoder earns its existence iff it beats arm 2 on unseen-protocol event F1 *and* streaming latency at matched compute, with positive grounding selectivity (relevant-vs-irrelevant intervention, your own eval design). If arm 2 wins: the product architecture is retrieval + task-graph, you publish that result plus the benchmark plus the Proof-0 card — still three real contributions. **Either outcome is "something."**

## 5. Compute (measured-anchored; no credits arithmetic)

From your repo's own telemetry: 4.40 s/step at 358M/batch-1/dense-interface on G4 → ≈3,200 positions/s [MEASURED]. Proof-0 and Proof-1 live at ≤25–50M with pooled/tokenized interfaces — per the corrected budget table in `BODY_MOTION_MODALITY_ROADMAP_2026-08-10.md`, every relevant cell is under ~5 G4-hours per pass even at unoptimized measured throughput; both proofs with seeds and ablations plausibly fit in **tens of G4-hours** [DERIVED from measured anchor]. First GPU action remains the 30–60 min calibration probe that converts the assumptions to measurements. Balance/rates stay UNKNOWN to automation — track hours, not credits.

## 6. Kill criteria (pre-committed)

- **Proof-0 dies** if held-out-class zero-shot ≤ chance + ε or cross-dataset transfer collapses to near-random → the compact pose-language substrate fails at this scale; pivot to supervised-streaming card + report. (The card is still a deliverable.)
- **Proof-1's decoder arm dies** by the decision rule above — and the loss is itself publishable against the named NeurIPS-2024 null.
- **Task-construction death:** if protocol-only priors match the best arm, the benchmark is leaked — rebuild before claiming anything (your D3Q audits showed exactly how).

## 7. Deliverables and rough sequence

- **Weeks 1–2:** LLM-on-pose-coordinates experiment; dataset license applications (Ego-Exo4D, BABEL, FLAG3D — the ~48 h/form-gated ones first); calibration probe; pre-registered splits/metrics doc published (GitHub).
- **Weeks 2–6:** Proof-0 training + streaming eval harness; capability card + weights + benchmark v0 release.
- **Weeks 6–16:** Proof-1 arms on Ego-Exo4D + one mistakes cell + the PT cell; decision; write-up (workshop-length first; the streaming-metrics + benchmark + verdict is a full paper if Proof-1 resolves either way).
- Standing rule from the holistic review: **every cycle ships at least one externally comparable number.** Proof-0's zero-shot numbers satisfy this from week ~4 onward; the Ego-Exo4D task-graph leaderboard ("Baseline — Coming Soon") is an open door to be early on.

## 8. What this proves in your language

If Proof-0 lands: *pose is a language-alignable native modality at edge scale, verified in a stream* — the substrate claim behind "AI at the edge that understands body movement," with the first streaming benchmark and metrics as your flag.
If Proof-1's decoder wins: *instruction-conditioned physical verification is a real decoder capability* — the SLM-frontier step you originally set out to show, proven on public data against the strongest published null.
If Proof-1's decoder loses: you have measured, on a benchmark you now own, exactly where structured-state pipelines end and decoders begin — which is the knowledge a company in this space would pay for, and which almost nobody else will have bothered to establish honestly.

## 9. Residual unverified items (do not cite without re-checking)

Absolute results tables for Differentiable Task Graph Learning; Ego-Exo4D keystep absolute SOTA (25.82%/54.69–56.74% figures are secondary); PREGO/TI-PREGO exact F1s; SIGNET/RVLF numbers were primary-fetched but SimulSLT/EvSign/Uni-Sign details were not; EgoExo-Fitness pose derivability and scale; PDPP/SCHEMA planning numbers; whether ProVideLLM handles unseen protocols. All flagged in the agent reports retained in the session transcript.
