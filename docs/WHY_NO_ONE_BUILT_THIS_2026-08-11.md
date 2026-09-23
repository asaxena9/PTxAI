# Why Hasn't Anyone Built the Pose Decoder? — An Adversarial Audit

**Date:** 2026-08-11
**Question (founder):** "Why do people not solve this problem? Is it that the body-pose models and the frameworks they come with are just too good that there is no need for a decoder? Genuinely answer this. Try to break the argument from every level."
**Method:** steelman "no decoder needed" at each level, then break it — or concede it — against the evidence base assembled 2026-08-10/11 (six research agents + in-repo measurements). Explanations of *why a field didn't do something* are sociology and cannot be fully verified; every level below is therefore anchored to at least one checkable fact, and the interpretive parts are labeled INTERPRETATION.

**Verified premise — the gap is real, not our ignorance:** cross-dataset skeleton zero-shot does not exist in the searched literature; the top zero-shot skeleton systems are transductive (need the whole test batch — cannot stream); the only open-vocabulary *online* detector (OV-OAD, NeurIPS'24) is RGB-only; no pose-native streaming OAD/TAS literature exists; no benchmark anywhere reports decision-latency or false-events/hour for pose; even "LLM reads raw pose coordinates" has never been published. [All AGENT-VERIFIED absences, 2026-08-10.]

---

## Level 1 — The founder's hypothesis: "the existing stack is too good, so no one needs a decoder"

**Steelman.** Pose estimation is commoditized (MediaPipe/RTMPose-class, real-time, on-device, free). On top of keypoints, small skeleton classifiers (≈1–5M params) sit at the top of every closed-set benchmark — a 3M-param HD-GCN beats Gemini on gesture recognition [SOURCED, Käs 2025]. For any *fixed* application — fall detection, rep counting, a 20-exercise library — the winning engineering is: enumerate the classes, train the small classifier, hand-code the FSM and thresholds. Commercial motion-PT products observably ship exactly this shape: fixed exercise libraries. Products enumerate their features; enumerated features don't need open vocabulary. So there is no commercial forcing function for a decoder.

**Where it holds — concede honestly.** For the head of the distribution (the 20–50 most common exercises/activities), this is true, and it is why *industry* never built a decoder. Our own landscape evidence agrees: discriminative models win on fixed label sets, which is exactly why the first-proof design only claims decoder value on open compositional outputs.

**Where it breaks.**
1. **The stack is "too good" only at the two bottom layers — and it ends at a label ID.** Estimation (pixels→keypoints) is excellent; classification (keypoints→one of K) is excellent. Everything a decoder would do — bind the label to an instruction, track protocol state, compose unseen combinations, emit typed events with pointers into a text protocol, abstain, explain — is **hand-written application code today, rebuilt per app and per exercise library**. There is no pose equivalent of "prompt the model." The founder's observation is true about the layers that exist and silent about the layer that doesn't. The gap *is* the missing third layer.
2. **The head players themselves are reaching for language models — which is near-proof the need exists.** BioCoach bolted a LLaMA-2-**7B** onto pose tokens precisely because hand-set thresholds don't scale to feedback (and still only covers 23 fixed exercises with curated reference trajectories) [SOURCED, primary-read]. Qualcomm's QEVD/Stream-VLM line builds streaming fitness feedback with a VLM [SOURCED]. If classifiers+rules were sufficient, neither system would exist. What's absent is not the *need* for a decoder — it's the **compact, pose-native, protocol-general** version of one.
3. **The long tail is structurally unserved.** Thousands of PT protocols, per-clinic variation, arbitrary industrial procedures — closed-set engineering scales per-task, so the tail waits forever. "Good enough" describes the head, not the distribution.

**Verdict on Level 1: partially true — it explains industry absence, and only industry absence.**

---## Level 2 — The inverse hypothesis: pose models are not too good, they're too *unreliable*

**Steelman.** In-the-wild pose is noisy — our own measurement: MediaPipe elbow error p95 = 43.1° [MEASURED, in-repo]. Occlusion, clothing, camera angle. Serious teams conclude the ceiling is too low and go RGB end-to-end; FormCoach explicitly argues pose-only input is insufficient [SOURCED]. Understanding built on pose inherits every estimator failure.

**Where it holds.** This is a real, measured risk and it stays in the plan as a design constraint: evaluate on *estimated* pose from day 1 (never only mocap), make `UNSURE` a first-class output, and keep pose+X input as an admissible variant.

**Where it breaks.** Event-level semantics (which step, started/finished/skipped) is a far coarser target than joint-angle-accurate form scoring; estimators crossed "good enough for event semantics" years before they'll cross "good enough for goniometry." BioCoach — whose entire product is form feedback — still chose pose tokens as its interface [SOURCED], which is the strongest available counter-testimony to "pose is too noisy to build on."

**Verdict: alive as a risk, not as an explanation for total absence — otherwise skeleton classification wouldn't be a thriving benchmark field.**

---

## Level 3 — Data: the strongest single explanation

**The break, stated plainly: a pose decoder was not buildable until roughly 2024, because the paired data did not exist and cannot be web-harvested.** Vision-language exploded because image-text pairs occur naturally on the web at billions-scale (LAION-class). **Pose-text pairs never co-occur anywhere.** The pose side must be manufactured by running estimators over video; the text side exists only in curated annotation efforts. The entire historical inventory: HumanML3D = 14,616 clips (2022); BABEL = ~43.5 h (2021); KIT = 3,911 clips [all SOURCED]. The ~500K–1M pair bar that every successful sub-1B adapter recipe needed (LLaVA-class) [SOURCED] became reachable **only** when Ego-Exo4D shipped 1,286 h with native 3D body pose and 432K atomic descriptions (released 2024, license-gated) [SOURCED] — and the field mostly uses it for its RGB.

**Compounding twist [INTERPRETATION anchored in SOURCED facts]:** the pose-text data that did exist was built pointing the *other way* — text→motion **generation** for animation/graphics (the paying customer). MotionGPT's evaluation is dominated by generation; motion→text is a side task [SOURCED, deep-dive]. The understanding direction got the leftovers of a corpus designed for synthesis.

---

## Level 4 — The task was ill-posed until ~2023, and undeployable until ~2024

- "Verify what a person is doing **against an arbitrary natural-language protocol**" requires a model that can *read the protocol*. Before instruction-following LMs, this task could not be formulated as a learning problem at all — only as per-protocol engineering. The reframe that makes it well-posed (verification against NL protocol; next-step = protocol-state lookup) is cheap **now** and impossible **then**.
- The deployment point — a compact decoder at the edge — required instruction-capable sub-1B LMs, which are a 2024–2026 phenomenon (SmolLM2, Qwen3-0.6B, Gemma 270M, LFM2; 350M-class measured at 88 tok/s on an iPhone 13 mini [SOURCED, third-party]). Before that the menu was: tiny classifier (deployable) or 7B decoder (server). The product point in between did not exist.

**The full ingredient stack — (a) commodity on-device pose, (b) instruction-following sub-1B LMs, (c) a ≥500K pose-text corpus, (d) streaming-eval formats — completed within roughly the last 18 months.** Gaps whose ingredient stack just completed are exactly the gaps that are still empty when a solo researcher arrives.

---

## Level 5 — Benchmark vacuum and the offline bias [INTERPRETATION anchored in SOURCED facts]

Research fields move where leaderboards point. Skeleton understanding's leaderboards are closed-set NTU splits; its zero-shot corner is in measurable protocol chaos (three incompatible split conventions; the same DeViSE baseline reported at 60.72 and 77.61) [SOURCED]. There is **no benchmark** for streaming pose understanding and no venue pressure to report latency or false events [VERIFIED absence]. And streaming lags offline *everywhere* in vision, not just in pose — online temporal segmentation was only seriously quantified in 2024, with a brutal gap (Breakfast F1@50: 30.5 online vs 75.9 offline) [SOURCED, OnlineTAS]. Pose isn't uniquely neglected; it's a smaller field standing further back in the same queue. No leaderboard → no papers → no field. Defining the benchmark is therefore not overhead — it is one of the contributions.

---

## Level 6 — Attention, ideology, and incumbent anti-incentive [INTERPRETATION]

- **The bitter-lesson bet:** the field's center of gravity holds that learned RGB features beat hand-chosen intermediate representations, so prestige flows to video-LLMs; a skeleton is ideologically a "hand-crafted bottleneck." This is partially rational and is exactly why our thesis survives only as an **efficiency-frontier claim** (the bitter lesson has no force at a fixed on-device compute budget — engineered representations historically persist precisely there, as mel filterbanks did in speech for decades).
- **Incumbent anti-incentive:** the companies that own large private motion-PT datasets have moats made of curated exercise libraries plus clinical validation. A general protocol-reading decoder would commoditize exactly that moat. The actors best positioned to build the general layer are structurally disincentivized to.
- **Domain friction:** the domains that need continuous body understanding (rehab, industrial safety, elder care) are regulated, privacy-gated, and not where ML researchers sit — producing the observable pattern of tiny one-off clinical datasets (REHAB24-6: 1.71 h) [MEASURED inventory].
- **File-drawer possibility:** quiet failed attempts can't be ruled out. Partial evidence against: the adjacent published attempts (MotionGPT-family motion captioning) *work* at clip level, no negative results are on record, and the trivial feasibility probe (LLM-on-raw-pose) was never even run [VERIFIED] — a field that had tried and failed en masse would have left more residue.

---

## Level 7 — The version of "no decoder needed" that survives and must be respected

Open-vocabulary matching alone does not require a generative decoder — a dual-encoder retriever plus an FSM over the parsed protocol graph might cover the task. **This is not a reason nobody tried; it is a hypothesis nobody tested — and it is pre-registered as our strongest null** (task-graph arm).

**Amended 2026-08-11 (founder objection, accepted):** the decoder is not required to beat that pipeline in the pipeline's own closed currency — that framing would make generative capacity pay rent only where it can't show. Decision rule v2 (`DEMO_FIRST_SUB1B_PLAN_2026-08-11.md` §3.1): **parity** on typed events at matched compute, plus **measured superiority** on surfaces the pipeline cannot represent (evidence-grounded explanation, off-graph recovery, stateful QA) — judged against the sharpest version of the null: **FSM + a small LM verbalizer bolted on top**. If even that hybrid matches the end-to-end decoder everywhere, the intelligence is modular, the field's implicit judgment was right, and we will have shown it with measurement. Either outcome resolves this document's question empirically.

---

## Adjudication

**"The existing stack is too good" is the right explanation only at the closed-set product layer, and even there the accurate phrase is "good enough for the head of the demand distribution."** It cannot explain the research absence, because the frameworks end at a label ID — the understanding layer above keypoints has never existed as a model anywhere, and the head players (BioCoach at 7B, Qualcomm with VLMs) are visibly reaching for it.

The research absence is over-determined by four structural causes, ranked by evidential strength:
1. **Data impossibility until ~2024** — pose-text pairs are not web-harvestable, and the first corpus crossing the adapter-recipe threshold is ~2 years old and gated [SOURCED].
2. **Task ill-posedness until instruction-following LMs + no deployable sub-1B host until 2024–25** [SOURCED].
3. **Benchmark vacuum + the field-wide offline bias** [SOURCED anchors, interpretive weight].
4. **Attention gradient toward RGB + incumbent anti-incentive** [INTERPRETATION].

**What survives against us after breaking everything breakable:** the closed-set sufficiency of the head (conceded — we don't compete there); the retrieval+FSM null (pre-registered, will be measured); the RGB endgame under collapsing compute costs (standing risk; the wedge is efficiency + privacy, and it is a wedge, not a wall); and estimator noise (measured, carried as a design constraint).

**Timing corroboration:** if this analysis is right — real gap, ingredients newly complete — the observable prediction is *multiple groups converging now*. That is exactly what March 2026 shows (BioCoach, LaDy, CAKE, the first streaming-SLT attempt, all weeks old) [SOURCED]. The absence was never evidence of worthlessness; it was a clock that only recently started, and it is now audibly ticking.
