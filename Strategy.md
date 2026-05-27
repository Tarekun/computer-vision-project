# Strategy — Coin Detection, Classification, Counting

Working document for *Image Processing and Computer Vision — Assignment Module #1*.
This file is not the deliverable: the deliverable is the notebook. This file is the
**spine** that justifies what ends up in the notebook. It is read primarily by the
author; it lives in the shared repo so a collaborator working on a parallel module
can take a look without needing the surrounding conversation.

**Convention — STABLE / LIVE.** The document is split in two halves:

- The **STABLE** half (§1–§3) fixes things that should rarely change: the constraints
  set by the brief, the evaluation criteria, and the pipeline skeleton. Treat edits
  here as deliberate revisions.
- The **LIVE** half (§4–§7) is append-mostly. Open decisions get closed and move to
  the decision log; the journal grows chronologically; milestones get checked off.

The boundary between the two is marked explicitly below. Tactical day-to-day TODOs
live in the session task tracker, *not* here — §7 holds only project-level milestones.

---

## === STABLE ===

## 1. Non-negotiable constraints

- **Traditional CV only.** No deep learning, no end-to-end learned models.
  Lightweight hand-crafted classifiers (e.g., a nearest-mean on hand-designed
  features) are allowed but must be motivated.
- **Justify every decision.** Every choice in the notebook must cite the relevant
  chapter of `IPCV_Part1_Summary_v2.pdf`.
- **Output format is fixed.** Per image: `Coin N {value: X.XXX}` and
  `Partial Amount {value: …}`. Across the dataset: `Total Amount: … €`.

## 2. Evaluation criteria

Taken verbatim from the brief, in the priority order the professor numbered them:

1. **Clarity & conciseness** — readable code, comment every important step.
2. **Procedural correctness** — many sound approaches exist; *justify every decision*.
3. **Results correctness** — *a thoroughly justified procedure with fewer solved
   instances is valued **more** than a poorly justified one that solves more*.

Operational consequence: the decision log (§5) is part of the work, not an extra.
Each justification cell in the notebook should be traceable to a closed entry in §5.

## 3. Pipeline decomposition

The assignment naturally factors into five blocks. Each block's *role* is fixed; the
*algorithm* implementing it is an open decision tracked in §4.

| # | Block | Inputs → Outputs | Theory anchor (summary §) |
|---|---|---|---|
| **A** | Pre-processing | RGB image → smoothed image (grayscale where needed) | §3 *Image Filtering* |
| **B** | Detection | smoothed image → list of `(cx, cy, r)` per coin | §4 *Edge Detection* + §6.3 *The Hough Transform* |
| **C** | Classification | each `(cx, cy, r)` → denomination ∈ {1¢, 2¢, 5¢, 10¢, 20¢, 50¢, 1€, 2€} | §6.1 *Template matching* and/or §5 *Local Invariant Features* |
| **D** | Aggregation | list of denominations per image → `Partial Amount`; across images → `Total Amount` | — (arithmetic + output formatting) |
| **E** | Calibration on `reference_set` | reference images → parameters used by A, B, C | — (methodology; drives the closures in §4) |

The "umbrella" chapter is **§6 Instance-level Object Detection**
(TL;DR p.33 bullet 6: *"limited variability ⇒ classical methods"*). Block E is not
a runtime block: it is the activity that, in absence of a learned model, derives the
constants and references that A–C consume from the 8 labelled reference images.

---

## === LIVE ===

## 4. Open decisions queue

Each entry describes a question that is not yet answered. When closed, the entry is
**moved to §5 Decision log** and replaced here by a one-line reference.

### 4.1 — Primary classification cue

- **Question.** Which feature, or combination of features, drives the 8-class
  decision in block C?
- **Evidence (M0/M1, 2026-05-26).** Both "obvious" absolute cues are dead cross-set:
  - *Absolute colour / family*: the `target_set` carries a strong blue specular cast
    (coin hue ≈ 100–118 vs reference 5–31; `b*` flips sign). It is **not** a global
    white balance (backgrounds are ≈ neutral) and is **not invertible** by a filter —
    it is specular reflection of the environment on the metal. Copper/gold do not
    separate, not even by eye. → options (b) eliminated (DEC-3).
  - *SIFT vs reference* (d): the ref×ref good-match matrix is **not** diagonal-dominant
    (false cross-matches 20–30), target predictions collapse onto the high-keypoint
    references, and only value-side-up coins could ever match (≈ 50% recall). Weak;
    parked unless rescued by RANSAC geometric verification.
  - *Relative radius within an image* is metric (DEC-1) and survives everything.
  - *Bi-metallic structure* is detectable (§4.4).
- **Chosen direction.** (f) **geometry-first joint inference**: per image, detect all
  coins, fit a single scale `s` mapping their radii onto the known diameter grid
  {16.25, 18.75, 19.75, 21.25, 22.25, 23.25, 24.25, 25.75}; anchor `s` with a detected
  bi-metallic coin (§4.4) or the constellation's self-consistency. Within-family size
  gaps are wide (≥ 2 mm), so the hard ~1 mm collisions (all cross-family) are the
  residual ambiguity, resolved by the bi-metallic flag or accepted as the limit.
- **Criterion to close.** Validate the scale-grid fit on target images with a
  bi-metallic anchor (M3); report per-image confidence.
- **Status.** 🟠 DIRECTION SET (geometry-first) — pending M2/M3 validation.

### 4.2 — Scale strategy

**CLOSED → DEC-1.** Absolute radius is not metric: `1€` is detected larger (r = 98 px)
than `2€` (r = 82 px) across reference shots, although physically smaller ⇒ the
camera-to-coin distance varies between shots. Coins *within one image* share distance
and pose, so their radius **ratios** are metric. Use intra-image relative radius +
a per-image scale anchor (§4.1 f).

### 4.3 — Background scale anchor

**CLOSED → DEC-2.** No known-size reference object in the target backgrounds (plain
surfaces — no ruler/hand/A4). Fall back to the intra-image scale strategy of §4.2.

### 4.4 — Bi-metallic identification (1€, 2€)

- **Question.** How do we identify the bi-metallic class with high confidence? It
  is the most distinctive cue available and very valuable as an anchor for §4.2.
- **Options.**
  (a) a second `cv2.HoughCircles` pass at a smaller radius to detect the inner ring;
  (b) radial colour profile sampled from the centre outwards: a sharp transition at
  ≈ 0.7 × r marks the inner/outer boundary (1€: 16 mm / 23.25 mm ≈ 0.69; 2€: 18 mm
  / 25.75 mm ≈ 0.70);
  (c) edge density at intermediate radius — a non-zero inner contour indicates a
  bi-metallic coin.
- **Evidence (2026-05-26).** On the reference set the radial profile shows the
  signature clearly for **2€** (localized edge peak + brightness step at r/R ≈ 0.72,
  matching 18/25.75 ≈ 0.70) and subtly for **1€**. Method (b) radial profile is
  chosen over a naïve inner-annulus energy score, which fails (it just tracks relief:
  1c scored higher than the bi-metallics). Being gradient-based, it should survive
  the colour cast.
- **Evidence (2026-05-27, target validation).** Two findings. **(i) Colour sign is
  not just dead but *inverted* on targets.** Scanning the 142 targets for big coins with
  a large radial `b*` step (|b\*ring − b\*core| > 9) yields 24 candidates, of which **20
  read "ring-silver / core-gold" (the 2€ pattern) and only 4 the 1€ pattern** — an
  impossible distribution. Cause: the blue specular cast pushes `b*` negative *more on
  the outer ring* (more reflective) than on the core, flipping/flattening the sign that
  on the reference separated 1€ (+14) from 2€ (−13). ⇒ colour **sign** is unusable for
  1€-vs-2€ on targets (hardens DEC-3); only the **magnitude** |b\*ring − b\*core| partly
  survives. **(ii) The structural marker (b) alone is weak on targets.** A radial
  gradient-magnitude profile finds an inner peak, but the separation is poor (mean
  structural score 0.36 on colour-candidates vs 0.19 on monometallic coins) and the
  top-scoring coins are monometallic pieces whose **relief/design** produces a gradient
  ring at ρ ≈ 0.79–0.81 (near the rim), not a true bi-metal boundary at ρ ≈ 0.70.
- **Revised direction.** Neither colour nor structure identifies bi-metals reliably
  *alone*. Combine the two weak cues: a coin is bi-metallic if it has **high
  |b\*ring − b\*core|** (magnitude, sign discarded) **AND** a gradient peak at **ρ ≈ 0.70**
  (not ≈ 0.80). If the combined test is still unreliable, fall back to the
  **constellation scale-fit** (§4.1 f) as the primary anchor, using the bi-metal flag
  only as confirmation.
- **Criterion to close.** Combined colour-magnitude + ρ≈0.70 test validated on targets
  with known 1€/2€; else demote the anchor to the constellation fit.
- **Status.** 🟠 OPEN (reopened by target validation) — reference signature confirmed,
  but on targets colour-sign is inverted (DEC-3) and the structural marker alone is weak;
  combined cue under evaluation.

### 4.5 — Smoothing before Canny

**CLOSED → DEC-4.** **Gaussian** (σ ≈ 2), *not* bilateral. Target images are noisy
(σ ≈ 17 vs reference ≈ 0.3), and the Canny comparison on a noisy target is decisive:
raw is a sea of noise; Gaussian leaves clean coin contours; bilateral, being
edge-preserving, keeps the background noise-texture that floods Hough with spurious
circles (13 circles on a 3-coin image vs 2 with Gaussian). For circle detection only
the strong rim matters, and it survives the Gaussian. (Denoising filters of cap. 3.4
do *not* touch the colour cast — they are a detection pre-step only, never a
classification tool.)

### 4.6 — `cv2.HoughCircles` parameter tuning

- **Question.** Values for `dp`, `minDist`, `param1` (Canny high threshold),
  `param2` (accumulator threshold), `minRadius`, `maxRadius`?
- **Options.** Tune on the single-coin reference images to fix a baseline radius
  range; validate on a 5–10 image sample of the target set.
- **Criterion to close.** On the validation subset, zero missed coins and zero
  spurious circles. Hard cap: three rounds of parameter sweeps; if the budget is
  exhausted, revisit §4.5 instead of pushing further on params.
**CLOSED → DEC-5.** Hough is the detector (§4.7). Working values on a mixed validation
sample: `dp=1.2, minDist≈95, param1=150, param2≈44, r ∈ [45,150]` on a `medianBlur(5)`
grayscale. The key fix is a *mid* `minDist` (~95): it merges duplicate votes for one coin
without dropping touching coins (`image_76`), so no custom de-duplication is needed. The
earlier `minDist=110` "miss" on `image_133` was an artefact of tuning on a single image.

### 4.7 — Detection method: Hough vs LoG/DoG scale-space blobs

**CLOSED → DEC-5 (LoG lead reversed).** A random-sample spike (2026-05-27,
`playground.ipynb` "Detection v3") overturned the earlier LoG lead. The LoG seeks *dark*
blobs only, so it silently **misses bright/specular coins** (`image_132`, `image_138`)
and **floods dark/textured backgrounds with false positives** (`image_103`: 19 "coins").
Hough detects the coin **rim** — an edge present at *both* contrast polarities — and with
a sensible `minDist` (§4.6) it gives exact counts on the showcase with zero FP.
→ **Hough adopted, LoG dropped.** Block-A Gaussian smoothing (DEC-4) is kept (Hough is
noise-sensitive).

### 4.8 — Detection robustness: the weak-rim failure mode

- **Question.** A full-set smoke test (`all_smoke.ipynb`, 2026-05-27) found the Hough
  detector returns **zero** on **32/142 images (23%)**. They are *not* empty and the
  coins are *not* tiny (r ≈ 40–55 px): they have **weak rims** — soft edges on textured
  light cloth — that score below the accumulator. Recover them without wrecking the easy
  images?
- **Evidence (what fails).** Every *global* lever couples recall and false positives:
  lowering `param2`, lowering `param1`, or CLAHE all surface the weak coins **and** flood
  the easy images with FP (`image_103`: 3→108 at low `param1`; CLAHE → ~115 circles/img).
  No single global setting serves both regimes.
- **Evidence (partial fix).** Sensitive Hough (high recall + FP) + a **rim-score** filter
  (fraction of the perimeter carrying a strong radial gradient, §4) halves the validation
  count-error (28→15) and recovers ~half the weak coins, but misses the softest ones and
  adds the odd FP/regression on easy images. A geometric filter cannot cleanly separate
  weak coins from FP — both have weak gradients.
- **Spike result (2026-05-27, `playground.ipynb` v5–v7).** Pipeline = **overshoot
  (CLAHE+median, recall 28/28 · ~95 FP) → per-candidate gate**. Gates compared on a 14-image
  validation set (8 weak + 6 easy), scored as recall-weak / FP-easy:
  - **rim-score** (geometric, §4): 5/28 · 0 FP — precision-perfect but kills the weak (it
    measures rim-gradient strength, exactly what these coins lack).
  - **ZNCC** (§6.1, intensity): 27/28 · 30 FP — keeps recall but imprecise (texture patches
    correlate in intensity).
  - **shape-based** (§6.2, gradient *direction*): **18–20/28 · 3–4 FP** — best
    precision/recall trade-off, ~10× cleaner than ZNCC. Two subtleties: the dot product must
    be **signed** (`|dot|` floors at E[|cos|]=0.64 on random gradients), and an outer `|·|`
    on the mean absorbs the cast's bright/dark polarity flip. Validates **§6.2 over §6.1** on
    the cast-corrupted set: gradient orientation beats intensity correlation (the cast
    corrupts intensity, not edge geometry).
  - **shape OR contrast / S-contrast** (combination attempt): **rejected** — adds a few weak
    (20→24) but at 3–4× the FP (3→12), because the softest coins lack both rim *and* contrast
    while FP stains have contrast. Combination does not improve the trade-off.
- **Chosen direction.** **Overshoot (CLAHE+median) → shape-based gate (§6.2, DEC-8)** for
  weak-rim recovery. Recovers ~64–71% of weak coins at ≈3–4 FP on the validation set.
- **Criterion to close.** Validate the full overshoot→shape-gate pipeline on a wider random
  target sample (rerun `all_smoke.ipynb`); confirm the zero-detection rate drops without the
  easy-image FP rate blowing up.
- **Status.** 🟢 DIRECTION SET — overshoot + shape-gate (§6.2); residual gap ≈10 softest-rim
  coins. Pending wide-sample validation before folding into `solution.ipynb`.

## 5. Decision log

Append-only. Each row records a *closed* decision with the date, what was decided,
why, and the chapter of the summary that legitimises it.

| ID  | Date       | Decision | Why | Summary § |
|-----|------------|----------|-----|-----------|
| DEC-1 | 2026-05-26 | Drop absolute radius; use intra-image **relative** radius + per-image scale anchor. | 1€ (r=98) detected larger than 2€ (r=82) across reference shots though physically smaller ⇒ pixel radius is not metric. Coins in one image share distance/pose ⇒ radius *ratios* are metric. | §6.3 Hough + scale reasoning |
| DEC-2 | 2026-05-26 | No background scale anchor; rely on intra-image scale (§4.1 f / §4.2). | Target backgrounds are plain surfaces — no ruler/hand/A4. | — |
| DEC-3 | 2026-05-26 | Colour eliminated as a cross-set classification cue. | Target coins carry a strong blue specular cast (hue ≈ 100 vs ref ≈ 10); not a global WB (bg ≈ neutral), not invertible by a filter; copper/gold inseparable even by eye. | §3.4 (filters cannot undo it) |
| DEC-4 | 2026-05-26 | Block A pre-Canny smoothing = **Gaussian** (σ ≈ 2), not bilateral. | On noisy targets the bilateral preserves background noise-texture ⇒ Hough floods with spurious circles (13 vs 2 on a 3-coin image); Gaussian suppresses it and keeps the coin rim. Empirically overturned the theory-based bilateral lean. | §3.3 Gaussian, §4 Canny |
| DEC-5 | 2026-05-27 | Detector = **Hough circles** (mid `minDist≈95`); LoG dropped. | LoG is dark-blob-only ⇒ misses bright/specular coins and floods dark backgrounds with FP (`image_103`: 19); the Hough rim is polarity-agnostic and exact on the showcase. Reverses §4.7's LoG lead. | §6.3 Hough, §4 edges |
| DEC-6 | 2026-05-27 | Reject *global* sensitivity levers (`param1`/`param2`/CLAHE) for weak-rim recovery. | Recall and false positives are coupled: any global rise in sensitivity recovers weak coins but floods easy images (`image_103`: 3→108). Recovery, if any, must be a post-hoc per-candidate validation (§4.8). | §4 edges, §6.3 |
| DEC-7 | 2026-05-27 | Discard the bi-metal colour **sign** on targets; identify bi-metals by \|b\* step\| **magnitude** + a radial gradient peak at ρ≈0.70 (combined), else fall back to the constellation scale-fit. | The blue specular cast inverts the ring/core `b*` sign on targets (24-coin scan: 20 read 2€-like, 4 1€-like — impossible); the structural marker alone is weak (monometallic relief peaks at ρ≈0.80). Neither cue suffices alone. | §4 edges, §6 instance-level |
| DEC-8 | 2026-05-27 | Weak-rim recovery = **overshoot (CLAHE+median) → shape-based gate** (§6.2, signed gradient-direction match vs the 8 references). Reject ZNCC, rim-score, and the contrast/S-channel combination as the gate. | On validation the shape gate gives 18–20/28 recall at 3–4 FP — ~10× cleaner than ZNCC (27/28 · 30 FP) and far more recall than rim-score (5/28). Gradient *direction* is intensity-invariant ⇒ survives the cast; the signed dot is what makes it discriminate. Contrast cues add FP faster than recall. | §6.2 shape-based matching, §4 |

## 6. Journal

Append-only, newest-on-bottom. Each entry is dated and describes what was tried,
what was observed, and what changes for the open decisions in §4.

### 2026-05-26 — M0 inspection + M1 calibration + cap.3.4 filter eval

- **Data.** `coin_dataset/`: reference_set = 8 labelled single-coin images,
  target_set = 142 multi-coin images, all 960×720.
- **M0/M1.** Confirmed: absolute radius not metric (→ DEC-1), no bg anchor (→ DEC-2),
  blue **specular** colour cast on target coins (→ DEC-3). SIFT-vs-reference weak
  (ref×ref matrix not diagonal-dominant; target predictions collapse onto high-keypoint
  refs; ≈ 50% two-sides recall) → §4.1(d) parked. Bi-metallic radial-profile signature
  confirmed on reference, clear for 2€ (→ §4.4).
- **Filter bench (cap. 3.4 / Table 1).** Mean/Gaussian/Median/Bilateral/NL-means.
  Coin hue unchanged by every filter ⇒ denoising does **not** remove the cast. But the
  target noise is real (σ ≈ 17 vs reference σ ≈ 0.3) ⇒ edge-preserving smoothing is
  justified as a *detection* pre-step (→ §4.5 leaning bilateral).
- **Direction.** Classification pivots to **geometry-first joint inference** (§4.1 f).
- **Artifacts.** `playground.ipynb`; scratch figures in `.scratch/` (gitignored).

### 2026-05-26 (b) — cap. 5.4 scale-space LoG blob detector

Tried on the hunch that "multiple blur levels" (§5.4) might help. A hand-built
scale-normalized LoG scale-space detects coins as blobs and reads their radius from the
characteristic σ. On `image_133` it finds **3/3** (Hough found 2/3, missing the
low-contrast coin) and **4/4** on `image_76`, cleanly; robust across a 12-image sample.
→ opened §4.7 (LoG leading over Hough); if adopted it folds block A into B and makes
§4.5/§4.6 moot. Honest limit: it is a detection/size tool only — it does not address
the colour cast or classification.

### 2026-05-27 — Detection hardening on the full target set

- **Showcase spike (`playground.ipynb` Detection v3).** Ran the detectors on a random
  6-image sample → **reversed §4.7**: the LoG misses bright coins (`image_132/138`) and
  floods dark backgrounds (`image_103`: 19 FP), while Hough with a mid `minDist≈95` is
  polarity-agnostic and exact (4/3/3/4/1/1), no custom dedup needed (DEC-5).
- **Full smoke test (`all_smoke.ipynb`, git-ignored).** 142 images, 228 coins. **32
  images (23%) return zero** — not empty: coins with **weak rims** (r ≈ 40–55 px) on
  textured cloth (→ §4.8). Over-detection is real density, not FP; duplicates a non-issue.
- **Weak-rim recovery attempts.** Global levers (`param1`/`param2`/CLAHE) all couple
  recall and FP (DEC-6). A sensitive-Hough + rim-score filter halves the validation
  count-error (28→15) but only partially. Author proposes an **instance-level gate**
  (overshoot + keep strong appearance matches); risk flagged, spike pending (§4.8).
- **Artifacts.** `playground.ipynb` (Detection v3 + v4); `all_smoke.ipynb`.

### 2026-05-27 (b) — Colour cast inverts bi-metal appearance on targets

- **Trigger.** Noticed 1€/2€ look colour-swapped on targets vs reality. Verified: on the
  **reference** the ring-vs-core `b*` is correct (1€ ring=gold +14; 2€ ring=silver −13).
- **On targets it inverts.** Among 24 big coins with a large radial `b*` step, **20 read
  "2€-like" (ring silver / core gold), only 4 "1€-like"** — impossible. The blue specular
  cast drives `b*` negative *more on the outer ring* than the core, flipping the sign that
  separated 1€/2€ on reference. ⇒ colour **sign** unusable for 1€-vs-2€ on targets
  (hardens DEC-3); only |b\* step| **magnitude** survives. → §4.4.
- **Structural marker validated, found weak.** Radial gradient profile: separation poor
  (0.36 vs 0.19), top scores are monometallic **relief** (peak at ρ ≈ 0.80, not 0.70).
  Revised direction: **combine** |b\* step| magnitude **AND** gradient peak at ρ ≈ 0.70;
  else demote the bi-metal anchor to the constellation scale-fit (§4.1 f). → §4.4 reopened.
- **Artifacts.** ad-hoc analysis scripts (not committed).

### 2026-05-27 (c) — Weak-rim recovery: overshoot + gate, shape-based wins

- **Transform sweep.** Tested 7 pre-Hough transforms (bilateral, median, morph-close, S+bilat,
  CLAHE, unsharp). Recall-weak and FP-easy are coupled across all of them: soft transforms
  recover nothing, aggressive ones (unsharp/CLAHE) recover 28/28 weak but at 95–384 FP. ⇒ no
  global transform breaks the trade-off (DEC-6); CLAHE+median is the best **overshoot**.
- **Gate comparison** (overshoot → per-candidate gate, validation set): rim-score 5/28·0FP;
  ZNCC (§6.1) 27/28·30FP; **shape-based (§6.2) 18–20/28·3–4FP** — the winner (DEC-8). Key:
  *signed* gradient-direction dot (the `|dot|` version floors at 0.64 and fails), outer `|·|`
  for the cast polarity flip.
- **Combination shape OR contrast / S-contrast.** Rejected: +recall (→24) costs 3–4× the FP.
- **Outcome.** §4.8 → 🟢 direction set: overshoot (CLAHE+median) → shape-based gate. Residual
  ≈10 softest-rim coins. Pending wide-sample (`all_smoke`) validation before `solution.ipynb`.
- **Artifacts.** `playground.ipynb` (Detection v4 rim-score, v5 ZNCC, v6 shape-based, v7 combo).

## 7. TODO — high-level milestones

Project-level milestones only. Day-to-day tasks are tracked in the session task
tracker, not here.

- [x] **M0** — Dataset in place; visual inspection done (2026-05-26). Closed the
      "blocked on dataset inspection" status on §4.1 / §4.2 / §4.3.
- [~] **M1** — Reference radii + colour measured; radius found non-metric and colour
      cast-corrupted (DEC-1/DEC-3). Per-denomination calibration to be finalized with
      the scale-grid fit.
- [~] **M2** — Detection = Hough (DEC-5), validated on a 6-image showcase + full
      142-image smoke test (2026-05-27). Closes §4.5–§4.7; **§4.8 (weak-rim) open**.
- [ ] **M3** — Classification block working on the reference set, evaluated
      out-of-sample on the eight reference images themselves (leave-one-out).
      Definitively closes §4.1.
- [ ] **M4** — End-to-end pipeline running on the full `target_set`,
      producing per-image `Partial Amount` and global `Total Amount`.
- [ ] **M5** — Notebook polish: prose, justifications anchored to the summary,
      discussion of failure cases and limits.
