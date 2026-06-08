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
- **Update (2026-05-27).** Most appearance matchers fail (SIFT/ZNCC/shape/edge-NCC/chamfer ≈ random),
  BUT a **fair GHT reopens the number cue** (DEC-12): uniform-density templates + size-normalized
  score score **4/8** on hand-read GT (top-2 in 6/8), with errors structured as cent-vs-euro. So the
  committed direction is a **fusion**: GHT-number (digit) **+** size **+** bimetal structural flag,
  where size/bimetal resolve the cent-vs-euro ambiguity the digit leaves. The per-image **scale
  anchor** (§4.2/§4.4) is still needed for the size half.
- **Update (2026-05-30, DEC-15).** The operating pipeline is now **§11.3 bis** (`scale_playground`):
  Hough → **whole-disc shape match-%** (gradient-orientation on per-coin control points) **+** material /
  bi-metal colour favour **+** relative-size cues (order + same-class) **+** per-coin localized blur on the
  reference. This is a **different line** from the GHT-digit of DEC-12 (set aside): it scores the whole disc,
  not the segmented number. §12 **falsified** the relative bronzo/oro colour split — colour transfers only as
  a **bi-metal gate**. The residual bottleneck is the **absolute scale** (`s=median(r/DIAM[pred0])` is
  seed-circular), not the colour. Whole-disc match-% stays low/close (~40–50%), so the cue fusion (shape +
  bi-metal + relative size), not the shape alone, carries the decision.
- **Status.** 🟢 DIRECTION SET (DEC-15: §11.3 bis fusion — whole-disc shape + bi-metal colour + relative
  size + reference blur). Open: absolute **scale anchor** (the real bottleneck) and a **larger GT**.
  GHT-digit (DEC-12) parked as an alternative line.

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
- **Alternative gate (2026-05-27, v15) — classifier *confidence*, comparable to the shape gate.**
  FP/duplicate circles get a **flat GHT profile** (margin ≈ 0), real coins a clear winner, so the
  GHT **margin** is itself an FP filter. Reproducible (overshoot `minDist=85 param2=42`, sharp/raw
  reference templates): `margin ≥ 0.08` → **21/28 · 3 FP**, `≥0.07` → 22/28 · 7 FP — **comparable**
  to the shape gate (18–20/28 · 3–4 FP). **FP is parameter-sensitive** (2–7 FP at 0.07 across
  denoise/rotation configs); the gate wants SHARP templates whereas the classifier wants denoised.
  rim-score still fails (kills the weak coins). Advantage = reuses the classifier (DEC-13).
- **Chosen direction.** **Overshoot (CLAHE+median, `minDist=85 param2=42`) → GHT-confidence gate
  (`margin ≥ 0.08`, sharp templates)** — DEC-13. ≈21/28 (75 %) weak-rim recovered at ≈3 FP, on par
  with the shape gate but reusing the classifier. (Shape gate DEC-8 remains a valid alternative.)
- **Criterion to close.** Validate on a wider random target sample (rerun `all_smoke.ipynb`);
  confirm the zero-detection rate drops without the easy-image FP rate blowing up.
- **Status.** 🟢 DIRECTION SET — overshoot + **GHT-confidence gate** (DEC-13; comparable to and
  preferred over the DEC-8 shape gate for reuse). Residual ≈5 softest coins have margin < 0.05
  (genuine limit). Pending wide-sample validation before folding into `solution.ipynb`.

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
| DEC-9 | 2026-05-27 | **Reject the §6.5 SIFT star model as detector/recall booster; keep SIFT+RANSAC only as an optional high-precision *confirmation* for classification.** Detection stays Hough (DEC-5); classification stays geometry-first (§4.1). | Tested all 3 roles on the validation set. **A** (star-model GHT detection) = **0/28** recall — votes too sparse. **C** (recover Hough-missed weak-rim) ≤ A = 0 by construction. **B** (Hough detects → SIFT classifies the disc) is the only viable one but **marginal: 1/13 confident verdicts, median top-1 = 6 RANSAC inliers**, and white-balancing the cast does not help. References separate perfectly when clean (8/8 via RANSAC) but the cast + wear + small scale (DEC-3/7) collapse target matching. | §6.5 star model, §6.1, §5.6 |
| DEC-10 | 2026-05-27 | **Reject appearance-based classification entirely (texture *and* edge); classification will be size/geometry-driven (scale-grid fit on the Hough radii), with colour-family/SIFT only as a secondary tie-breaker.** | Both feature families fail on the cast targets: SIFT+RANSAC 1/13 confident (DEC-9), and the edge-based per-reference shape match (§6.2) gives a top1−top2 margin of only ≈0.02–0.03 (random). Root cause is not the feature choice but that the discriminative relief is destroyed by the cast + small scale (r≈50–110 px); only the **diameter** survives, and intra-image radius ratios are metric (DEC-1). | §4.1f geometry-first, §6.2 |
| DEC-11 | 2026-05-27 | **Close appearance-based per-denomination classification — the whole matcher family is exhausted.** Classification = partial **geometry-first** (bimetal structural flag + size where an anchor exists) + a documented limit, per the brief's "justified partial > blind complete". | v8–v13 tested SIFT, ZNCC, shape-gradient (§6.2), edge-NCC, chamfer, and the GHT (§6.4, in-circle, rotation-invariant) — **all ≈ random** — across raw/CLAHE/denoise/unsharp. Root cause is **dilution**: the discriminative digit is a small fraction of the in-circle edges, dominated by the common-side design that is *identical across denominations* (Europe map + stars) plus wear; and the digit cannot be cleanly segmented to de-dilute (subtle relief, no threshold). The number IS visible on ~all coins — the earlier "national-side cap" was a misread of *rotated* digits — it is simply not recoverable by hand-crafted matching (that needs learned OCR, forbidden). | §6.1–6.4, §4.1f |
| DEC-12 | 2026-05-27 | **SUPERSEDES the GHT part of DEC-11: a *fair* GHT does recover the number cue — reopen appearance classification as a partial cue.** | DEC-11's "GHT dead" rested on a **broken** GHT (v13): the strong-edge filter gave wildly uneven template sizes (1cent 0 edges, 2cent 15, 20cent 1287) and the peakedness score rewarded sparse templates, so "2cent" won by artifact. A **fair** GHT (v14) — *uniform-density* top-N (=250) strongest-gradient points per coin + *size-normalized* score (fraction of N matched at best rotation, in-circle) — scores **4/8** on hand-read GT (random ≈1/8), right answer in the **top-2 for 6/8**, errors **structured as cent-vs-euro** (the digit matches, e.g. 2cent→2euro). So the digit/number cue is **partially recoverable** and should be **fused with size + bimetal** to settle cent-vs-euro. | §6.4 GHT, §4.1f |
| DEC-13 | 2026-05-27 | **Weak-rim gate: GHT-confidence (`margin = top1−top2`) as an alternative to the DEC-8 shape gate — *comparable* recall/FP, preferred because it reuses the classifier (no ad-hoc dedup).** | FP/duplicate circles get a flat GHT profile (margin ≈ 0 — classifier undecided), real coins a sharp winner, so the margin doubles as an FP filter. Reproducible (overshoot `minDist=85 param2=42`, sharp/raw reference templates): `margin ≥ 0.08` → **21/28 · 3 FP**, `≥0.07` → 22/28 · 7 FP — comparable to the shape gate (18–20/28 · 3–4 FP). **FP rate is parameter-sensitive** (2–7 FP at margin 0.07 across denoise/rotation configs; an earlier config showed a fragile 23/28·2FP). Note: the **gate wants SHARP templates, the classifier (v14) wants DENOISED** — opposite goals. ~5 softest coins (margin < 0.05) remain the documented limit. | §6.4 GHT, §4.8 |
| DEC-14 | 2026-05-27 | **Bi-metal flag = chroma-gradient ring (a*/b* radial profile peak at ρ≈0.70), NOT the grayscale gradient.** Resolves cent-vs-euro for the GHT digit 1/2. | The cast kills *absolute* colour (DEC-3), but a bi-metal coin still has a **chroma discontinuity** at the metal boundary that survives the cast. Chroma-ring score cleanly separates the references (1euro +0.53, 2euro +0.65 vs all mono ≤ +0.11) where the luminance-gradient version failed (2cent +0.30 ≈ 2euro +0.31), and flags exactly the two euros in `image_85` (+0.70, +0.45 vs ≤0.05). Threshold ≈0.40; a few mono targets are borderline. | §4.4, §2 colour |
| DEC-15 | 2026-05-30 | **Adopt §11.3 bis (`scale_playground`) as the operating detection+classification pipeline.** Hough detection (DEC-5) → whole-disc **shape-edge match-%** (gradient-orientation agreement on per-coin Canny control points, 360°-rotation-searched) **+** soft **material / bi-metal colour favour** **+** relative-**size** cues (diameter-order `W_ORDER11=0.25`, same-class-size `W_SAME11=0.15`, scale `s=median(r/DIAM[pred0])`) **+** per-coin **localized feathered blur on the reference** (`BLUR3b`: ring/disc/box, panel-tunable). | Most parametrized, self-contained framework to iterate on; fuses the three surviving cues (shape + structural colour + relative size) with per-coin reference control and targeted blur to drop spurious reference edges. **Honest limits (measured in §11/§12):** whole-disc match-% is **low and close** (~40–50%, consistent with DEC-10's "edge-shape weak"); the scale `s` is **circular** (depends on the seed prediction); colour separates **only bi-metal** (DEC-3/14) — the relative bronzo/oro split was **falsified** on targets (§12, e.g. `image_58` 3.50→0.40 when forced); the blur deliberately breaks ref/target denoise symmetry. **Sets aside the DEC-12 GHT-digit** (different line; not used by §11.3 bis). | §6.2 shape matching, §6.3 Hough, §4.4 colour, §5 local features |

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

### 2026-05-27 (d) — Instance-level §6.5 (SIFT star model): three roles tested, all weak

- **Question.** Use DoG/SIFT features (§6.5 star model) for detection or classification?
  Clarified the terminology trap: the §6.5 "Hough" is a **4D GHT vote**, not `cv2.HoughCircles`
  — it *replaces* the circle detector, it does not feed it.
- **Feasibility (v8).** SIFT descriptors separate the 8 references **perfectly only via RANSAC**:
  ref-vs-ref inlier matrix is diagonal-dominant (16–84 self vs 0–6 off), **8/8** correct. Raw
  ratio-test match *counts* are noise (winner flips with scale, ties) — this is why the M0 spike
  (2026-05-26) read SIFT as hopeless: it lacked geometric verification.
- **Role A — star-model GHT detection.** Pooled all reference keypoints + radius-normalized
  joining vectors, voted the centre in a 2D accumulator. **0/28** weak-rim recall at every vote
  threshold — too few coherent matches per coin to peak.
- **Role B — Hough detects → SIFT classifies the disc.** Hough localises (robust); RANSAC
  inliers vs the 8 references pick the denomination. **Marginal: 1/13 confident** (top1 ≥ 8 &
  margin ≥ 4), **median top-1 = 6 inliers**; gray-world white balance does **not** help (the bg
  dominates the WB stats). Works only on the rare strong match (e.g. `image_35` → 1cent = 17).
- **Role C — recover Hough-missed weak-rim by voting.** Bounded by A (≤ 0); weak-rim coins are
  the most cast-degraded ⇒ the worst for SIFT. Dead by construction.
- **Outcome → DEC-9.** Detection stays Hough; classification stays geometry-first (§4.1); SIFT
  +RANSAC demoted to an *optional* high-precision confirmation. Root cause: DEC-3/DEC-7 cast.
- **Artifacts.** `playground.ipynb` (Detection v8 feasibility, v9 role A, v10 role B, verdict).

### 2026-05-27 (e) — Smoke rerun + edge-based classifier (replace SIFT); size is the cue

- **Smoke rerun (`all_smoke.ipynb`).** Re-ran the clean Hough detector on all 142 targets:
  **228 coins, 32 zero-detection images (23%)** — unchanged (detector untouched), confirms the
  weak-rim baseline. The 32 zeros are the same weak-rim set feeding §4.8.
- **Edge-based classifier (v11), replacing SIFT.** Tried shape-based matching (§6.2) applied
  *per reference* (argmax over the 8 coins) so the cast-immune gradient *direction* names the
  denomination. Interior control points (rim is shared) and a full-disc variant (adds the
  bi-metal ring step). Result: top1−top2 margin **mean 0.018 / median 0.015** (interior),
  **0.028 / 0.021** (full disc) — i.e. **random**; slightly worse than SIFT's 1/13.
- **Reading → DEC-10.** The failure is not texture-vs-edge: on these targets (r≈50–110 px +
  cast + wear) the discriminative relief itself is gone, starving *both* feature families.
  What separates cleanly is **size**: intra-image Hough radii cluster tightly and
  proportionally (`image_132`: 96/99/101/106; `image_76`: 98/101/103/106), and ratios are
  metric (DEC-1). ⇒ classification becomes **size-driven** (scale-grid fit on the Hough radii),
  appearance only as a secondary tie-breaker. Next: prototype the scale-grid classifier; the
  open sub-problem is the per-image scale anchor for single-coin images (§4.2/§4.3).
- **Artifacts.** `playground.ipynb` (Detection v11 + verdict); `all_smoke.ipynb` (re-run).

### 2026-05-27 (f) — Appearance classification exhausted; number cue investigated and closed

- **National-side correction.** The earlier "national-side cap" was **wrong**: the coins flagged as
  faceless (15, 68, 109, 116, 123) all show the value, just **rotated** (image_15 = a 5 turned 90°,
  image_68 = an upside-down 1, image_109 = a tilted 2, …). The number is visible on essentially all
  coins; the ~60% common-side estimate is moot. Fixed the false note in playground v12.
- **Cue measurements (validation set).** Colour→family: the cast collapses it even *relatively*
  (copper a* only ~6 above gold, distributions overlap). SIFT raw 1/13, SIFT enhanced 0/13 (unsharp
  amplifies noise). Edge-shape (§6.2) margin ≈0.02. ZNCC-enh margin 0.03 (0.05 with denoise).
  Edge-NCC / chamfer in-circle 0/8. **GHT in-circle** (rotation-invariant, §6.4): permissive
  *saturates* (2/8), strict+peakedness *degenerates* onto the most-structured ref (1/8).
- **Denoise (cap. 3.4 / lab 1).** NL-means visibly cleans the σ≈17 noise and lifts the ZNCC margin
  0.03→0.05 — helps *legibility*, not classification. Fights the noise, not the cast (DEC-3/4).
- **Why everything fails — dilution.** The common side is shared across denominations (map + stars);
  the only discriminator (the digit) is ~10–20 % of the disc and cannot be segmented (subtle relief,
  no clean threshold), so *both* holistic matchers and rotation-robust voting are swamped. Reading it
  needs human/learned gestalt OCR (forbidden).
- **Decision → DEC-11.** Close appearance classification. Classification becomes partial
  geometry-first (bimetal + size + documented limit) — the brief explicitly rewards this.
- **Artifacts.** `playground.ipynb` (v8–v13); this evidence *is* the procedural justification.

### 2026-05-27 (g) — A *fair* GHT recovers the number cue (DEC-11 was premature)

- **Trigger.** "Why does the GHT always predict 2cent?" → the score was an **artifact**. Strong-edge
  counts per reference were wildly uneven (1cent **0**, 2cent **15**, 10cent 63, … 20cent **1287**),
  and the peakedness score (max − mean over rotation) **rewards sparse templates** → 2cent (15 pts)
  produced the spikiest profile and won regardless of the target. v13 was not a fair test of GHT.
- **Fair GHT (v14).** Uniform density: take the **N=250 strongest-gradient points** within the circle
  for *every* coin; score the **fraction of N matched** at the best rotation (size-normalized, K=18
  orientation bins, ±2 px / ±1 bin tolerance), in-circle. Result on the 8-coin hand-read GT: **4/8**
  (random ≈1/8), correct denomination **top-2 in 6/8**. Errors are **structured cent-vs-euro**
  (`image_22` 2cent→2euro, same digit "2"; `image_4` 1cent→2cent with 1cent close 2nd): the *digit*
  matches, the residual ambiguity is cent vs euro.
- **Decision → DEC-12.** Reopen appearance classification: the number cue is **partially recoverable**
  via a fair GHT, to be **fused with size + bimetal** to resolve cent-vs-euro. (Corrected the v13
  verdict + the v12 "national-side" note in the playground.)
- **Next.** Expand the hand-read GT, tune the fair GHT, and build the GHT-number + size + bimetal
  fusion; the per-image scale anchor (§4.2/§4.4) is still needed for the size half.
- **Artifacts.** `playground.ipynb` (v14 fair GHT; v13 annotated with the correction).

### 2026-05-27 (h) — Colour rescues the bi-metal flag; GHT-confidence becomes the FP gate

- **Colour, done right (DEC-14).** Absolute colour stays dead under the cast (image_85 is all
  blue), BUT the **chroma-gradient ring** at ρ≈0.70 survives — it is a *relative* discontinuity
  between two metals, not an absolute hue. On references it cleanly flags the bi-metals (1euro
  +0.53, 2euro +0.65 vs mono ≤ +0.11) where the grayscale gradient failed (2cent ≈ 2euro), and
  on `image_85` it flags exactly the two euros. So colour is useful **specifically for the
  bi-metal flag**, the cent-vs-euro discriminator the GHT digit needs.
- **GHT confidence = FP filter (DEC-13).** On image_85, FP/duplicate circles had a **flat** GHT
  profile (margin ≈ 0.01–0.03) and low rim, while real coins had margin 0.10–0.14 and rim ≥ 0.50.
  Generalised on the harness (overshoot `minDist=85 param2=42`, raw 28/28 · 124 FP), the gate is
  **comparable to the shape gate, and parameter-sensitive**. Honest/reproducible (sharp/raw
  reference templates): `margin ≥ 0.08` → **21/28 · 3 FP**, `≥0.07` → 22/28 · 7 FP (shape gate:
  18–20/28 · 3–4 FP). **Caveat found the hard way:** the FP rate swings with preprocessing — a
  controlled ref×target denoise sweep gave 2 FP (raw refs), 24 FP (denoised refs) at margin 0.07,
  and an earlier config's headline "23/28 · 2 FP" was that fragile corner, not a robust result. The
  **gate wants sharp templates, the classifier wants denoised** (opposite goals). Rim-score still
  kills the weak coins. Net advantage of the confidence gate: it **reuses the classifier**.
- **Decisions.** DEC-13 (confidence gate supersedes shape gate), DEC-14 (chroma-ring bi-metal).
- **Artifacts.** `playground.ipynb` (v15 confidence gate + chroma-bimetal); `DECISIONS.md` digest.

### 2026-05-28 — Full pipeline visualized end-to-end on 3 input variants (playground v17)

- **Pipeline shown** (real images in the notebook, 6 samples × 3 stages): overshoot **colour-saliency**
  detection → **C1** number/design match (fair GHT) per circle → **FP filter** (drop circles with GHT
  score < 0.30). Run on **natural / negative (255−img) / R↔B-swap** inputs.
- **3-variant equivalence confirmed *visually*.** Negative and R↔B-swap give **practically identical**
  detection and survivors to natural — colour-saliency is (near) invariant and gradients are sign/relabel
  -invariant. So neither transform changes the algorithm; both are cosmetic (coins look coppery, useful only
  for the eye). Closes the channel-swap / negative thread.
- **Weak-rim recovery visible.** On `image_38` (luminance-zero) the colour-saliency front-end detects the
  coins in all three variants — the colour "pop" works.
- **Honest state of the pipeline.** Overshoot recall ✓ and FP filter visibly thins the over-detection, but
  **C1 inherits the ~50 % GHT reliability** and the GHT-score FP filter is imprecise (drops some real coins,
  keeps some FP) — reliable digit *isolation* (the de-diluting step that would lift C1) is still unsolved on
  these worn/cast coins. The architecture is sound; the open crux remains number-only matching on degraded targets.
- **Denoise BEFORE stage 1 (v17b) — halves the FP flood.** The saliency was running on the *raw*
  image, so background noise/texture spawned spurious salient spots. Computing it on the
  **NL-means-denoised** image cuts the textured-background flood roughly in half (image_43 85→30,
  image_55 86→41, image_24 81→50) at a small weak-rim cost (20→18/28). Fix applied to the v17 pipeline.
- **Artifacts.** `playground.ipynb` v17 (3 montages) + v17b (raw-vs-denoised detection), real images.

### 2026-05-28 (b) — Isolate the number: reference-side de-dilution (zoom + light denoise) — counterproductive alone

Goal: beat the ~50 % GHT ceiling by de-diluting the template to the numeral. Worked the **reference side**.

- **Denoise level (colour) sweep** on refs+targets symmetric: GHT top-1 peaks at **h≈7 (4/8)**; h=0→1/8, h=12→2/8.
  A sweet spot exists, but it only *equals* the existing 4/8. NL-means h=12 was mildly over-smoothing the *clean*
  references (visible: digit washed out), but using **raw** references is worse (2/8) — the references must match the
  target's smoothness (the targets are denoised), so smoothness must be aligned, not maximised.
- **Numeral-only reference** (auto-localised box, edges only): **4/8** — no gain over full-disc; the auto box was a real
  ~15 %-of-disc region, so the test was valid.
- **Manual numeral boxes.** Built an in-notebook **ipywidgets selector** (no GUI/X needed; persists to
  `manual_boxes.json`) so the boxes can be hand-drawn; `select_boxes.py` is a cv2 alternative (needs WSLg).
- **Manual box → zoom → GaussianBlur(σ) → features**, matched vs the full-disc target: **1/8 (worse)**. Predictions
  collapse onto **20c/50c**. Cause = **scale mismatch**: zooming only the reference makes the numeral fill the 160-frame,
  while the target numeral stays a small fraction of its full disc → references with larger numerals win.
- **Conclusion.** Isolating the number is the right idea but must be done on **both** sides; de-diluting *only* the
  reference is counterproductive. The wall remains **localising the numeral on the (degraded) target**, which we cannot
  do robustly — so the appearance/number cue stays capped at ~50 %, and the committed plan is unchanged
  (Hough detection + counting solid; classification partial, geometry/bimetal-first; appearance as a weak confirmation).
- **Artifacts (`playground.ipynb`).** Reference-features view; denoise sweep + h=7-vs-h=12; interactive box selector;
  manual-box features (zoom→σ); classification test with full-colour denoised target figures. Plus `manual_boxes.json`,
  `select_boxes.py`.

### 2026-05-30 — Scale-aware line (§11): whole-disc shape match-%; family-colour falsified; §11.3 bis adopted (DEC-15)

- **§11 (`scale_playground`).** Built a self-contained scale-aware classifier: a whole-disc **shape-edge
  match-%** (gradient-orientation agreement on per-coin Canny control points, 360°-rotation-searched)
  replaces the digit-box ZNCC; a **colour favour** (material group + bi-metal chroma ring) and **soft
  multiscale size cues** (diameter-order + same-class-size) are fused in. §11.3 makes the reference side
  per-coin (Canny params + edge-preserving filter), symmetric ref/target.
- **§12 (family recognition) — falsified the relative colour split.** Tried a reliable
  bronzo|oro|bi-metallo split. Findings on certain-GT data: **absolute colour thresholds do not transfer**
  ref→target in *any* space; Lab `a*−b*` separates bronzo/oro only *relatively, intra-scene*; but on real
  multi-coin scenes the clustering rarely splits → the relative split is **unreliable**. **Colour is robust
  only as a bi-metal gate** (core/ring chroma step, adaptive boundary). Fusing colour+size into the value
  classifier (§12.7) **worsened** predictions (soft favour on ambiguous families + a self-confirming scale +
  a steep size penalty reinforce each other's errors; `image_58` 3.50→0.40). Variant §12.8 (hard bi-metal
  gate + consensus scale + order-DP) recovers GT (mono 4/4, `image_49` Total 1.70) but the **scale** stays the
  bottleneck on large-coin scenes.
- **Digit pipeline check.** Confirmed a digit-patch ZNCC exists (§0–§10, `digit_match`) — usable to
  *straighten* a coin (best-θ) and as an alternative matcher. The user's "oro digit on the right, others on
  the left" rule was **falsified** on the real coins (1c and 50c both carry the number on the left).
- **§11.3 bis.** Adds a per-coin **localized, feathered blur on the reference** (`BLUR3b`; ring/disc/box) to
  suppress spurious reference edges, with §11.3's tuned parameters preserved (snapshot `*3b`). Verified: blur
  area + suppressed edges render correctly; classification runs on the blurred atlas.
- **Decision → DEC-15.** Adopt **§11.3 bis** as the operating detection+classification pipeline. Open nodes
  unchanged and now sharper: the **absolute scale anchor** is the real bottleneck (§4.1/§4.2), and a **larger
  GT** is needed; the colour family beyond bi-metal is a *documented dead end* (§12).

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
