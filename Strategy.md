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
- **Criterion to close.** Validate on a target bi-metallic and harden the 1€ detector.
- **Status.** 🟢 STRUCTURAL SIGNATURE CONFIRMED on reference (b); pending target
  validation.

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
- **Evidence (2026-05-26).** Now the **active bottleneck**. With Gaussian + (dp=1.2,
  minDist=110, param1=120, param2=42, r ∈ [38,210]), `image_133` detects 2 of its 3
  coins — the lighter, low-contrast coin is missed. Lowering `param2` should recover
  it without reintroducing the bilateral's false positives.
- **Status.** 🟡 OPEN — *but possibly moot:* §4.7 may replace Hough with a LoG
  scale-space blob detector; tune only if Hough is retained.

### 4.7 — Detection method: Hough vs LoG/DoG scale-space blobs

- **Question.** Detect coins with `cv2.HoughCircles` (on a Gaussian-smoothed image)
  or with a **LoG/DoG scale-space blob detector** (§5.4)?
- **Evidence (2026-05-26).** A coin is a dark blob on a lighter background. A hand-built
  scale-normalized LoG scale-space (σ ∈ [25,115], r = √2·σ\*) detects, on `image_133`,
  **3/3 coins including the low-contrast one Hough missed**, and 4/4 on `image_76`,
  with no false positives; it returns the radius for free, needs **no separate block-A
  smoothing** (the scale-space *is* the blur), and is immune to the σ ≈ 17 target noise
  (gone at coin scale). Generalizes well on a 12-image sample; one anomalous (greenish)
  image still to be checked.
- **Criterion to close.** Validate on a larger target sample + the single-coin
  reference images; check failure cases (touching coins, glare, the greenish image).
- **Status.** 🟢 LoG LEADING — strong on first tests; if adopted it supersedes §4.5
  and §4.6 and folds block A into block B.

## 5. Decision log

Append-only. Each row records a *closed* decision with the date, what was decided,
why, and the chapter of the summary that legitimises it.

| ID  | Date       | Decision | Why | Summary § |
|-----|------------|----------|-----|-----------|
| DEC-1 | 2026-05-26 | Drop absolute radius; use intra-image **relative** radius + per-image scale anchor. | 1€ (r=98) detected larger than 2€ (r=82) across reference shots though physically smaller ⇒ pixel radius is not metric. Coins in one image share distance/pose ⇒ radius *ratios* are metric. | §6.3 Hough + scale reasoning |
| DEC-2 | 2026-05-26 | No background scale anchor; rely on intra-image scale (§4.1 f / §4.2). | Target backgrounds are plain surfaces — no ruler/hand/A4. | — |
| DEC-3 | 2026-05-26 | Colour eliminated as a cross-set classification cue. | Target coins carry a strong blue specular cast (hue ≈ 100 vs ref ≈ 10); not a global WB (bg ≈ neutral), not invertible by a filter; copper/gold inseparable even by eye. | §3.4 (filters cannot undo it) |
| DEC-4 | 2026-05-26 | Block A pre-Canny smoothing = **Gaussian** (σ ≈ 2), not bilateral. | On noisy targets the bilateral preserves background noise-texture ⇒ Hough floods with spurious circles (13 vs 2 on a 3-coin image); Gaussian suppresses it and keeps the coin rim. Empirically overturned the theory-based bilateral lean. | §3.3 Gaussian, §4 Canny |

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

## 7. TODO — high-level milestones

Project-level milestones only. Day-to-day tasks are tracked in the session task
tracker, not here.

- [x] **M0** — Dataset in place; visual inspection done (2026-05-26). Closed the
      "blocked on dataset inspection" status on §4.1 / §4.2 / §4.3.
- [~] **M1** — Reference radii + colour measured; radius found non-metric and colour
      cast-corrupted (DEC-1/DEC-3). Per-denomination calibration to be finalized with
      the scale-grid fit.
- [ ] **M2** — Detection block tuned on the reference set; validated on at least
      three target images. Closes §4.5 and §4.6.
- [ ] **M3** — Classification block working on the reference set, evaluated
      out-of-sample on the eight reference images themselves (leave-one-out).
      Definitively closes §4.1.
- [ ] **M4** — End-to-end pipeline running on the full `target_set`,
      producing per-image `Partial Amount` and global `Total Amount`.
- [ ] **M5** — Notebook polish: prose, justifications anchored to the summary,
      discussion of failure cases and limits.
