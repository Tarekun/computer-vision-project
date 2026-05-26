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
- **Options.**
  (a) radius only;
  (b) HSV colour only, using copper / gold / bi-metallic priors;
  (c) NCC or ZNCC template matching against the reference set;
  (d) SIFT keypoints + Lowe ratio test against the reference set;
  (e) multi-cue voting combining a subset of the above.
- **Criterion to close.** After visual inspection of the reference set and a sample
  of the target set (M0), compute, on the reference set, the per-denomination radius
  distribution and the mean HSV in the disc interior. Pick the cue (or combination)
  with the cleanest separation on the ambiguous pairs `{1¢, 2¢, 5¢}` (all copper)
  and `{10¢, 20¢, 50¢}` (all gold).
- **Status.** 🟡 OPEN — blocked on dataset inspection.

### 4.2 — Scale strategy

- **Question.** How do we cope with the fact that pixel radius is not metric — it
  depends on the camera-to-subject distance, which the dataset does not document?
- **Options.**
  (a) absolute scale: assume distance is roughly constant across images and use
  absolute radius thresholds calibrated on the reference set;
  (b) intra-image relative scale: identify one coin by a non-size cue (e.g., the
  bi-metallic 1€ or 2€) and use its known diameter as a per-image scale anchor;
  (c) scale-invariant classification: avoid the problem by matching with a method
  that does not require knowing the absolute scale (SIFT, NCC over a scale pyramid).
- **Criterion to close.** After M0, judge from a 10+ image sample of the target set
  whether camera distance is visibly constant. If not, (a) is out and we choose
  between (b) and (c) based on §4.1.
- **Status.** 🟡 OPEN — blocked on dataset inspection.

### 4.3 — Background scale anchor

- **Question.** Does the `target_set` background contain a known-size reference
  object (a hand, a ruler, an A4 sheet, a printed grid)?
- **Options.**
  (a) yes — exploit it as a per-image scale calibration, which simplifies §4.2;
  (b) no — fall back to one of the options of §4.2.
- **Criterion to close.** Visual inspection of a 10+ image sample of the target set.
- **Status.** 🟡 OPEN — blocked on dataset inspection.

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
- **Criterion to close.** Validate the chosen method on the two bi-metallic reference
  images and on at least one target image containing a 1€ or a 2€. Accept the first
  option that succeeds on all three.
- **Status.** 🟢 LIKELY EASY — strong cue, expected to close quickly.

### 4.5 — Smoothing before Canny

- **Question.** Gaussian or bilateral smoothing as the pre-Canny step in block A?
- **Options.**
  (a) Gaussian (§3.3) — fast, separable, may slightly oversmooth the coin rim;
  (b) bilateral (§3.4) — preserves edges, roughly an order of magnitude slower.
- **Criterion to close.** Compare the Canny output on a handful of reference and
  target images. Default to Gaussian unless the edge of the coin is visibly degraded
  by it.
- **Status.** 🟡 OPEN.

### 4.6 — `cv2.HoughCircles` parameter tuning

- **Question.** Values for `dp`, `minDist`, `param1` (Canny high threshold),
  `param2` (accumulator threshold), `minRadius`, `maxRadius`?
- **Options.** Tune on the single-coin reference images to fix a baseline radius
  range; validate on a 5–10 image sample of the target set.
- **Criterion to close.** On the validation subset, zero missed coins and zero
  spurious circles. Hard cap: three rounds of parameter sweeps; if the budget is
  exhausted, revisit §4.5 instead of pushing further on params.
- **Status.** 🟡 OPEN.

## 5. Decision log

Append-only. Each row records a *closed* decision with the date, what was decided,
why, and the chapter of the summary that legitimises it.

| ID  | Date       | Decision | Why | Summary § |
|-----|------------|----------|-----|-----------|
| —   | —          | *(empty — no decisions closed yet)* | — | — |

## 6. Journal

Append-only, newest-on-bottom. Each entry is dated and describes what was tried,
what was observed, and what changes for the open decisions in §4.

> *(empty — fills up as work progresses)*

## 7. TODO — high-level milestones

Project-level milestones only. Day-to-day tasks are tracked in the session task
tracker, not here.

- [ ] **M0** — Dataset in place; first visual inspection of `reference_set` and
      a sample of `target_set`. Closes the "blocked on dataset inspection" status
      on §4.1 / §4.2 / §4.3.
- [ ] **M1** — Calibration baseline on `reference_set`: per-denomination radius
      distribution and mean HSV. Inputs for closing §4.1 and §4.2.
- [ ] **M2** — Detection block tuned on the reference set; validated on at least
      three target images. Closes §4.5 and §4.6.
- [ ] **M3** — Classification block working on the reference set, evaluated
      out-of-sample on the eight reference images themselves (leave-one-out).
      Definitively closes §4.1.
- [ ] **M4** — End-to-end pipeline running on the full `target_set`,
      producing per-image `Partial Amount` and global `Total Amount`.
- [ ] **M5** — Notebook polish: prose, justifications anchored to the summary,
      discussion of failure cases and limits.
