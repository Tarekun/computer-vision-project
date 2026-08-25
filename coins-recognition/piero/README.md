# piero — coin detection & classification

Traditional-CV coin detection, geometric validation and classification, ported from
[`solution.ipynb`](solution.ipynb) into plain modules — duplicated rather than shared with
[`../giacomo/`](../giacomo/) and [`../tiziano/`](../tiziano/) so all three can be diffed and
compared independently. This is a genuinely different design from the other two solutions, not
a variant of the same one: its own Hough tuning, its own classifier (`S11.3-bis`), and a
from-scratch Generalized Hough Transform edge-voting model that the notebook builds but never
actually wires into its own production pipeline.

## Pipeline stages

### 1. Detection (`detection.py`)

`hough_clean` is a precision-first `cv2.HoughCircles` pass (mid `minDist≈95`) tuned for ~zero
background false positives — no multi-pass/rescue scheme like giacomo/tiziano use. It is
followed by a single cheap filter, `rim_score`: the fraction of a candidate circle's perimeter
that carries a strong *radially-aligned* gradient. A Hough circle fitted to the gap or shadow
between two touching coins lacks a complete rim and is dropped (`rim_thr=0.42`). This is the
detector actually used by the notebook's final results loop.

**Documented limit**: coins with soft/worn rims (~23% of images, per the notebook's own
count) score below the clean Hough accumulator and are never detected at all — a known,
declared gap rather than a bug.

### 2. The unused GHT edge-voting model (`ght_edges.py`)

Section 2 of the notebook builds an actual Generalized-Hough-style model: for each reference
coin, the `GHT_N=250` strongest-gradient edge points (with orientation, mod 180°) are kept on a
size-normalized 160×160 disc; matching a target coin means rotating the model's points through
every 12° step and counting how many land on a same-orientation edge in the target (an
orientation-binned, dilated boolean mask per bin — `ght_vote`). A companion recall-first
detector (`overshoot`) and a GHT-confidence gate (`weak_rim_fallback`) were built to recover
weak-rim coins this way.

**Neither is used by the notebook's own final pipeline.** `detect_coins` never calls
`weak_rim_fallback`, and the Section-4 classifier never references the GHT model at all — the
notebook says so explicitly ("NOT used in production ... its precision collapses ... kept for
reference"). This module is ported for completeness and so it can be diffed against the other
solutions' approaches to shape matching, but `pipeline.py`'s `initialize_ght`/`evaluate` never
call anything in it, exactly matching the notebook's own behaviour.

### 3. Classification — S11.3-bis (`appearance.py`, `reference.py`, `classify.py`)

Per coin, three things are combined, independently per coin (no per-scene beam search over all
coins jointly, unlike giacomo/tiziano):

- **Shape match-%** (`scores3b`) — same idea as giacomo's shape atlas (per-class Canny control
  points + gradient direction, rotation-searched, polarity-invariant `cos 2Δθ`), but with
  per-class manual Canny thresholds and a **localized blur applied to the reference disc**
  before the atlas is built (`apply_blur3b`) — a region-specific fix (e.g. blurring the
  variable-position core of the 5c/1€ references) rather than giacomo's whole-relief blur.
- **Material/bi-metal colour favour** (`favour_11`) — a coin's Lab a*/b* signature is compared
  to copper/gold/bimetal group centroids built from the raw references; a chroma-gradient ring
  strength (`chroma_11`) separately flags bimetal coins, softly favouring 1€/2€ classes.
- **Relative size re-rank** — same idea as giacomo's soft size terms (diameter-order consistency
  + same-predicted-class consistency), but simpler: no absolute bimetal scale anchor, no joint
  beam search — each coin's score is just adjusted independently against a scene scale seeded
  from the median first-pass prediction.

`build_reference_model(reference_dir)` (in `reference.py`) builds the two things this needs: the
material colour atlas and the per-class blurred edge atlas — this package's `initialize_ght`
stand-in.

### 4. Entry point (`pipeline.py`)

Same contract as the other two: `initialize`/`initialize_ght`, `process_image`, `evaluate`
(returning `(denomination, center_x, center_y, radius)` tuples), and `_main()` reproducing the
notebook's console report. Running `python -m piero.pipeline` from `coins-recognition/`
reproduces it directly.

## Honesty notes (from the notebook)

- Detection and the bimetal 1€/2€ flag are solid (chroma-ring separates the references cleanly).
- Per-denomination classification of the six *monometallic* coins is **approximate** — the
  notebook is explicit that the blue specular cast, wear, and low resolution defeat every
  appearance matcher tried, and reports this as a justified partial result rather than
  overclaiming accuracy.

## Porting notes (functional equivalence)

- All identifier names/suffixes (`11`, `3b`) are kept as in the notebook — Section 4 is
  explicitly titled "SELF-CONTAINED" and deliberately redefines its own copy of helpers
  independent of Section 2's GHT code; this port preserves that separation (`appearance.py`'s
  `href_11` vs `ght_edges.py`'s own reference-locating Hough call are two independently-ported
  copies of the same snippet, matching the notebook's own duplication).
- `favour_11`'s `r`/`r_scene_max` parameters and `classify_scene3b`'s `r_max` local were dead
  (never read in either function body) and were dropped — the model built from reference_set
  is otherwise an unmodified, function-for-function port.
- `edgesfor3b`'s adaptive-Canny fallback (`autocanny_11`/`adaptive_11`) is kept, though given
  every class in `EDGE_PARAMS3b` specifies manual `low`/`high` thresholds, that branch is never
  actually exercised while building this atlas (`mode` is `"manual"` for all 8 classes) — kept
  for fidelity rather than proving its removal is safe.

Verified against a from-notebook transcription across the full 142-image target set: detected
circles and predicted denominations identical on every image.
