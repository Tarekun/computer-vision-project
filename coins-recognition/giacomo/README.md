# giacomo — coin detection & classification

Traditional-CV coin detection, geometric validation and classification, ported from
[`solution_2.ipynb`](solution_2.ipynb) into plain modules. The notebook remains the
design record (it explains the *why* behind every constant below); this package is a
functionally-identical, non-notebook copy of the same computation, reorganised into
one file per pipeline stage and parameterised instead of relying on module-level
globals computed at import time.

No behaviour was changed during the port — only redundant recomputation was removed
(the notebook built parts of the reference model in two separate loops, calling
`find_ref_coin` twice for six of the eight references; this port does it once per
reference and reuses the result). Everything else — every threshold, formula and
loop — is copied unchanged.

Detection (`detection.py`), the appearance cues (`appearance.py`), the base reference-model
builder (`reference.py`), and the generic scene-classification helpers (`zscore`, `pair_score`,
`abs_anchor`, `beam_assign`) are identical between this package and [`../tiziano/`](../tiziano/),
so they now live in [`../common/`](../common/) and are imported from there. This package only
keeps what's actually specific to it: its own cue weights (`config.py`) and `classify_scene`,
the per-coin cue fusion function (`classify.py`).

## Why this stands in for a GHT

`coins-recognition/pipeline.py` defines a generic `initialize_ght(reference_dir)` /
`evaluate(image, ght_models)` contract for the eventual Generalized Hough Transform
implementation. This solution does not build an R-table; instead its "model" is a
per-class **atlas** of Canny control points plus the gradient direction at each point,
matched against a target crop by trying every quantised rotation. It plays the same
role (a one-off model built once from `reference_set`, then reused for every target
image) so it satisfies the same interface. `giacomo/pipeline.py` exposes
`initialize`/`initialize_ght` and `evaluate` for exactly this purpose; the top-level
`pipeline.py` just imports and calls them.

## Pipeline stages

### 1. Detection (`common/detection.py`)

Coins are found in three passes of `cv2.HoughCircles`, used as a *permissive
candidate generator* rather than a final detector:

- Two strict passes (`param1=100`, two accumulator thresholds) at two minimum
  center-distances, deduplicated against each other.
- One weak-edge "rescue" pass (`param1=60`) that recovers coins with faint rims —
  but it is trusted only if it adds **at most 1** new candidate (`K_MAX`); more than
  that means it's amplifying background texture, and the whole rescue pass is
  dropped for that image (a scene-level parsimony rule).

Every candidate is then **validated**: 72 rays are cast from its center, and the
strongest *radial* Sobel gradient near the candidate radius (±14 px) is kept per ray.
A candidate needs at least 20 rays / 45% support to be accepted. The surviving edge
points go through two rounds of MAD-based outlier rejection and a Kása algebraic
least-squares circle refit, which becomes the coin's final `(cx, cy, r)`.

Three physical gates then clean up the survivors:

1. **Containment** — a circle whose center falls inside a larger circle is internal
   structure (a coin's relief or bimetal core), not a separate coin.
2. **Background similarity** — a disc whose median Lab colour sits within the
   scene background's own robust dispersion (median/MAD, relative not absolute) is
   texture, not a coin. The z-distance (`bg_z`) is kept on every surviving detection
   and reused later as a legibility signal by the classifier.
3. **Relative scale** — within one planar photo, radius *ratios* are physically
   constrained by the real euro diameter ratios (16.25–25.75 mm). Detections outside
   the largest internally-consistent radius band are dropped, but only when they are
   a minority of the scene (so a genuine one-coin or two-coin photo is never touched).

### 2. Appearance cues (`common/appearance.py`)

Every detected coin is cropped to a canonical 150×150 disc (this removes apparent
size — which depends on camera distance, not identity — from every appearance cue;
size is instead used at the scene level in stage 4). A fixed affine colour de-cast
(gain/offset per BGR channel, calibrated once on the references) neutralises the
scene's global colour cast, and crops are denoised with non-local means before
comparison — the opposite of detection, where every smoothing pass made the weak
rims *harder* to find.

Four cues come out of this module:

- **`shape_scores`** — the fraction of a class's reference control points whose
  gradient direction still agrees (within the polarity-invariant `cos 2Δθ ≥ τ`,
  τ = cos 60°) after rotating the crop, maximised over all 360 one-degree rotation
  hypotheses. This is the main identity signal.
- **`two_tone_family`** — is the coin's background-anchored tone closer to the
  copper or the gold prototype (both derived from the six mono references through
  the same processing chain)? Abstains outright when the core/ring chroma step
  looks bimetal.
- **`bimetal_step`** / **`ring_sign`** — the core-vs-ring chroma difference (a bimetal
  giveaway) and which of the two rings is gold vs silver (a weak 1€/2€ tie-breaker).
- **`first_pass`** — a cheap seed guess (shape + a soft colour/bimetal favour) used
  **only** to seed the scene's pixel-to-mm scale in stage 4; it is not a final answer,
  every coin is re-decided jointly afterwards.

### 3. Reference model (`common/reference.py`)

`build_reference_model(reference_dir)` reads the 8 reference images and produces the
per-class atlas (Canny control points + gradient directions, with per-class Canny
thresholds/NLM strengths and — for two references — a selective blur over relief that
was acting as a false attractor) plus the two colour prototypes described above. This
is the `initialize_ght` half of the contract: a model built once, reused for every
target image.

### 4. Scene-level classification (`classify.py`)

A coin is never labelled from its own cues alone. Per coin, the four appearance cues
are combined into a per-class score:

- the shape score, de-biased by each class's measured average off-class score and
  z-scored across classes,
- the family score (agree/disagree with the predicted copper/gold family),
- the bimetal score (agree/disagree with "this coin is bimetal", plus the weak
  1€/2€ ring-colour nudge),
- both appearance scores are damped when the coin's `bg_z` (from detection) says it's
  hard to read against the background.

Size is folded in through the *scene*, not the coin, because pixel radius only
becomes a diameter once you know the scene's mm-per-pixel scale:

- If any coin looks confidently bimetal, its radius fixes an **absolute** scale
  (it must be a 1€ or 2€, so `r/D` for the best-fitting one pins `mm-per-pixel` for
  the whole photo).
- Otherwise the scale is only seeded from the cheap first-pass guesses.

Every coin then gets a size term (distance from its diameter, under that scale, to
the nearest legal denomination) plus a pairwise term for every other coin in the
scene (radius *ratios* must match real diameter ratios). All of this — unary scores
plus pairwise terms — is solved **jointly** for the whole scene with a beam search
(width 256) over all label assignments, not coin-by-coin independently.

### 5. Entry point (`pipeline.py`)

- `initialize(reference_dir)` / `initialize_ght(reference_dir)` — builds the
  reference model (§3).
- `process_image(img, model)` — runs detection (§1) then classification (§4),
  returning one dict per coin: `cx`, `cy`, `r`, `support`, optional `bg_z`, `pred`
  (denomination string), `value` (euros).
- `evaluate(image, ght_models)` — the exact contract expected by the top-level
  `coins-recognition/pipeline.py`: same computation as `process_image`, reshaped into
  a list of `(denomination, center_x, center_y, radius)` tuples.
- Running `python -m giacomo.pipeline` (or `python giacomo/pipeline.py` from inside
  `coins-recognition/`) reproduces the notebook's original full-target-set console
  report (per-image coin list, partial and total amounts).

## Measured performance (from the notebook)

Against a hand-labelled positional ground truth (99/142 images, 227 coins):

- **Detection**: 0 misses, 1 false positive (itself a coin missing from the ground
  truth) — median center error 1.9 px, median radius error 1.4%.
- **Classification**: 0.734 coin accuracy overall (0.735 dev / 0.733 sealed, on a
  strict even/odd train/test split used to keep every calibration honest).

## Known limits

- Coins photographed at a strong angle appear elliptical; the circular detector model
  does not handle that (e.g. `image_85.jpg`).
- The background-similarity gate's threshold has a narrow measured margin (worst
  false positive z = 0.84 vs worst true coin z = 0.90).
- Remaining classification errors are concentrated *within* material families
  (20c/50c and 2c confusions) — on heavily degraded coins, shape and colour don't
  separate neighbouring denominations with these cues.

See `solution_2.ipynb` for the full derivation, measured ablation tables, and the
negative results (approaches tried and rejected).
