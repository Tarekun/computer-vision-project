# tiziano — coin detection & classification (SIFT variant)

Traditional-CV coin detection, geometric validation and classification, ported from
[`solution_2_T.ipynb`](solution_2_T.ipynb) into plain modules — the same kind of port as
[`../giacomo/`](../giacomo/). The two packages originally duplicated their common code
file-by-file so they could be diffed independently; that's since been consolidated into
[`../common/`](../common/) (detection, appearance cues, the base reference-model builder,
and the generic scene-classification helpers), leaving each package holding only what's
actually specific to it:

```
common/detection.py                                # shared — identical detector
common/appearance.py                                # shared — identical shape/colour cues
common/reference.py                                 # shared — base atlas + colour prototypes
common/classify.py                                  # shared — zscore/pair_score/beam_assign/abs_anchor
diff giacomo/config.py     tiziano/config.py         # cue weight recalibration + new SIFT constants
tiziano/reference.py                                 # wraps common.reference + builds a SIFT descriptor cache
diff giacomo/classify.py   tiziano/classify.py        # classify_scene: + SIFT cue folded into the unary score
                            tiziano/sift_cue.py       # new file, no giacomo equivalent
```

`solution_2_T.ipynb` is `solution_2.ipynb` (giacomo's notebook) with one addition: a fifth
classification cue built from SIFT keypoint matching against the reference photos, plus a
recalibration of two existing cue weights. Detection (§1–5) and the shape/colour/bimetal cues
(§6.1–6.3) are untouched — see [`../giacomo/README.md`](../giacomo/README.md) for those.

## What's different from giacomo

### 1. The SIFT cue (`sift_cue.py`)

For each reference coin, `calibrate_sift_templates` runs CLAHE + bilateral preprocessing then
SIFT keypoint detection **on the whole raw reference photo** (not the cropped canonical disc
the shape atlas uses) and keeps only spatially-separated keypoints (10 px minimum distance).
This is a deliberately different, cruder approach than the shape atlas's canonical-crop
pipeline: keypoints can land on the background surrounding the coin, not just the coin itself.
It is kept exactly as designed in the notebook — a genuine difference in approach, not a bug
fixed during porting.

A target crop is matched the same way (`sift_extract_with_nms`) and scored against every
class's cached descriptors with `cv2.BFMatcher` + Lowe's ratio test (`sift_match_scores`);
`sift_confidence` turns the raw per-class match counts into a 0–1 confidence that is high only
when one class's match count clearly dominates the runner-up.

### 2. Folding SIFT into the unary score (`classify.py`)

`classify_scene` extracts SIFT descriptors for every detected coin crop (grayscale, same
crop used by the shape cue) and adds a `sift_sc` term next to `template` (shape) and `fam_sc`
(colour family) inside the same `bg_z`-legibility-damped group:

```python
sift_norm = SIFT_FLOOR + (1 - SIFT_FLOOR) * coin["sift_conf"]
sift_sc = {c: W_SIFT * sift_norm * (2.0 * coin["sift_scores"][c] - 0.25) for c in CLASSES}
...
u = damp * (fam_sc[c] + sift_sc[c] + template[c]) + bim[c] - ...
```

The `2.0 * score - 0.25` remapping centers a class with no matches at all classes tied
(`score=0 -> -0.25`) below one with an average share (`score=1/8 -> 0`), so a coin with no SIFT
matches at all does not silently favour whichever class happens to be first in `CLASSES`.

### 3. Recalibrated size weights (`config.py`)

`W_ORDER` and `W_SAME` — the weights on the "does this radius match the legal diameter" and
"does this radius match its predicted class's other members" pairwise terms — are doubled
relative to giacomo (`0.5`/`0.2` vs `0.25`/`0.15`). Everything else in the frozen cue-weight
block (`TEMPLATE_W`, `FAM_POS`/`FAM_NEG`, the bimetal constants, `REL_W`, `BEAM_K`, `ANCHOR_W`,
...) is unchanged.

## Building the model (`reference.py`)

`build_reference_model(reference_dir)` returns everything `initialize_ght` needs to hand to
`evaluate`: the shape atlas and colour prototypes (identical to giacomo's, see its README),
plus `sift_cache` — the per-class SIFT descriptor arrays built once from `reference_set` and
reused for every target image, exactly mirroring how the notebook calls
`calibrate_sift_templates(SIFT_TEMPLATES)` once before its final full-dataset loop.

## Entry point (`pipeline.py`)

Same contract as `giacomo/pipeline.py`: `initialize`/`initialize_ght`, `process_image`,
`evaluate` (returning `(denomination, center_x, center_y, radius)` tuples), and a `_main()`
that reproduces the notebook's console report. Running `python -m tiziano.pipeline` from
`coins-recognition/` reproduces it directly.

## Porting notes (functional equivalence)

Nothing here changes the notebook's computation. The only structural differences from a
literal transcription:

- `calibrate_sift_templates` returns the descriptor cache instead of mutating a module-level
  `_SIFT_CACHE` global — necessary once the model is threaded through function parameters
  (as `giacomo`'s port already does) instead of relying on notebook cell execution order.
- The notebook's dead code was not ported: a placeholder `_SIFT_CACHE = {<path strings>}`
  assignment that gets fully overwritten by `calibrate_sift_templates` before any real use,
  a stray `demo = [...]` variable immediately overwritten by a later cell, and an unused
  `idx`/loop-index variable in `classify_scene` — none of these affect `process_image`'s
  output on any input.
- `FAMILY_OF` (the setup cell's alternate copper/nordic_gold/bimetal naming) is kept in
  `config.py` even though no function references it, since it is a genuine (if inert) part
  of this notebook's design and worth surfacing when diffing against giacomo's `config.py`.

Verified against a from-notebook transcription across the full 142-image target set:
denomination, center, radius and value identical to full float precision on every image.
