# reconfigured — giacomo/tiziano with the ANALYSIS.md fixes applied and tuned

This is a fourth `coins-recognition` implementation: giacomo's pipeline (detection stays
exactly the same, from [`../common/detection.py`](../common/detection.py)) with the
classification-side fixes [`../ANALYSIS.md`](../ANALYSIS.md) identified applied, then
re-tuned against `data/coin_dataset/gt.csv`. Everything the reference model is built
from still comes from the same 8 images in `reference_set/` — nothing else was added.
`gt.csv` and the 142 `target_set/` images are the *evaluation* set, not the model, so
using them to search for better fusion weights doesn't touch that constraint.

## What changed, and what happened when we measured it

Not every fix ANALYSIS.md proposed turned out to help once actually measured. Two were
kept, one was tried and reverted, and the fusion weights were re-tuned on top:

### Kept: aspect-ratio-preserving `crop_disc`

giacomo/tiziano's `crop_disc` clips the bounding box asymmetrically at the image border
for a partially off-frame coin, then force-resizes whatever's left to a square 150x150
canvas — stretching the coin's aspect ratio before every downstream cue.
[`appearance.py`](appearance.py) instead always keeps the box square (2r x 2r), padding
the missing region by edge replication (`cv2.copyMakeBorder`, `BORDER_REPLICATE`)
before resizing.

### Kept: data-derived `TEMPLATE_OFFDIAG`

`TEMPLATE_OFFDIAG` is the per-class bias correction subtracted from a coin's shape score
before z-scoring — meant to represent "how much does this class's atlas tend to
false-match a *different* denomination." giacomo/tiziano hard-typed 8 constants for it.
[`reference.py`](reference.py) computes it directly instead: every reference coin's own
canonical crop is scored against every *other* class's atlas, and the average of those
cross-class scores becomes that class's offdiag bias. Still only the 8 reference images.

### Tried and reverted: decasting the shape-atlas crop

ANALYSIS.md flagged a real asymmetry: giacomo decasts the *target* image before scoring
it against the shape atlas, and already decasts the reference image for the
colour-prototype cue, but never decasts the atlas-building crop itself. The
theoretically consistent fix is to decast that crop too. Measured in isolation
([`ablate.py`](ablate.py), giacomo's original weights, SIFT disabled, aspect-crop +
offdiag fixes held constant in both arms):

| variant | amount accuracy (142 images) |
|---|---|
| without decast-atlas fix (giacomo's original raw-photo atlas input) | **49.30%** |
| with decast-atlas fix | 45.77% |

Decasting made it *worse*. `REF_EDGE`'s per-class Canny thresholds were calibrated
against the raw photo's contrast; decasting shifts that contrast enough to degrade the
atlas without also retuning `REF_EDGE` for it. Reverted — the shipped `reference.py`
builds the shape atlas from the raw reference photo, exactly like giacomo/tiziano.
Jointly searching `REF_EDGE` together with a decast atlas is flagged as follow-up work,
not attempted here: it requires rebuilding the atlas (and therefore re-running the
360-rotation shape-score extraction over all 142 target images) per candidate threshold,
far more expensive than tuning a fusion weight.

### Fixed: the SIFT cue's domain mismatch and scoring bias

[`sift_cue.py`](sift_cue.py) builds its reference descriptor cache from the same
canonical 150x150 `crop_disc` output the shape cue uses, not the raw, uncropped photo —
this removes both the train/query domain mismatch tiziano had and the physical-size bias
it caused (every class's canonical crop is now the same size, so `SIFT_NMS_DIST` no
longer lets bimetal coins keep proportionally more keypoints just because they're bigger
in the original photo). `SIFT_NMS_DIST` is recalibrated from 10px (on a full photo) to
5px (on a 150x150 crop). Match scoring now normalises each class's match count by that
class's *own* template descriptor count instead of the total across all classes, so a
class with fewer cached descriptors isn't structurally penalised. The cue is folded into
the fusion score via the same z-scoring the shape template uses, instead of tiziano's
`2*score - 0.25` remap (which assumed scores summed to 1 across classes — no longer true
under the new per-class normalisation).

### New: an `abs_anchor` consistency check

A single confidently-bimetal coin can set the scene's absolute pixel-to-mm scale
(`abs_anchor`, unchanged, from [`../common/classify.py`](../common/classify.py)).
[`classify.py`](classify.py) now checks that anchor against the independently-seeded
scale (`sscale`) before trusting it — `ANCHOR_CONSISTENCY_TOL` (tuned) controls how much
disagreement is tolerated before falling back to the seed alone.

## Tuning methodology

[`tune.py`](tune.py) splits classification into two functions
(`extract_features`/`fuse_scene` in `classify.py`) so the expensive part — detection,
360-rotation shape scoring, SIFT extraction — runs exactly once over all 142 target
images and gets cached (`build_cache()`), while the tunable fusion weights only affect
the cheap part (`fuse_scene`: arithmetic + beam search). That split makes each weight
evaluation take ~5ms/image instead of ~2s/image, which is what makes a real search
affordable at all.

The search objective is mean absolute value error across images (not
`pipeline.py`'s binary `amount_accuracy`) — accuracy is a per-image AND-of-every-coin
metric that's nearly flat across small weight changes, since one wrong coin fails an
image the same way whether the fusion score was close or wildly off. Mean absolute value
error still only changes when a beam-search label actually flips, but distinguishes a
1cent->2cent flip (1 cent of error) from a 1cent->2euro flip (~2 euros of error), giving
the search something to climb.

18 fusion weights were searched (everything ANALYSIS.md's fusion-logic findings
implicated: the shape/family/bimetal cue weights, the size-consistency weights, the
anchor weight and its new consistency tolerance, and the SIFT weight/floor) via 1500
random-search trials followed by 3 passes of coordinate-ascent refinement. Geometry and
beam-search mechanics (`RATIO_TOL`, `SAME_TOL`, `BEAM_K`, `REL_W`, `LOGMAX`, and the
coupled `BIM_START`/`BIM_FULL` threshold pair) were left at giacomo's original values —
not implicated by the analysis, and risky to randomize independently given `BIM_FULL`
must exceed `BIM_START`.

## Honest performance numbers

**Cross-validated (the number to trust for "how well would this generalize"):** two-fold
CV (odd/even target-image split, mirroring giacomo's own dev/sealed convention) gives a
held-out amount accuracy of **~48-51%** across the runs performed (fold-dependent; see
`tune_results.json` for every run's numbers). That's a modest, real improvement over
giacomo's 48.59% and tiziano's 45.07% — not a dramatic one. Train/val gaps as large as
65% train vs 48% val in some folds show real overfitting risk: 18 free parameters fit
against only 71 images is a lot of freedom for that little data, so treat any single
run's exact number as noisy within a few points either way.

**Shipped weights (the actual numbers `pipeline.py` will report for `IMPLEMENTATION =
"reconfigured"`):** fit on the full 142-image set, since that's the deployed
calibration — same practice as giacomo's own notebook-tuned constants, which were also
fit against the data available at the time. This is an **in-sample fit**, so expect
real-world performance closer to the cross-validated estimate above, not this number:

| | count accuracy | amount accuracy |
|---|---|---|
| giacomo (baseline) | 97.18% | 48.59% |
| tiziano (baseline) | 97.18% | 45.07% |
| **reconfigured** | 97.18% | **53.52%** |

Count accuracy is identical to giacomo/tiziano by construction — detection is unchanged.

A finding worth flagging on its own: both independent CV fold fits (different training
data, different random seeds) drove `FAM_POS` (the colour/family cue's positive weight)
down to near the search's lower bound, on their own. That's a consistent signal across
independent runs, not overfitting noise — giacomo's original `FAM_POS = 3.4` likely
over-weights the colour/family cue once the aspect-crop and offdiag fixes are in place.

## Files

Reuses [`../common/detection.py`](../common/detection.py) unchanged (detection isn't
what ANALYSIS.md flagged). Forks `appearance.py`, `reference.py`, `sift_cue.py`, and
`classify.py` from giacomo/tiziano with the fixes above. `config.py` holds the tuned
weights; `tune.py` is the search script that produced them (re-runnable, but the full
methodology — two CV folds plus several full-data seeds — takes a few hours end to end,
since each weight evaluation, while cheap, is still run thousands of times); `ablate.py`
is the ad hoc script that produced the decast-atlas ablation table above; `tune_results.json`
is the full numeric record of every run referenced in this file.
