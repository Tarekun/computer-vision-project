# Review: where `reconfigured/` stands, and what's left to try

This is the status review requested before starting the `experimental/` iteration
loop. It has two parts: (1) an honest recap of `reconfigured/`'s fixes and numbers,
and (2) the prioritized backlog of ideas this directory will work through.

## 1. Current status: `reconfigured/` is the best baseline, with real caveats

`reconfigured/` (see [`../reconfigured/README.md`](../reconfigured/README.md) for the
full account) fixed three concrete bugs found in [`../ANALYSIS.md`](../ANALYSIS.md) —
an aspect-ratio-distorting crop for off-frame coins, a hand-typed `TEMPLATE_OFFDIAG`
replaced by a value derived from the 8 reference images, and a SIFT cue that was
matching in the wrong domain (raw photo vs. canonical crop) and scoring with a biased
normalization — then re-tuned 18 fusion weights via random search + coordinate ascent
against `data/coin_dataset/gt.csv`.

One proposed fix was tried and **reverted** after measurement: decasting the
shape-atlas-building crop for symmetry with the target-side preprocessing made things
worse (49.30% → 45.77% amount accuracy in isolation), because `REF_EDGE`'s per-class
Canny thresholds were implicitly calibrated against the raw photo's contrast.
`reconfigured/README.md` flags jointly retuning `REF_EDGE` together with a decast atlas
as follow-up work, not attempted there — that is backlog item 4 below.

**Numbers, both of them, not just the flattering one:**

| | count accuracy | amount accuracy |
|---|---|---|
| giacomo (baseline) | 97.18% | 48.59% |
| tiziano (baseline) | 97.18% | 45.07% |
| reconfigured (in-sample, shipped weights) | 97.18% | **53.52%** |
| reconfigured (2-fold cross-validated) | 97.18% | **~48–51%** |

The cross-validated number is the one to trust for "how well does this generalize" —
it's a modest, real improvement over giacomo, not a dramatic one. `tune_results.json`
shows train/val gaps as large as 65% (train) vs. 48% (val) in one fold: 18 free
parameters fit against 71 images per fold is a lot of freedom for that little data.
Any single accuracy number reported anywhere in this project should be read with that
noise band in mind — a few points either way is not a meaningful difference.

**Detection is not the bottleneck.** Count accuracy has been 97.18% across every
classical-CV variant tried (giacomo, tiziano, reconfigured) because they all share
`common/detection.py`. The gap that matters is entirely on the classification side —
this review and backlog are exclusively about the appearance/shape/colour cues and
their fusion, not the Hough-based coin localization.

## 2. Backlog for this iteration (priority order)

Scope constraints carried over from earlier work: **classical CV only** (no
pretrained deep embeddings, even though torch/torchvision happen to be installed in
this repo for an unrelated aircraft-classification assignment), and the reference
model — shape atlas, colour prototypes, any per-class statistic — may only be built
from the 8 images in `reference_set/`. `target_set/` + `gt.csv` remain fair game for
evaluation and for fitting fusion/model hyperparameters, exactly as `reconfigured/`
already relied on for its 18 tuned weights.

1. **Multi-exemplar robustness from the single reference photo** — flagged in
   `ANALYSIS.md` §3 as "likely the single biggest lever." With only one image per
   class, this can't mean more real exemplars; it means extracting the atlas and
   colour prototype from a small photometric-jitter ensemble of that one image
   (brightness/contrast/blur perturbations) instead of one raw instance, so the
   reference model is less sensitive to that specific photo's exact lighting/noise
   realization. Still built only from the 8 originals.
2. **Learned fusion classifier over the cue vector** — `ANALYSIS.md` §6's second
   bigger bet: replace (some of) the hand-set additive weights with a small learned
   model. Caveat to resolve during implementation: `gt.csv` has no per-coin labels
   (only per-image total + count), so any learned model needs either a weak/aggregate
   training signal (matching what `tune.py` already does for the hand-set weights) or
   a genuinely per-coin-labeled subset if one turns out to exist — being checked before
   this item starts.
3. **Two-stage "size-first" classification** — resolve cross-family denomination via
   the strong absolute/relative size cue first (euro diameters are well-separated
   across families), then use appearance cues only to break the two within-family ties
   (1c/2c, 20c/50c) that size cannot resolve.
4. **Joint `REF_EDGE` threshold retuning with the decast atlas** — revisit the
   reverted fix from `reconfigured/`, this time searching per-class Canny thresholds
   jointly with decasting, instead of assuming giacomo's original thresholds are still
   right for a different atlas input. Expensive: requires rebuilding the atlas and
   re-scoring shape matches per candidate.
5. **ORB/BRISK vs. SIFT** — now that the SIFT cue's domain mismatch and scoring bias
   are fixed, check whether a cheaper descriptor performs comparably; informative for
   the write-up even without an accuracy change.
6. **Cleanup** — drop the dead `FAMILY_OF` dict, reconsider the hardcoded `REF_BLUR`
   per-photo pixel patch, reconsider `SIFT_FLOOR`'s always-on floor vs. a stricter gate.
7. **CV protocol upgrade** — the existing 2-fold odd/even split showed real overfitting
   risk (large train/val gaps) on only 71 images per fold. This iteration switches to a
   steadier k-fold (k=4) for its own idea-acceptance decisions, to reduce noise in
   "kept vs. reverted" calls before they get made.

Each item will get an entry in [`EXPERIMENTS.md`](EXPERIMENTS.md) recording the
hypothesis, what was implemented, the measured cross-validated result against a
controlled baseline, and the keep/revert decision — the same evidence-based pattern
`reconfigured/README.md` used for its own reverted decast-atlas experiment. The final
architecture and honest numbers land in [`SOLUTION.md`](SOLUTION.md) once the backlog
is worked through.
