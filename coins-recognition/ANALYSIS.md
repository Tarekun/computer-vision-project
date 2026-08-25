# Accuracy analysis: giacomo & tiziano

This is a report, not a change log — nothing in `giacomo/` or `tiziano/`'s classification/fusion
logic was touched to produce it. It exists to answer three questions about the `coins-recognition`
benchmark: why coin-*count* accuracy (97.18% for both) and coin-*amount* accuracy (well under 50%
for both) are so far apart, why adding a SIFT cue in `tiziano` made amount accuracy *worse* rather
than better, and what's worth trying to close that gap. `piero` is a different design entirely and
is out of scope here.

**Follow-up:** [`reconfigured/`](reconfigured/) is a fourth implementation that actually applies a
subset of the recommendations below and measures the result — including one recommendation
(the decast/atlas preprocessing fix) that was tried and empirically reverted because it measured
*worse*, not better. See [`reconfigured/README.md`](reconfigured/README.md) for what was kept,
what was reverted and why, the weight-tuning methodology, and an honest (cross-validated, not just
in-sample) performance estimate: roughly 48-51% amount accuracy generalizes, with an in-sample
fit reaching 53.52% on the full 142-image set giacomo/tiziano were also measured against.

**Second follow-up:** [`experimental/`](experimental/) picks up where `reconfigured/` left off —
a further design/implement/test loop over 6 more ideas (a self-trained learned fusion classifier,
size-first two-stage classification, joint `REF_EDGE`+decast-atlas retuning, ORB vs. SIFT, a
multi-exemplar reference atlas, and two hand-tuned-patch cleanup checks), evaluated under a
stricter 4-fold cross-validation than `reconfigured/`'s own 2-fold estimate. None beat
`reconfigured/`'s architecture, so nothing changed in the shipped configuration — but the stricter
protocol itself revised the honest generalization estimate further down, to ~48.6%. See
[`experimental/REVIEW.md`](experimental/REVIEW.md) for the backlog, [`experimental/EXPERIMENTS.md`](experimental/EXPERIMENTS.md)
for every idea's measured result, and [`experimental/SOLUTION.md`](experimental/SOLUTION.md) for
the full write-up.

## 1. Why count accuracy and amount accuracy diverge

`pipeline.py` tracks two independent checks per image, against `data/coin_dataset/gt.csv`:

```python
count_ok = detected_count == gt["coins"]
amount_ok = abs(detected_total - gt["total"]) < 1e-6
```

`count_ok` only needs the *number* of detected coins to match. `amount_ok` needs every single coin
in the image to be both detected **and** correctly denominated — one wrong label anywhere in a
multi-coin photo fails the whole image's amount check, even if every other coin in it was read
correctly. These are very different bars: 97.18% count accuracy says detection is excellent (both
packages miss/hallucinate a coin in only 4 of 142 images), while <50% amount accuracy says the
*joint*, all-coins-correct classification job is hard.

This isn't actually a contradiction with giacomo's own README, which reports **0.734 per-coin**
classification accuracy (0.735 dev / 0.733 sealed) on a hand-labelled 99-image/227-coin subset —
a *per-coin* metric, not a *per-image-all-coins-correct* one. A 73% per-coin accuracy compounds
quickly: if a photo has 4 coins and each is independently right 73% of the time, the chance all 4
are correct is only ~0.73⁴ ≈ 28%. The two numbers you're seeing (73.4% vs <50%) are consistent
with each other once you account for how many coins are typically in a target photo — they aren't
two conflicting measurements of the same thing.

**Takeaway:** the detector is not the bottleneck. All of the leverage for improving the benchmark
is in per-coin classification accuracy — even a modest per-coin accuracy gain compounds into a much
bigger amount-accuracy gain on multi-coin images.

## 2. Root cause of tiziano's SIFT regression

Tiziano adds a fifth cue (SIFT keypoint matching against the reference photos) on top of
giacomo's four cues, and independently doubles two existing weights (`W_ORDER`/`W_SAME`,
0.25/0.15 → 0.5/0.2). The result is *slightly worse* amount accuracy than giacomo (45.07% vs
48.59%). Tracing through `tiziano/sift_cue.py` and `tiziano/classify.py`, plus empirically running
the actual SIFT calibration/matching functions against the real reference images, points to a
fairly clear-cut chain of misconfiguration:

**a. Domain mismatch between the SIFT cache and every other cue.** `calibrate_sift_templates`
builds its reference descriptor cache from the *raw, uncropped* reference photo
(`cv2.imread(path, cv2.IMREAD_GRAYSCALE)`, no `find_ref_coin`/`crop_disc`). Every other cue —
the shape atlas, the colour prototypes, and the target-side SIFT extraction itself
(`sift_extract_with_nms`, called on the 150×150 canonical `crop_disc` output) — operates on the
tightly cropped, scale-normalized coin disc. The README calls this deliberate ("a genuine
difference in approach, not a bug fixed during porting"), but the effect below suggests it's
costing more than it's worth.

**b. That domain mismatch produces a large, class-dependent descriptor-count imbalance.**
`SIFT_NMS_DIST = 10` is an *absolute pixel* radius, applied to the full raw photo (e.g. `1cent.jpg`
is 960×720). Physically larger coins (1€/2€, ~166–192px diameter in-frame) occupy more image area
and keep proportionally more spatially-separated strong keypoints than small coins (~114–152px).
Actually running `calibrate_sift_templates` against `data/coin_dataset/reference_set` gives:

```
1cent   :  6 descriptors (after NMS)
2cent   :  3 descriptors (after NMS)
5cent   : 11 descriptors (after NMS)
10cent  :  4 descriptors (after NMS)
20cent  :  9 descriptors (after NMS)
50cent  :  9 descriptors (after NMS)
1euro   : 54 descriptors (after NMS)
2euro   : 47 descriptors (after NMS)
```

An ~18× spread (3 to 54), with 1€/2€ far ahead of everything else — driven purely by in-frame
physical size, not visual distinctiveness.

**c. Match scoring compounds the imbalance instead of correcting for it.** `sift_match_scores`
calls `bf_matcher.knnMatch(tmpl_des, des_query, k=2)` with the *template* as the matcher's query
set, so a class's achievable "good match" count is capped by its own cached descriptor count.
Scores are then normalized by the *total* good matches across all 8 classes
(`scores[c] = raw_counts[c] / total`), so classes with more cached descriptors structurally win a
bigger share of the match budget regardless of true visual similarity — 1€/2€ start every
comparison with a large built-in advantage.

**d. Empirically, the SIFT cue is unreliable even in its best case.** Querying each class's cache
with a crop taken from *the same photo the cache was built from* (the best possible case for a
correct match) still mispredicts 5 of 8 classes, three with high self-reported confidence:

```
true=1cent    pred_by_sift=1euro   conf=1.00   (raw: {'1euro': 2, others: 0})
true=2cent    — zero SIFT keypoints extracted at all
true=5cent    pred_by_sift=1euro   conf=0.46
true=10cent   pred_by_sift=2euro   conf=0.67
true=50cent   pred_by_sift=1euro   conf=0.60
true=20cent   pred_by_sift=20cent  conf=0.00   (correct, but a tie — not a real signal)
true=1euro    pred_by_sift=1euro   conf=0.64
true=2euro    pred_by_sift=2euro   conf=0.71
```

Only the two large-descriptor-count classes (and 20cent, at zero confidence) self-match correctly.

**e. The fusion weighting doesn't discount this.** `sift_sc = W_SIFT * sift_norm * (2*score - 0.25)`
with `W_SIFT = 1` and `sift_norm = SIFT_FLOOR + (1-SIFT_FLOOR)*sift_conf` (`SIFT_FLOOR = 0.3`, so
the cue is never fully switched off, even at zero confidence). A confidently-wrong signal like
`1cent → 1euro` (conf 1.00) contributes `sift_sc[1euro] = +1.75` and `sift_sc[1cent] = -0.25` into
the same damped sum that carries the shape template (`TEMPLATE_W=0.5 * z-score`, typically ±0.5–1.5)
— large enough to plausibly flip or bias the beam search's argmax toward 1€/2€, concentrated on
exactly the small coins the cache serves worst.

**f. Confounded by an unrelated, simultaneous change.** `W_ORDER`/`W_SAME` were independently
doubled in the same variant, with no isolated ablation in the available artifacts — so it isn't
possible from this analysis alone to say how much of the net regression is the SIFT cue vs. the
size-weight change. Isolating them (SIFT fix alone, size-weight change alone) before drawing final
conclusions is one of the "quick wins" below.

**Bottom line:** this reads as a genuine misconfiguration — an unnormalized descriptor-count budget
plus a train/query domain mismatch — not evidence that a SIFT-based cue can't work here.

## 3. Giacomo's classification weaknesses (inherited unchanged by tiziano)

These are the likely drivers of the shared <50%/73.4% ceiling, independent of the SIFT issue above:

- **Single reference image per class.** `build_reference_model` reads exactly one `.jpg` per
  denomination. All shape/colour "learning" is a one-shot exemplar match, not a distribution —
  inherently brittle to the lighting, wear, and angle variation actually present across 142 target
  photos. Likely the single biggest lever available.
- **Reference vs. target preprocessing asymmetry, feeding the primary shape signal.** Colour
  de-cast (`decast`) is applied to the *target* image before every shape comparison
  (`classify_scene`'s `dcx = decast(img)`), but is **never** applied to the reference image before
  the shape atlas is built (`reference.py`'s atlas construction uses raw `ref_bgr`). Denoising
  strength is also asymmetric: targets always use a fixed `TGT_NLM=4`, references use a per-class
  hand-tuned `REF_NLM` ranging 2–7. Since `shape_scores` compares gradient *directions* derived from
  grayscale on both sides, this means the atlas and the query are built from differently
  colour-corrected, differently smoothed inputs.
- **Per-class Canny thresholds vary by an order of magnitude** (`REF_EDGE`: `1cent=(5,30)` vs
  `20cent=(60,140)`), giving each class's atlas a very different edge-point density/reliability.
  `TEMPLATE_OFFDIAG` (a per-class bias-correction term, values ~0.42–0.52) looks like a **post-hoc
  patch** for this known per-class bias rather than a fix at the source.
- **`crop_disc` anisotropically stretches partially-off-frame coins.** A coin near the image border
  gets an asymmetrically clipped, non-square bounding box, which is then unconditionally force-
  resized to a square 150×150 canvas — stretching its aspect ratio before every downstream cue.
  There's no aspect-ratio-preserving fallback; the only special-case handling in `classify.py`
  triggers on a fully *empty* crop, not a merely non-square one.
- **`shape_scores` compares gradient direction at every reference control point regardless of local
  edge strength**, including flat/textureless regions of the target where "direction" is closer to
  noise. Because the match is `max` over all 360 rotation hypotheses, this creates room for
  spuriously well-aligned rotations on noise to inflate a class's score.
- **A single "confirmed bimetal" coin can anchor the absolute scale for the entire scene** with a
  large weight (`ANCHOR_W=4.0`, `abs_anchor` in `classify.py`), with no consistency check against
  the independently-computed `sscale` seed — one bad bimetal call can distort every other coin's
  size term in a multi-coin photo.
- **~30 hand-tuned scalar weights in one flat `config.py`**, calibrated once against a specific
  dev/test split, with no learning or cross-validation mechanism.
- **`REF_BLUR`'s hardcoded pixel-circle coordinates** (e.g. `"5cent": (10, (107, 75, 30))`) are tied
  to the exact relief layout of two specific reference photos — fragile if those images are ever
  replaced, and not really a general mechanism despite the name.

## 4. What to reconfigure (lower-risk, testable in isolation)

- Fix the `decast`/NLM preprocessing asymmetry between reference and target before shape scoring.
- Fix the SIFT cache to build from the same canonical `crop_disc` output the shape cue uses
  (removes both the domain mismatch and the descriptor-count size bias, since every class would
  then be captured at the same physical scale), and/or normalize `sift_match_scores` by each
  class's own template descriptor count rather than the global total across classes.
- Add an aspect-ratio-preserving fallback to `crop_disc` for coins clipped by the image border.
- Add a sanity check on `abs_anchor`: reject (or down-weight) the absolute-scale anchor if it
  diverges too far from the independently-seeded `sscale`.
- Re-test tiziano's `W_ORDER`/`W_SAME` doubling in isolation from the SIFT fix, to know how much of
  the net regression each change is actually responsible for.

## 5. What to drop or reconsider

- The dead `FAMILY_OF` dict in `tiziano/config.py` — confirmed unreferenced by any function.
- The `REF_BLUR` per-photo pixel patch — a single-purpose fix tied to two specific images' exact
  content, not a general mechanism; worth replacing with something that doesn't hardcode pixel
  coordinates, or dropping if the underlying relief issue can be addressed upstream.
- Whether `SIFT_FLOOR` (currently 0.3, meaning the SIFT cue is never fully switched off even at
  zero confidence) should exist at all, given how unreliable the cue is on small coins — a stricter
  confidence gate (only trust SIFT above some higher bar, otherwise contribute nothing) is a smaller
  change than removing the cue outright.

## 6. What to try next (bigger bets)

- **Systematically re-tune the fusion weights** against the existing 99-image/227-coin labelled dev
  subset (grid/random search or coordinate ascent) instead of the current manual, notebook-era
  values — this directly targets the metric that matters and removes guesswork from ~30 free
  parameters.
- **Replace the hand-set additive linear fusion with a small learned classifier** (logistic
  regression or kNN) over the existing cue vector (shape scores per class, family cue, bimetal cue,
  size ratio), trained on the dev split and evaluated on the sealed split — likely to generalize
  better than hand-picked weights and removes most of the magic numbers in one move.
- **Consider ORB/BRISK as a cheaper alternative to SIFT** if, after fixing the domain-mismatch and
  scoring-normalization issues above, the cue still isn't informative enough to be worth its cost.
- **Lean harder on the physical-size cue** where it's genuinely strong (across denominations, since
  euro diameters are well-separated), while accepting it fundamentally can't resolve the
  documented *within-family* confusions (20c/50c, 1c/2c) that need appearance cues to break the tie.

## Priority punch list

**Quick wins** (try first, cheap to isolate and measure):
1. Fix tiziano's SIFT cache to use the canonical crop, not the raw photo.
2. Fix the giacomo `decast`/NLM asymmetry between reference and target.
3. Drop the dead `FAMILY_OF` dict; tighten or drop the `SIFT_FLOOR` cue-can't-fully-turn-off floor.
4. Isolate tiziano's `W_ORDER`/`W_SAME` change from the SIFT change to know what each is worth.

**Bigger bets** (more effort, likely bigger upside):
5. Re-tune the ~30 fusion weights against the labelled dev subset instead of manual values.
6. Replace the linear fusion with a small learned classifier over the same cue vector.
7. Add an aspect-ratio-preserving crop fallback and an `abs_anchor` consistency check.
