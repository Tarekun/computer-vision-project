# Experiments log

One entry per backlog item from [`REVIEW.md`](REVIEW.md), in the order they were
tried. Each entry records the hypothesis, what was implemented, the measured result
against a controlled baseline, and the keep/revert decision — the same
evidence-based pattern [`../reconfigured/README.md`](../reconfigured/README.md) used
for its own reverted decast-atlas experiment. Numbers here are the ones that decide
what ships in `experimental/config.py`/`classify.py`; the final combined result is
written up in [`SOLUTION.md`](SOLUTION.md).

All comparisons in this log use the item 7 protocol below (k=4 cross-validation)
unless stated otherwise — 2-fold CV (`reconfigured/`'s original approach) showed
large train/val gaps on only 71 images per fold, and a 4-fold split gives ~3x more
training data per fold while still reporting honest held-out numbers.

---

## Item 7: CV protocol upgrade (2-fold → 4-fold)

**Status: adopted, used to evaluate every other item below.**

`reconfigured/`'s 2-fold odd/even CV (`tune.py`'s original `cross_validate`) fit 18
parameters against only 71 training images per fold and showed train/val gaps as
large as 65% (train) vs. 48% (val) in one fold — a real overfitting signal, not just
noise. `experimental/tune.py`'s `cross_validate` now takes a `k` parameter (default
4): each fold trains on the other 3 (~107 images) and validates on the held-out ~35,
rotated over all 4 folds, and reports the min/max spread across folds alongside the
mean so a single lucky/unlucky split can't pass as a stable estimate.

**Day-0 baseline** (the unmodified `reconfigured/` fork, evaluated under the new
protocol with a lighter search budget than the final shipped-weight fit — 500 random
trials + 2 coordinate-ascent passes per fold instead of 1500/3 — to keep per-idea
screening affordable; see `day0_quick_cv4.json` for the full per-fold record):

| fold | train n | val n | val amount_acc |
|---|---|---|---|
| 0 | 107 | 35 | 42.86% |
| 1 | 106 | 36 | 58.33% |
| 2 | 106 | 36 | 61.11% |
| 3 | 107 | 35 | 51.43% |
| **mean** | | | **53.43%** (min 42.86%, max 61.11%) |

This is noticeably higher than `reconfigured/`'s own 2-fold estimate (~48-51%) — more
training data per fold (107 vs. 71) fits better, but the fold spread (42.86% to
61.11%) shows k=4 still has real variance on a 142-image dataset. **53.43% mean
held-out amount accuracy is this iteration's number to beat**, not `reconfigured/`'s
in-sample 53.52%, and not its 2-fold estimate either (different protocol, not
directly comparable). Every item below is measured against this same protocol
(500 trials / 2 passes / k=4, seed 0) unless noted.

---

## Item 1: multi-exemplar robustness from the single reference photo

**Status: tried and reverted — single-instance atlas kept.**

**Hypothesis:** `ANALYSIS.md` §3 flagged the shape atlas as a one-shot exemplar
match (built from a single raw read of the one reference photo per class) as
"likely the single biggest lever" — inherently brittle to that specific instance's
lighting/noise. Building it instead from a small photometric-jitter ensemble
(brightness ±12, contrast ±15%, a slight blur — [`config.py`](config.py)'s
`REF_JITTERS`) of the SAME single photo, voting on which Canny control points
survive across jitters (`ATLAS_VOTE_FRAC`), should make the atlas represent the
coin's edges rather than that one photo's noise. Still built from nothing but the
8 reference images.

**Implementation:** [`appearance.py`](appearance.py)'s `photometric_jitter` +
[`reference.py`](reference.py)'s `_jittered_atlas` (control point kept if present in
≥50% of 6 jittered variants; direction = circular mean). Gated by
`C.JITTERED_ATLAS` so the reverted code stays available for reference.

**Measured result:** full k=4 CV (500 trials/2 passes per fold, same protocol as
the day-0 baseline):

| variant | mean held-out amount_acc | fold spread |
|---|---|---|
| day-0 baseline (single-instance atlas) | **53.43%** | 42.86% – 61.11% |
| jittered-atlas ensemble | 43.55% | 31.43% – 55.56% |

**Decision:** reverted — a clear, large regression (-9.9 points), not noise (every
fold got worse, not just the mean). Most likely explanation: `REF_EDGE`'s per-class
Canny thresholds (like the decast-atlas finding in `reconfigured/`) are implicitly
calibrated against the exact contrast/noise profile of the single raw photo; voting
across brightness/contrast/blur variants systematically shifts which edges clear
those fixed thresholds, in a direction the thresholds weren't tuned for. This is the
same lesson `reconfigured/README.md` already drew from the decast-atlas experiment,
now confirmed a second time on a different perturbation: `REF_EDGE` is a load-bearing,
narrowly-calibrated constant, and changing the atlas's input image (in any way)
without rethinking `REF_EDGE` alongside it tends to hurt rather than help. `Item 4`
(below) is precisely the attempt to retune `REF_EDGE` jointly with an atlas-input
change instead of treating it as fixed.

---

## Item 5: ORB vs. SIFT

**Status: rejected — SIFT kept.**

**Hypothesis:** now that the SIFT cue's domain mismatch and scoring bias are fixed
(`reconfigured/sift_cue.py`), a cheaper binary descriptor (ORB) might perform
comparably, since the earlier problems weren't really about SIFT itself.

**Implementation:** [`orb_cue.py`](orb_cue.py) mirrors `sift_cue.py`'s design exactly
(reference cache from the canonical crop, per-class match-count normalisation by that
class's own template size) but with `cv2.ORB_create` and Hamming-distance matching
instead of SIFT + L2. [`orb_compare.py`](orb_compare.py) swaps ORB's scores into the
same `coin["sift_scores"]`/`coin["sift_conf"]` slots SIFT normally fills, so
`classify.fuse_scene` runs completely unmodified — the descriptor is the only
variable that changes.

**Measured result:** substituting ORB for SIFT under the *same* SIFT-tuned fusion
weights (`W_SIFT`/`SIFT_FLOOR` from `reconfigured/config.py`, unchanged):

| descriptor | count accuracy | amount accuracy | mean abs error |
|---|---|---|---|
| SIFT (shipped) | 97.18% | 53.52% | 0.157 |
| ORB (substituted, same weights) | 97.18% | **41.55%** | 0.287 |

**Decision:** rejected. A ~12-point drop under weights that already give the
descriptor cue every benefit of its current tuning is a large enough gap that
retuning specifically for ORB is very unlikely to close it — ORB's binary
descriptors are evidently a weaker discriminator for these coins' relief/texture
than SIFT's gradient-histogram descriptors at this crop resolution (150x150).
SIFT stays; no further ORB tuning attempted.

---

## Item 6a: SIFT_FLOOR soft floor vs. a hard confidence gate

**Status: rejected — soft `SIFT_FLOOR` kept.**

**Hypothesis:** `SIFT_FLOOR` keeps the SIFT cue partially "on" even at zero
top1-vs-top2 confidence (`sift_norm = SIFT_FLOOR + (1-SIFT_FLOOR)*conf`); a hard
gate (`SIFT_HARD_GATE`, `classify.py`) that switches the cue fully off below a
confidence threshold might be a cleaner, equally effective design.

**Implementation:** reused each of the day-0 protocol's 4 fold-specific hand-tuned
weight vectors (already fit with the soft floor) unchanged, and re-evaluated each
fold's held-out val set with `SIFT_HARD_GATE=True` at 5 threshold candidates
(0.1-0.5), instead of running a fresh search — a fair enough comparison given how
small the effect turned out to be.

**Measured result:**

| variant | mean held-out amount_acc |
|---|---|
| soft floor (day-0 baseline, item 7) | 53.43% |
| hard gate, threshold=0.1 | 49.88% |
| hard gate, threshold=0.2 | 52.00% |
| hard gate, threshold=0.3 | 53.41% |
| hard gate, threshold=0.4 | 51.29% |
| hard gate, threshold=0.5 | 50.58% |

**Decision:** rejected. The best hard-gate threshold (0.3) is statistically
indistinguishable from the soft floor (53.41% vs. 53.43%) and every other threshold
is worse — no evidence a hard gate is either better or meaningfully simpler in
practice. Not worth the extra parameter; `SIFT_FLOOR`'s soft floor stays.

---

## Item 3: size-first two-stage classification

**Status: rejected — joint single-stage fusion kept.**

**Hypothesis:** the physical-size cue is strong and well-separated across the
copper/gold/bimetal families; restricting a confidently-resolved coin's beam-search
candidates to its own family's 2-3 classes (instead of all 8) should stop a
cross-family shape false-match from overriding a solid colour/bimetal read.

**Implementation:** [`classify.py`](classify.py)'s `beam_assign_restricted` +
`FAMILY_CLASSES`, gated by `C.TWO_STAGE_SIZE_FIRST` with two new tunable gates
(`FAMILY_GATE_CONF`, `FAMILY_GATE_BIM`). [`two_stage_search.py`](two_stage_search.py)
adds both gates to the standard 18-parameter search (20 total) and runs the full k=4
CV with `TWO_STAGE_SIZE_FIRST=True` fixed for the whole run.

**Measured result:**

| variant | mean held-out amount_acc |
|---|---|
| day-0 baseline (single-stage, all 8 classes always) | **53.43%** |
| two-stage, gates jointly tuned | 49.25% |

**Decision:** rejected (-4.2 points). The likely explanation: restricting candidates
removes exactly the safety net that lets the pairwise size-consistency term in the
existing joint beam search correct a wrong family call using the OTHER coins in the
same scene — hard-gating a family decision per-coin, before the joint search runs,
throws that cross-coin correction away whenever the per-coin family/bimetal cue is
(confidently but) wrong. The existing single joint 8-class beam search already
encodes "lean on size," just without discarding recovery options; leaning harder by
restricting candidates outright is a net loss here.

---

## Item 2: learned fusion classifier over the cue vector

**Status: tie — not adopted as the shipped default, but a notable finding.**

**Hypothesis:** `ANALYSIS.md` §6's second bigger bet: replace the hand-set/searched
additive fusion weights with a small learned model over the same cue vector.

**Caveat resolved before implementing:** confirmed (see conversation) that no
per-coin-labeled dataset exists anywhere in this repo — `gt.csv` has only
per-image `{total, coins}`, and giacomo's own README's "0.734 per-coin accuracy on
a hand-labelled 99-image/227-coin split" was a one-off manual notebook measurement,
never saved as a label file. [`learned_fusion.py`](learned_fusion.py) instead
self-trains: it pseudo-labels a TRAIN image's coins only when a baseline fusion's
whole-image prediction exactly matches `gt.csv` (count AND total) AND that
(count, total) pair has only ONE feasible multiset of the 8 denominations
(`feasible_multisets`) — so the predicted label *set* must be the true one. This
does not rule out a within-image swap between two coins that both belong to that
one correct multiset; a documented, unresolved limitation of this weak-supervision
signal (see the module docstring). ~50-60% of train images passed this filter per
fold (49-64 of ~107).

**Implementation:** a shared 7-weight linear softmax (multinomial logistic
regression, `scipy.optimize.minimize`, L-BFGS-B) over the same features `fuse_scene`
already computes (z-scored shape template, family/bimetal indicators, z-scored SIFT,
the three log-distance size penalties) replaces `config.py`'s ~11 independently
hand-set/searched weights for the unary score — the beam search's pairwise mechanics
are untouched. Fit per fold on that fold's own pseudo-labels (generated by that
fold's own hand-tuned weights, so VAL is never touched by either model).

**Measured result:** k=4 CV, same folds as every other item:

| | fold 0 | fold 1 | fold 2 | fold 3 | mean |
|---|---|---|---|---|---|
| hand-tuned/searched (day-0 baseline) | 42.86% | 58.33% | 61.11% | 51.43% | **53.43%** |
| learned (self-trained) | 48.57% | 63.89% | 50.00% | 51.43% | **53.47%** |

**Decision:** not adopted as the shipped default — the means are statistically
indistinguishable (+0.04 points) and the per-fold swings in both directions
(learned +5.7 and +5.6 points on two folds, -11.1 on another) show this is noisy,
not a reliable win. **But it's a genuinely notable finding anyway**: a 7-parameter
linear model, fit from nothing but weak self-training on ambiguity-filtered
pseudo-labels, matches a far more heavily hand-engineered and directly-searched
~11-weight fusion. That says the *feature set* (shape/family/bimetal/SIFT/size cues)
is carrying the real discriminative power here, and the specific fusion mechanism
on top of it matters less than which cues are included — a useful thing to know
before investing more effort in further hand-tuning weights rather than cues. Code
kept in the repo as a documented alternative; `classify.py`'s hand/searched
`fuse_scene` stays the shipped default for simplicity (no self-training pipeline or
scipy dependency needed at inference time).

---

## Item 4: joint REF_EDGE threshold + decast-atlas retuning

**Status: rejected a second time, more decisively — no-decast atlas kept.**

**Hypothesis:** `reconfigured/README.md` reverted decasting the atlas-building crop
because `REF_EDGE`'s per-class Canny thresholds were calibrated against the raw
photo's contrast — retuning `REF_EDGE` jointly with the decast should let the atlas
benefit from the theoretically-more-consistent decasted input without that penalty.

**Implementation:** two stages. (1) [`ref_edge_search.py`](ref_edge_search.py): an
isolated search (giacomo's original fusion weights, SIFT off — same methodology as
`reconfigured/ablate.py`) over `decast_atlas` x 10 randomly perturbed `REF_EDGE`
dicts (±40% per threshold). This reproduced `ablate.py`'s no-decast control exactly
(49.30%) and found one candidate ("trial 8") edging out the control on the search's
own objective (mean_abs_err 0.2723 vs. 0.2744) — see `ref_edge_search_results.json`.
(2) Took trial 8's thresholds (`config.py`'s `REF_EDGE_RETUNED`, gated by
`C.DECAST_ATLAS`) and ran the FULL pipeline (SIFT on, all cues) through the same
k=4 CV protocol as every other item, with fusion weights properly retuned for this
new atlas — the fair test the isolated ablation alone can't provide.

**Measured result:**

| variant | mean held-out amount_acc |
|---|---|
| day-0 baseline (no decast, original REF_EDGE) | **53.43%** |
| decast=True + REF_EDGE_RETUNED, weights retuned | 44.98% |

**Decision:** rejected, more decisively than the isolated ablation suggested. The
isolated giacomo-weights test's small apparent edge (0.2723 vs. 0.2744 mean_abs_err)
did not survive proper fusion-weight retuning under honest CV — a -8.5 point swing
once every other cue and weight is allowed to adapt around the new atlas. Two
lessons: (1) `reconfigured/README.md`'s original call to leave `REF_EDGE` alone was
right, confirmed a second, more expensive way; (2) a 10-trial random search over 16
coupled threshold parameters, scored only by an isolated single-weight-vector
ablation, is not enough signal to find a genuinely better joint configuration in a
space this size — the isolated screening test on its own would have been
misleading here without the full-pipeline follow-up.

---

## Item 6b: REF_BLUR hardcoded per-photo patch

**Status: rejected — REF_BLUR kept despite its fragility.**

**Hypothesis:** `ANALYSIS.md` §5 flagged `common/config.py`'s `REF_BLUR` (a
hand-placed Gaussian-blur patch at hardcoded pixel coordinates, applied to exactly
2 of the 8 reference discs — `5cent`, `2cent` — before atlas-building) as fragile:
tied to the exact content of two specific photos, liable to silently do nothing
useful (or something wrong) if those photos were ever replaced. Worth checking
whether it's still earning its keep.

**Implementation:** `C.DISABLE_REF_BLUR` (new flag, `reference.py`'s
`apply_ref_blur`) skips the patch entirely; full k=4 CV with everything else
unchanged (same REF_EDGE, no decast, weights retuned).

**Measured result:**

| variant | mean held-out amount_acc |
|---|---|
| day-0 baseline (REF_BLUR applied) | **53.43%** |
| REF_BLUR disabled | 51.37% |

**Decision:** rejected (-2.1 points) — despite being exactly the kind of
hand-placed, single-purpose patch `ANALYSIS.md` was right to flag as fragile, it is
measurably still helping on the current 8 reference photos. Kept as-is; the
fragility concern is valid as a maintenance note (if `reference_set/5cent.jpg` or
`2cent.jpg` are ever replaced with different photos, this patch's fixed pixel
coordinates would need re-placing or the effect could silently invert), but "fragile
and helping" isn't a reason to remove something with no replacement in hand — noted
in SOLUTION.md as remaining future work rather than acted on here.

---

## Summary: all 7 backlog items resolved

| item | idea | result |
|---|---|---|
| 1 | Multi-exemplar jittered atlas | rejected (43.55% vs 53.43%) |
| 2 | Learned fusion classifier (self-trained) | tie (53.47% vs 53.43%), not shipped |
| 3 | Size-first two-stage classification | rejected (49.25% vs 53.43%) |
| 4 | Joint REF_EDGE + decast-atlas retuning | rejected (44.98% vs 53.43%) |
| 5 | ORB vs. SIFT | rejected (41.55% vs 53.52% in-sample) |
| 6a | SIFT hard gate vs. soft floor | rejected (53.41% vs 53.43%, a wash) |
| 6b | Drop the REF_BLUR patch | rejected (51.37% vs 53.43%) |
| 7 | 2-fold → 4-fold CV protocol | **adopted**, used to judge every item above |

None of the six substantive ideas (1-6) beat `reconfigured/`'s architecture once
measured fairly under k=4 CV — see [`SOLUTION.md`](SOLUTION.md) for what that
means for the final shipped configuration and the honest final numbers.
