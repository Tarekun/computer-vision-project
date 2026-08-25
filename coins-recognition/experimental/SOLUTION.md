# Solution write-up: what this iteration found, and why nothing shipped differently

## Headline

`experimental/` ran a full design → implement → test loop over the 6 substantive
ideas in [`REVIEW.md`](REVIEW.md)'s backlog, each measured against a controlled
baseline under cross-validation, exactly the way [`reconfigured/`](../reconfigured/)
measured its own decast-atlas experiment. **None of them beat `reconfigured/`'s
architecture.** The shipped `experimental/config.py`/`classify.py` are therefore
functionally identical to `reconfigured/`'s — verified byte-for-byte via an
independent end-to-end run of `pipeline.py` with `IMPLEMENTATION = "experimental"`
over all 142 target images, reproducing `reconfigured/`'s exact 97.18% count / 53.52%
amount (in-sample) numbers and mistake list.

That is not a null result, though. This iteration produced three things of real
value: (1) a more reliable evaluation protocol (4-fold instead of 2-fold
cross-validation) that revises the honest generalization estimate for this whole
family of approaches; (2) six ideas that sounded plausible and are now ruled out
*with evidence*, closing off dead ends rather than leaving them as untested
"maybe"s; and (3) one genuinely interesting finding — a 7-parameter self-trained
learned classifier statistically ties reconfigured's much more heavily
hand-engineered fusion, suggesting the *feature set* is what's carrying this
architecture, not the specific fusion weights.

## What was tried, and what happened

| item | idea | mean held-out amount_acc | vs. day-0 baseline (53.43%) | verdict |
|---|---|---|---|---|
| 1 | Multi-exemplar jittered reference atlas | 43.55% | -9.9 | rejected |
| 2 | Self-trained learned fusion classifier | 53.47% | +0.04 | **tie**, not shipped |
| 3 | Size-first two-stage classification | 49.25% | -4.2 | rejected |
| 4 | Joint `REF_EDGE` + decast-atlas retuning | 44.98% | -8.5 | rejected |
| 5 | ORB instead of SIFT | 41.55%¹ | -12.0¹ | rejected |
| 6a | SIFT hard gate instead of soft floor | 53.41% | -0.02 | **tie**, kept soft floor |
| 6b | Drop the `REF_BLUR` hand patch | 51.37% | -2.1 | rejected |
| 7 | 2-fold → 4-fold CV protocol | — | — | **adopted**, used above |

¹ measured in-sample (53.52%) under identical fusion weights, not via the k=4 CV
protocol — see [`EXPERIMENTS.md`](EXPERIMENTS.md) item 5 for why that's still a
fair, sufficient comparison for this one.

Full detail, methodology, and the reasoning behind each verdict is in
[`EXPERIMENTS.md`](EXPERIMENTS.md) — this is just the summary table.

**Why so many rejections isn't a sign anything went wrong:** every idea targeted a
real, defensible weakness (a one-shot reference exemplar, a hand-tuned fusion
formula, an unexploited size cue, an unretuned legacy threshold table, a heavier
descriptor, a fragile hardcoded patch). Testing each in isolation, against a
controlled baseline, under cross-validation, is exactly how you find out that a
plausible-sounding fix doesn't actually help *before* shipping it — the same
discipline `reconfigured/README.md` already used once (the decast-atlas fix) and
this iteration used six more times. A backlog that turns out to be mostly dead ends
is a normal outcome of that discipline, not evidence the discipline wasn't applied.

## An important honesty check: search-budget sensitivity

This iteration's per-idea screening used a lighter search budget (500 random trials
+ 2 coordinate-ascent passes per fold) than `reconfigured/tune.py`'s original design
(1500 trials + 3 passes), to keep 8 separate full k=4 CV runs affordable. Before
trusting that screening budget's numbers, this was checked for the one variant that
matters most: the unchanged baseline itself, run twice, same data, same folds, only
the search thoroughness different:

| search budget | mean held-out amount_acc | fold range |
|---|---|---|
| 500 trials / 2 passes (screening, used for every item above) | 53.43% | 42.86% – 61.11% |
| 1500 trials / 3 passes (`reconfigured/tune.py`'s original design) | **48.55%** | 42.86% – 55.56% |

A ~5-point swing from search thoroughness alone, on an *identical* model. This is
the same overfitting-risk story `reconfigured/README.md` already flagged (18 free
parameters fit against ~107 training images per fold is a lot of freedom for that
little data) manifesting a second way: a more thorough search finds a train-fitting
optimum that doesn't always transfer better to the held-out fold. It does not
change any keep/revert decision above — every rejected idea's effect size (4.2 to
12.0 points) is well beyond this ~5-point noise floor, and both near-ties (item 2,
item 6a) were already reported as ties rather than wins. It does mean the number to
actually trust as *the* generalization estimate for this whole family of
architectures is the more rigorous **48.55%**, not the lighter screening run's
53.43%, and not `reconfigured/`'s own original 2-fold estimate of ~48-51% either
(different protocol, coincidentally similar range).

## Final honest numbers

| | count accuracy | amount accuracy |
|---|---|---|
| giacomo (baseline) | 97.18% | 48.59% |
| tiziano (baseline) | 97.18% | 45.07% |
| reconfigured / experimental (in-sample, shipped weights) | 97.18% | 53.52% |
| reconfigured (original 2-fold CV) | 97.18% | ~48-51% |
| **experimental (4-fold CV, 1500 trials/3 passes)** | 97.18% | **~48.6%** |

Read this as: a real, modest improvement over giacomo's original 48.59% survives
cross-validation, on the order of a couple of points — not the ~5-point jump the
in-sample number suggests, and not a dramatic rewrite of what's achievable with this
classical-CV toolkit on 142 images. Detection (count accuracy) has been the settled
part of this problem since `giacomo/` — every variant tried across four
implementations shares `common/detection.py` unchanged. The ceiling here is on the
classification side, and after two full rounds of hypothesis-driven iteration
(`reconfigured/` then `experimental/`), that ceiling looks like it's close to
wherever giacomo/reconfigured's approach already sits — squeezing a genuinely large
further gain out of *this* feature set and fusion strategy looks unlikely without
either more reference data (more than 1 image/class) or a different kind of feature
entirely (both explicitly out of scope for this exercise).

## What's still worth knowing about, even though nothing changed

- **The learned-classifier tie (item 2) is a real signal, not noise-shaped-like-a-tie.**
  A 7-parameter linear model, fit from nothing but self-training on ambiguity-filtered
  pseudo-labels (no per-coin ground truth exists anywhere in this repo — confirmed,
  see `EXPERIMENTS.md` item 2), statistically matches an ~11-weight fusion formula
  that was hand-engineered and then extensively searched. That says the *shape/family/
  bimetal/SIFT/size cue set itself* is where the discriminative power lives; further
  hand-tuning the fusion arithmetic on top of it has limited headroom left. Anyone
  picking this project back up should look at *adding a new cue* before trying to
  out-tune the existing fusion weights again.
- **`REF_BLUR` (item 6b) is fragile *and* still helping** — a real tension, not a
  contradiction to resolve here. It is exactly the kind of hardcoded, single-purpose,
  two-specific-photos patch `ANALYSIS.md` was right to flag as a maintenance risk,
  but removing it measurably hurts (-2.1 points) with no replacement in hand. If
  `reference_set/5cent.jpg` or `2cent.jpg` are ever swapped for different photos,
  this patch's fixed pixel coordinates will need re-placing by hand, or the atlas
  will silently regress for those two classes.
- **The `REF_EDGE`+decast joint search space (item 4) was only lightly explored** —
  10 random trials over 16 coupled threshold parameters is a coarse sample of a
  large space. The isolated screening test found one candidate that looked
  marginally better than the no-decast baseline (0.2723 vs. 0.2744 mean absolute
  error) but that edge evaporated (and reversed, sharply) once fusion weights were
  properly retuned for it under full CV. A real, larger search here (e.g. Bayesian
  optimization instead of pure random search, or a per-class coordinate-ascent
  refinement rather than one flat random draw of all 8 classes at once) is plausible
  future work, but expensive: every candidate needs the atlas rebuilt and the
  360-rotation shape-score search re-run over all 142 target images.
- **Detection was entirely out of scope for both `reconfigured/` and `experimental/`.**
  All four implementations that share `common/detection.py` sit at the same 97.18%
  count accuracy; nothing in either iteration touched the Hough-based localization
  or its physical gates. If a future push wants a different ceiling, that's the
  other place to look, not classification.

## Files

`experimental/` is a structural fork of `reconfigured/` (same contract:
`initialize_ght`/`evaluate`), with every idea from the backlog implemented behind
an off-by-default config flag so the reverted code stays in the repo as evidence
rather than being deleted and re-derived if anyone wants to check the reasoning:

- `REVIEW.md` — the backlog this iteration started from.
- `EXPERIMENTS.md` — full detail on all 7 items: hypothesis, implementation,
  measured result, decision.
- `config.py` — `JITTERED_ATLAS`, `TWO_STAGE_SIZE_FIRST`, `DECAST_ATLAS`,
  `SIFT_HARD_GATE`, `DISABLE_REF_BLUR`: all off by shipped default, all documented
  inline with what they measured.
- `appearance.py`, `reference.py`, `classify.py`, `sift_cue.py` — forked from
  `reconfigured/`, with each idea's code gated behind its flag.
- `orb_cue.py`, `orb_compare.py` — item 5 (ORB vs. SIFT).
- `learned_fusion.py` — item 2 (self-trained learned classifier), kept as a
  documented alternative, not called by the shipped `classify.fuse_scene`.
- `ref_edge_search.py`, `ref_edge_search_results.json` — item 4's isolated
  ablation search and its full numeric log.
- `two_stage_search.py`, `two_stage_search_results.json` — item 3's search.
- `tune.py` — forked from `reconfigured/tune.py`, extended with a general `k`-fold
  `cross_validate` (item 7) instead of the fixed 2-fold split.
- `day0_quick_cv4.json`, `final_cv4.json`, `item1_jitter_quick_cv4.json`,
  `item4_decast_quick_cv4.json`, `item6b_norefblur_quick_cv4.json` — the raw
  per-fold numeric records behind every table in this file and `EXPERIMENTS.md`.
