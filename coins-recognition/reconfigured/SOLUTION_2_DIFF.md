# `solution_2.ipynb`: giacomo → reconfigured

[`solution_2.ipynb`](solution_2.ipynb) here is a port of
[`../giacomo/solution_2.ipynb`](../giacomo/solution_2.ipynb) onto the fixes and tuned
weights that already existed as plain modules in this directory (`appearance.py`,
`reference.py`, `sift_cue.py`, `classify.py`, `config.py` — see [`README.md`](README.md)
for how those were derived). It is self-contained like the original: no cell imports from
`reconfigured.*`, everything is defined inline, and it reads like the next revision of the
same notebook rather than a generated wrapper around the package. This file is a thin
diff of the two notebooks; it doesn't repeat what's already in `README.md` or
`../ANALYSIS.md`.

## What's identical

- §0 Setup, §1–§5 (detection: candidate generation, geometric validation, physical gates,
  assembly, measured results) — byte-for-byte the same cells. Detection wasn't touched by
  the reconfiguration.
- §6.3 (colour/bimetal tone cues) — same functions, same constants.
- §7 (`process_image`, the final full-target-set run) — same code, same output format.
- Demo cells use the same images (`image_23/24/55` for detection, `image_5/24` for
  classification).

## What changed

- **§6.1 `crop_disc`** — now pads a border-clipped coin to a square box by edge
  replication instead of force-resizing a clipped, non-square patch, so it no longer
  stretches an off-frame coin's aspect ratio. One new markdown paragraph explains why.
- **§6.2 reference atlas** — a new block after the atlas loop computes `TEMPLATE_OFFDIAG`
  from the reference set itself (each class's atlas scored against every *other*
  reference's crop) instead of using 8 hand-typed constants. The markdown adds this and a
  paragraph on a fix that was tried and reverted: decasting the atlas-building crop too
  measured *worse* (49.30% → 45.77% amount accuracy) because `REF_EDGE`'s Canny thresholds
  were implicitly tuned against the raw photo's contrast — so the atlas still reads
  `ref_bgr` raw, unchanged from giacomo.
- **New §6.4, "A fifth cue: SIFT keypoint matching"** — giacomo's notebook never had a
  SIFT cue at all (that was `tiziano`'s addition, and it made things worse there). This
  section and its code cell are new material: the reference descriptor cache is built from
  the same canonical crop the shape atlas uses (not a raw photo), `SIFT_NMS_DIST` is
  recalibrated 10px → 5px for that crop size, and per-class match counts are normalised by
  each class's own descriptor count rather than the global total.
- **§6.5 (was §6.4) scene-level labelling** — `classify_scene` now also extracts the SIFT
  cue and folds it in via z-scoring (same mechanism as the shape template), reads the
  data-derived `TEMPLATE_OFFDIAG` from §6.2 instead of a literal dict, and adds an
  `ANCHOR_CONSISTENCY_TOL` check that drops the absolute bimetal scale anchor when it
  disagrees with the independently-seeded scale by too much. The weights block is the
  output of `tune.py`'s random-search + coordinate-ascent run, not hand-picked values.
  Note: the standalone package (`classify.py`) splits this into `extract_features` +
  `fuse_scene` so `tune.py` can cache the expensive part and re-run only the cheap fusion
  arithmetic per candidate weight vector — that split is a tuning-script concern, so this
  notebook keeps `classify_scene` as one function, the way giacomo's original does.
- **§8 measured results** — rewritten, not just re-numbered. Giacomo's §8 reports per-coin
  accuracy (0.734) against a hand-labelled 99-image/227-coin subset. This version reports
  **amount accuracy** against `data/coin_dataset/gt.csv`'s full 142-image coverage instead
  (48.59% → 53.52% in-sample, ~48–51% under two-fold cross-validation), because that's both
  the metric the fusion weights were actually tuned against and the one the project's own
  grading harness (`pipeline.py`) reports. It also documents the decast-atlas ablation
  table, the mean-absolute-value-error tuning objective (and why raw amount accuracy is a
  poor search signal), the honest CV numbers and their overfitting caveat, and the
  cross-fold `FAM_POS` finding — none of which existed in giacomo's version, which had its
  own different (and, for this variant, no longer applicable) dev/sealed per-coin ablation
  table.
- **Title cell** — rewritten to describe the reconfigured pipeline's five cues and the
  measured/cross-validated numbers instead of giacomo's four cues and per-coin accuracy.

## What's out of scope here

`ablate.py`, `tune.py` and `tune_results.json` (the tuning/ablation infrastructure and its
full numeric log) are not reproduced in the notebook — they're referenced from §6.2/§8 the
same way giacomo's notebook keeps some ablation detail "outside the notebook." See
[`README.md`](README.md) for the full tuning methodology and every run's numbers.
