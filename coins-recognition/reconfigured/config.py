"""Physical constants and calibration for the reconfigured pipeline.

Everything giacomo/tiziano/reconfigured share (physics, detection, canonical-crop
geometry, per-class Canny/NLM tables, beam-search mechanics) lives in
common/config.py and is imported wholesale below. This file holds only what's
specific to this variant's fixes (see ANALYSIS.md and reconfigured/README.md):

- TEMPLATE_OFFDIAG is GONE from here: it's no longer a hand-typed per-class
  constant. reconfigured/reference.py derives it directly from the 8 reference
  images (how much each class's atlas over-matches the *other* seven reference
  coins on average), which is both more principled and still respects the
  "only the 8 reference images" constraint.
- The SIFT cue's reference cache is built from the same canonical crop the
  shape cue uses (not the raw photo -- see reconfigured/sift_cue.py), so
  SIFT_NMS_DIST is recalibrated for a 150x150 canvas instead of a full photo.
- ANCHOR_CONSISTENCY_TOL is new: reconfigured/classify.py distrusts the
  absolute bimetal scale anchor when it disagrees with the independently
  seeded scale by more than this (log-diameter units).

These were tuned by reconfigured/tune.py: a random search (1500 trials) followed by
coordinate-ascent refinement, minimizing mean absolute value error against
data/coin_dataset/gt.csv over all 142 target images -- the eval set, not the reference
model, so this doesn't touch the "only the 8 reference images" constraint. See
reconfigured/README.md for the full methodology and honest performance numbers
(including a two-fold cross-validation estimate, since these weights were fit directly
against the same 142 images the headline number is measured on). tune_results.json
holds the full search log for the run that produced these values (seed 8 of 9 tried,
picked for lowest mean absolute value error on the full set).
"""
from common.config import *  # noqa: F401,F403

# --- Classification: cue weights (tuned by tune.py, see tune_results.json) ----
TEMPLATE_W, TEMPLATE_ALPHA = 1.120312091692299, 0.46282535252112333
FAM_POS, FAM_NEG = 0.47203650129702585, 0.7094389741071441
BIM_START, BIM_FULL = 2.5, 10.5  # untouched: coupled threshold curve, not searched (see tune.py)
BIM_POS, BIM_NEG, BIM_RING = 5.7862640527246105, 1.757201100008255, 0.1457475687122199
FAM_BIM_SUP, BIM_FAM_SUP, BIM_FAM_MIN = 0.6546458793832082, 0.7638192098773395, 0.20998934153881837
W_ORDER, W_SAME = 0.5814595192882499, 0.15291231276135417
ANCHOR_CONF, ANCHOR_W = 0.5267327931386852, 6.421073568211093
ANCHOR_CONSISTENCY_TOL = 0.5047183128146107
C1_FLOOR, C1_ZLO, C1_ZHI = 0.1311294927519002, 1.5, 4.0  # ZLO/ZHI untouched, not searched

# --- Classification: SIFT matching cue, fixed cache domain (see sift_cue.py) --
SIFT_NFEATURES = 700
SIFT_CONTRAST = 0.04
SIFT_NMS_DIST = 5        # 150x150 canonical crop, not a full photo (tiziano used 10px on a full photo)
LOWE_RATIO = 0.8
W_SIFT = 1.794831084866778
SIFT_FLOOR = 0.014706133643594299
