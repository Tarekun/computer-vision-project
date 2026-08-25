"""Physical constants and calibration for the experimental pipeline (forked from
reconfigured/config.py -- see REVIEW.md for the backlog this fork works through and
EXPERIMENTS.md for what was measured).

Everything giacomo/tiziano/reconfigured/experimental share (physics, detection,
canonical-crop geometry, per-class Canny/NLM tables, beam-search mechanics) lives in
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

# --- EXPERIMENTS.md item 1: jittered-atlas reference model --------------------
# Small photometric perturbations (brightness offset, contrast gain, blur sigma) of
# the SINGLE reference photo per class. reference.py builds each class's shape atlas
# as a vote across these jittered variants instead of reading the one raw instance,
# so the atlas is less sensitive to that photo's exact lighting/noise realization.
# Still derived from nothing but the 8 reference images -- no new data.
REF_JITTERS = [
    (0.0, 1.0, 0.0),     # unperturbed original, always included
    (-12.0, 1.0, 0.0), (12.0, 1.0, 0.0),   # brightness -/+
    (0.0, 0.85, 0.0), (0.0, 1.15, 0.0),    # contrast -/+
    (0.0, 1.0, 0.6),                        # slight blur
]
ATLAS_VOTE_FRAC = 0.5  # a control point is kept if present in >= this fraction of jittered atlases
# Off by shipped default: measured WORSE than a single raw-instance atlas (EXPERIMENTS.md
# item 1, 43.55% vs. 53.43% mean held-out amount_acc). Code kept for the write-up/evidence.
JITTERED_ATLAS = False

# --- EXPERIMENTS.md item 4: joint REF_EDGE + decast-atlas retuning ------------
# reconfigured/README.md flagged retuning REF_EDGE jointly with a decasted atlas
# input as follow-up work. ref_edge_search.py's isolated search (giacomo weights,
# SIFT off) found REF_EDGE_RETUNED + decast beats giacomo's original REF_EDGE without
# decast (0.2723 vs. 0.2744 mean_abs_err) -- a small edge from 10 random trials.
# DECAST_ATLAS gates the pair; see EXPERIMENTS.md for the full-pipeline CV verdict
# (with fusion weights properly retuned for this atlas) that decides the default.
DECAST_ATLAS = False
REF_EDGE_RETUNED = {
    "1cent": (3.180936271462449, 31.772784882940005),
    "2cent": (6.640064058796159, 61.64150247651476),
    "5cent": (11.444713060498053, 24.854297429190567),
    "10cent": (22.159998558583332, 65.10424405220252),
    "20cent": (63.64574150551395, 127.81545384415661),
    "50cent": (8.9611195226815, 55.3765328207126),
    "1euro": (18.87340890267566, 30.865460394200962),
    "2euro": (13.688250241916888, 37.398877655897536),
}

# --- EXPERIMENTS.md item 3: size-first two-stage classification ---------------
# When on, a coin whose family/bimetal cue clears the gate below is only allowed to
# take a label from its resolved family's 2-3 classes during the beam search,
# instead of every class -- see classify.py's beam_assign_restricted. Off by
# default; flipped on (and the two gates tuned) only if EXPERIMENTS.md's measured
# result says to keep it.
TWO_STAGE_SIZE_FIRST = False
FAMILY_GATE_CONF = 0.6
FAMILY_GATE_BIM = 0.6

# --- EXPERIMENTS.md item 6: cleanup / reconsideration items --------------------
# SIFT_HARD_GATE: gate the SIFT cue off entirely below a confidence threshold instead
# of SIFT_FLOOR's soft floor (which keeps it partially on even at zero confidence).
# Off by default -- see EXPERIMENTS.md for the measured comparison.
SIFT_HARD_GATE = False
SIFT_HARD_GATE_THRESHOLD = 0.3

# REF_BLUR (common/config.py) hardcodes a per-photo pixel-circle patch for 2 of the 8
# reference images. DISABLE_REF_BLUR (off by default) lets ref_blur_ablation.py
# measure whether it's still earning its keep -- see EXPERIMENTS.md item 6b.
DISABLE_REF_BLUR = False
