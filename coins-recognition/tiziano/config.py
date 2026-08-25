"""Physical constants and frozen calibration for the tiziano pipeline.

Everything tiziano shares with giacomo (physics, detection, canonical-crop
geometry, shape/colour cue calibration, most of the fusion-weight block) lives
in common/config.py. This file only holds what's actually specific to
tiziano: the SIFT matching cue's constants, the dead FAMILY_OF mapping kept
for fidelity to the notebook, and the recalibrated W_ORDER/W_SAME.
"""
from common.config import *  # noqa: F401,F403

# Class -> family mapping defined in the notebook's setup cell with different
# family names ("nordic_gold" instead of "gold"). Kept for fidelity: it is
# never referenced by any function in solution_2_T.ipynb (FAMILY above is
# what's actually used everywhere), so it has no effect on any output.
FAMILY_OF = {
    "1cent": "copper", "2cent": "copper", "5cent": "copper",
    "10cent": "nordic_gold", "20cent": "nordic_gold", "50cent": "nordic_gold",
    "1euro": "bimetal", "2euro": "bimetal",
}

# --- Classification: SIFT matching cue (tiziano's addition) ---------------
SIFT_NFEATURES = 700
SIFT_CONTRAST = 0.04
SIFT_NMS_DIST = 10       # spatial NMS min distance (pixels)
LOWE_RATIO = 0.8         # Lowe ratio test threshold
W_SIFT = 1               # weight of the SIFT cue in the unary
SIFT_FLOOR = 0.3         # floor of the SIFT confidence-based damping

# NOTE: W_ORDER/W_SAME are recalibrated here vs giacomo (0.25/0.15 there) --
# this variant weighs the pairwise/same-class size terms twice as heavily.
W_ORDER, W_SAME = 0.5, 0.2
