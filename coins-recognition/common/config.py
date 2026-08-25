"""Physical constants and frozen calibration shared by giacomo and tiziano.

Every number here is a measured/calibrated value from solution_2.ipynb -- see
giacomo/README.md and tiziano/README.md for what each stage does and why these
values were chosen. Nothing here should be tuned casually; changing a value
changes measured detection/classification accuracy.

W_ORDER/W_SAME (the pairwise/same-class size-term weights) are NOT here: the two
packages deliberately use different values for them, so each defines its own in
its local config.py.
"""
import numpy as np

# --- Coin physics --------------------------------------------------------
DIAMETER_MM = {
    "1cent": 16.25, "2cent": 18.75, "10cent": 19.75, "5cent": 21.25,
    "20cent": 22.25, "1euro": 23.25, "50cent": 24.25, "2euro": 25.75,
}
VALUE = {
    "1cent": 0.01, "2cent": 0.02, "5cent": 0.05, "10cent": 0.10,
    "20cent": 0.20, "50cent": 0.50, "1euro": 1.00, "2euro": 2.00,
}
CLASSES = ["1cent", "2cent", "5cent", "10cent", "20cent", "50cent", "1euro", "2euro"]
FAMILY = {
    **{c: "copper" for c in ("1cent", "2cent", "5cent")},
    **{c: "gold" for c in ("10cent", "20cent", "50cent")},
    **{c: "bimetal" for c in ("1euro", "2euro")},
}

# --- Detection: candidate generation (permissive Hough, S1/S6) -----------
R_MIN, R_MAX, K_MAX = 30, 130, 1

# --- Detection: radial-edge validator (S2) --------------------------------
N_RAYS, MAX_DR, MIN_SUPPORT, MIN_RAYS = 72, 14, 0.45, 20

# --- Detection: physical gates (S3) ---------------------------------------
BG_Z_THR = 0.85
SCALE_MAX_RATIO = 1.75

# --- Classification: canonical crop geometry (S6.1) -----------------------
CANON = 150
GRID_Y, GRID_X = np.ogrid[:CANON, :CANON]
RHO = np.hypot(GRID_X - CANON / 2, GRID_Y - CANON / 2) / (CANON / 2)
DMASK = RHO < 0.95

# Fixed affine colour de-cast (gain/offset per BGR channel), calibrated once
# on the reference set (S6.1).
CAST_GAIN = (0.745, 0.908, 0.722)
CAST_OFF = (14.12, 18.38, 53.97)

# --- Classification: shape-based matching (S6.2) --------------------------
ANGLES = range(0, 360)
TAU = float(np.cos(2 * np.radians(30)))
REF_EDGE = {
    "1cent": (5, 30), "2cent": (5, 60), "5cent": (10, 40), "10cent": (20, 60),
    "20cent": (60, 140), "50cent": (10, 40), "1euro": (30, 50), "2euro": (10, 50),
}
REF_NLM = {
    "1cent": 3, "2cent": 2, "5cent": 5, "10cent": 3,
    "20cent": 5, "50cent": 7, "1euro": 3, "2euro": 5,
}
REF_BLUR = {
    "5cent": (10, (107, 75, 30)),
    "2cent": (10, (100, 60, 40)),
}
TGT_NLM = 4

# --- Classification: colour cues (S6.1/S4.1) ------------------------------
BIMETAL_ABSTAIN_STEP = 13.0

# --- Classification: cue weights, frozen calibration (S8) -----------------
TEMPLATE_W, TEMPLATE_ALPHA = 0.5, 0.2
TEMPLATE_OFFDIAG = {
    "10cent": 0.473, "1cent": 0.518, "1euro": 0.467, "20cent": 0.423,
    "2cent": 0.430, "2euro": 0.470, "50cent": 0.456, "5cent": 0.479,
}
FAM_POS, FAM_NEG = 3.4, 1.45
BIM_START, BIM_FULL, BIM_POS, BIM_NEG, BIM_RING = 2.5, 10.5, 4.0, 1.0, 0.16
FAM_BIM_SUP, BIM_FAM_SUP, BIM_FAM_MIN = 0.92, 0.75, 0.15
REL_W, BEAM_K = 0.75, 256
RATIO_TOL, SAME_TOL = 0.18, 0.045
C1_FLOOR, C1_ZLO, C1_ZHI = 0.3, 1.5, 4.0
ANCHOR_CONF, ANCHOR_W = 0.4, 4.0
LOGMAX = float(np.log(25.75 / 16.25))
