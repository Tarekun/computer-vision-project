"""Physical constants and frozen calibration for the piero pipeline.

Duplicate of giacomo/config.py's role, but for a differently-designed
solution (solution.ipynb): its own Hough detector tuning, its own §11.3-bis
classifier, and a Section-2 GHT edge-voting model that the notebook builds
but never actually wires into the production detect_coins/classify path
(see piero/README.md and piero/ght_edges.py).

Identifier names/suffixes (11, 3b) are kept as in the notebook, matching its
own section numbering (S11 baseline classifier, S11.3-bis refinement), to
keep this file easy to diff against the notebook source.
"""
import numpy as np

# --- Coin physics --------------------------------------------------------
DIAMETER_MM = {
    "1cent": 16.25, "2cent": 18.75, "5cent": 21.25, "10cent": 19.75,
    "20cent": 22.25, "50cent": 24.25, "1euro": 23.25, "2euro": 25.75,
}
VALUE_EUR = {
    "1cent": 0.01, "2cent": 0.02, "5cent": 0.05, "10cent": 0.10,
    "20cent": 0.20, "50cent": 0.50, "1euro": 1.00, "2euro": 2.00,
}
REF_NAMES = sorted(VALUE_EUR)
MONO = ["1cent", "2cent", "5cent", "10cent", "20cent", "50cent"]
BIMETAL = ["1euro", "2euro"]

# --- Section 2: GHT edge-voting model ------------------------------------
# Built by the notebook (GHT models built: [...] is printed) but its only
# consumer, weak_rim_fallback, is never called by the production detector
# (detect_coins) or classifier (classify) -- see ght_edges.py.
GHT_N, GHT_K, GHT_TOL, GHT_STEP, GHT_M = 250, 18, 2, 12, 160

# --- Section 3: detection -------------------------------------------------
HOUGH_CLEAN_MIN_DIST, HOUGH_CLEAN_PARAM2 = 95, 44
HOUGH_CLEAN_RMIN, HOUGH_CLEAN_RMAX = 45, 150
OVERSHOOT_MIN_DIST, OVERSHOOT_PARAM2 = 85, 42
OVERSHOOT_RMIN, OVERSHOOT_RMAX = 45, 150
WEAK_RIM_CONF = 0.08
RIM_SCORE_T, RIM_SCORE_N = 40, 72
DETECT_RIM_THR = 0.42

# --- Section 4: the operating classifier (S11.3-bis, DEC-15) --------------
CANON11 = 150
ROT_STEP11 = 1
ANGLES11 = list(range(0, 360, ROT_STEP11))
MATCH_TOL11 = 30
TAU11 = float(np.cos(2 * np.radians(MATCH_TOL11)))
NLM_REF11, GSIG_REF11 = 1, 0.3
CASTG11 = [0.745, 0.908, 0.722]
CASTO11 = [14.12, 18.38, 53.97]
MATERIAL11 = {
    "1cent": "copper", "2cent": "copper", "5cent": "copper",
    "10cent": "gold", "20cent": "gold", "50cent": "gold",
    "1euro": "bimetal", "2euro": "bimetal",
}
GROUPS11 = ["copper", "gold", "bimetal"]
TH_BI_W11, TH_BI_S11 = 0.10, 0.40
COLOR_FAV11, BI_FAV11 = 0.6, 1.2

W_ORDER11 = 0.25
W_SAME11 = 0.15
NLM_ZOOM11 = 4
SOFTMAX_T11 = 0.05
LOGMAX11 = float(np.log(max(DIAMETER_MM.values()) / min(DIAMETER_MM.values())))

# Per-class reference Canny thresholds -- every class specifies both 'low'
# and 'high', so edgesfor3b's adaptive-threshold fallback is never actually
# exercised while building this atlas (kept anyway for fidelity).
EDGE_PARAMS3b = {
    "1cent": {"low": 5, "high": 50}, "2cent": {"low": 10, "high": 90}, "5cent": {"low": 10, "high": 50},
    "10cent": {"low": 20, "high": 80}, "20cent": {"low": 10, "high": 80}, "50cent": {"low": 10, "high": 40},
    "1euro": {"low": 30, "high": 80}, "2euro": {"low": 10, "high": 50},
}
NLM_REF3b = {"1cent": 3, "2cent": 2, "5cent": 9, "10cent": 3, "20cent": 5, "50cent": 7, "1euro": 5, "2euro": 5}
REF_FILTER3b = "nlm"
BILAT_D3b, BILAT_SC3b, BILAT_SS3b = 7, 50, 50
EP_SS3b, EP_SR3b = 60.0, 0.4
MED_K3b = 5
# Per-coin: area of the REFERENCE disc to blur before the edge atlas is built
# (suppress unstable edges, e.g. the variable centre). Coords in the 150x150
# canonical disc (centre 75, 75).
BLUR3b = {
    "5cent": {"sigma": 10, "regions": [("disc", 107, 75, 30)]},
    "1euro": {"sigma": 4, "regions": [("disc", 100, 75, 27)]},
}
BLUR_SIGMA3b = 4
BLUR_FEATHER3b = 6

GRID_Y11, GRID_X11 = np.ogrid[:CANON11, :CANON11]
RHO11 = np.hypot(GRID_X11 - CANON11 / 2, GRID_Y11 - CANON11 / 2) / (CANON11 / 2)
DMASK11 = RHO11 < 0.95
