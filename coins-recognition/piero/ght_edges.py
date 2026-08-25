"""Section 2's GHT edge-voting model and the recall-first overshoot detector
plus GHT-confidence weak-rim fallback (Section 3) built alongside it.

IMPORTANT: none of this is used by the production pipeline. solution.ipynb
builds GHT_MODEL and defines weak_rim_fallback, but detect_coins (Section 3)
never calls weak_rim_fallback, and classify (Section 4, S11.3-bis) never
references GHT_MODEL/ght_rank at all -- the notebook's own comments say so
explicitly ("NOT used in production ... its precision collapses ... kept for
reference"). This module is ported for completeness and diffability (per-file
comparison against the other solutions' approaches), matching what the
notebook actually contains, but piero/pipeline.py's initialize_ght/evaluate
do not call anything here. See piero/README.md.
"""
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .appearance import href_11


def nlm(gray):
    """Non-local-means denoise -- used only by the GHT edge model below."""
    return cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)


_gy, _gx = np.ogrid[:C.GHT_M, :C.GHT_M]
GCIRC = ((_gx - C.GHT_M / 2) ** 2 + (_gy - C.GHT_M / 2) ** 2) <= (0.78 * C.GHT_M / 2) ** 2


def _disc(gray, cx, cy, r):
    R = int(r)
    return cv2.resize(gray[max(0, cy - R):cy + R, max(0, cx - R):cx + R], (C.GHT_M, C.GHT_M),
                      interpolation=cv2.INTER_CUBIC)


def ght_edges(gray, cx, cy, r):
    """N strongest-gradient edge points (orientation 0-180) inside the circle, on the NLM-denoised disc."""
    d = _disc(nlm(gray), cx, cy, r).astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, 3)
    mag = np.hypot(gx, gy)
    mag[~GCIRC] = 0
    idx = np.argpartition(mag.ravel(), -C.GHT_N)[-C.GHT_N:]
    ys, xs = np.unravel_index(idx, mag.shape)
    return xs.astype(np.float32), ys.astype(np.float32), np.rad2deg(np.arctan2(gy[ys, xs], gx[ys, xs])) % 180


def ght_masks(xs, ys, ori):
    m = np.zeros((C.GHT_K, C.GHT_M, C.GHT_M), np.uint8)
    b = (ori / (180 / C.GHT_K)).astype(int) % C.GHT_K
    m[b, ys.astype(int), xs.astype(int)] = 1
    ker = np.ones((2 * C.GHT_TOL + 1, 2 * C.GHT_TOL + 1), np.uint8)
    for k in range(C.GHT_K):
        m[k] = cv2.dilate(m[k], ker)
    return m


def ght_vote(model, masks):
    """Fraction of model edges that find a same-orientation target edge, at the best rotation."""
    rxs, rys, rori = model
    best = 0.0
    for th in range(0, 360, C.GHT_STEP):
        a = np.deg2rad(th)
        ca, sa = np.cos(a), np.sin(a)
        ix = np.round((rxs - C.GHT_M / 2) * ca - (rys - C.GHT_M / 2) * sa + C.GHT_M / 2).astype(int)
        iy = np.round((rxs - C.GHT_M / 2) * sa + (rys - C.GHT_M / 2) * ca + C.GHT_M / 2).astype(int)
        ok = (ix >= 0) & (ix < C.GHT_M) & (iy >= 0) & (iy < C.GHT_M)
        ob = ((rori + th) / (180 / C.GHT_K)).astype(int) % C.GHT_K
        ixo, iyo, obo = ix[ok], iy[ok], ob[ok]
        hit = masks[obo, iyo, ixo] | masks[(obo + 1) % C.GHT_K, iyo, ixo] | masks[(obo - 1) % C.GHT_K, iyo, ixo]
        best = max(best, hit.sum() / C.GHT_N)
    return best


def build_ght_model(reference_dir):
    """{class_name: (xs, ys, orientations)} -- the GHT edge model, per reference coin."""
    reference_dir = Path(reference_dir)
    model = {}
    for name in C.REF_NAMES:
        gray = cv2.cvtColor(cv2.imread(str(reference_dir / f"{name}.jpg")), cv2.COLOR_BGR2GRAY)
        cx, cy, r = href_11(gray)
        model[name] = ght_edges(gray, cx, cy, r)
    return model


def ght_rank(bgr, cx, cy, r, pool, ght_model):
    """Rank `pool` references by GHT vote for a target coin. Returns [(name, score), ...] desc."""
    masks = ght_masks(*ght_edges(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cx, cy, r))
    return sorted(((n, ght_vote(ght_model[n], masks)) for n in pool), key=lambda x: -x[1])


def overshoot(bgr, min_dist=C.OVERSHOOT_MIN_DIST, param2=C.OVERSHOOT_PARAM2,
             rmin=C.OVERSHOOT_RMIN, rmax=C.OVERSHOOT_RMAX):
    """Recall-first detector: CLAHE+median + sensitive Hough (high recall, many FP)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    en = cv2.medianBlur(cv2.createCLAHE(2.0, (8, 8)).apply(g), 5)
    c = cv2.HoughCircles(
        en, cv2.HOUGH_GRADIENT, 1.2, minDist=min_dist, param1=150,
        param2=param2, minRadius=rmin, maxRadius=rmax,
    )
    return [] if c is None else [tuple(map(int, np.round(x))) for x in c[0]]


def weak_rim_fallback(bgr, ght_model, conf=C.WEAK_RIM_CONF):
    """Explored weak-rim recovery: overshoot + GHT-confidence gate. NOT used in production --
    on the full set its precision collapses (textured/dark backgrounds make the overshoot flood
    with candidates that the gate cannot all reject), inflating counts ~3x. Kept for reference."""
    coins = []
    for (cx, cy, r) in overshoot(bgr):
        sc = ght_rank(bgr, cx, cy, r, C.REF_NAMES, ght_model)
        if sc[0][1] - sc[1][1] >= conf:
            coins.append((cx, cy, r))
    return coins
