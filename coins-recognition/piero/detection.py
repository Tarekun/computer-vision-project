"""Production coin detector (Section 3 of solution.ipynb): precision-first
Hough circles + a rim-score FP filter. This is the detector actually used by
detect_coins/classify (Section 5's final loop) -- the recall-first overshoot
detector and GHT-confidence weak-rim fallback explored alongside it are kept
in ght_edges.py, unused here, exactly as the notebook leaves them unused.
"""
from math import pi as PI

import cv2
import numpy as np

from . import config as C


def hough_clean(bgr, min_dist=C.HOUGH_CLEAN_MIN_DIST, param2=C.HOUGH_CLEAN_PARAM2,
                rmin=C.HOUGH_CLEAN_RMIN, rmax=C.HOUGH_CLEAN_RMAX):
    """Precision-first detector: exact on normal images, ~0 background FP."""
    g = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, 1.2, minDist=min_dist, param1=150,
        param2=param2, minRadius=rmin, maxRadius=rmax,
    )
    return [] if c is None else [tuple(map(int, np.round(x))) for x in c[0]]


def rim_score(gx, gy, cx, cy, r, T=C.RIM_SCORE_T, n=C.RIM_SCORE_N):
    """Fraction of the circle perimeter carrying a strong radially-aligned gradient.
    A real coin has a near-complete rim; a Hough FP fitted to an inter-coin gap/shadow does not."""
    H, W = gx.shape
    cnt = tot = 0
    for k in range(n):
        th = 2 * PI * k / n
        nx, ny = np.cos(th), np.sin(th)
        x = int(round(cx + r * nx))
        y = int(round(cy + r * ny))
        if 0 <= x < W and 0 <= y < H:
            tot += 1
            if abs(gx[y, x] * nx + gy[y, x] * ny) > T:
                cnt += 1
    return cnt / tot if tot else 0.0


def detect_coins(bgr, rim_thr=C.DETECT_RIM_THR):
    """Precision-first Hough + a light rim-score FP filter that drops circles fitted to
    inter-coin gaps/shadows (rim < threshold)."""
    coins = hough_clean(bgr)
    if not coins:
        return coins
    g = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    return [c for c in coins if rim_score(gx, gy, *c) >= rim_thr]
