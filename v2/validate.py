# v2/validate.py
# Geometric validator (port of e4): per-ray strongest radial edge, MAD outlier
# rejection, Kasa LSQ circle fit, edge-support accept gate. Also exports the
# selected edge points for the ellipse fit (rectify.py).
import math
import cv2
import numpy as np

N_RAYS, MAX_DR, MIN_SUPPORT, MIN_RAYS = 72, 14, 0.45, 20

def _kasa(xs, ys):
    A = np.c_[2*xs, 2*ys, np.ones(len(xs))]
    b = xs**2 + ys**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    return cx, cy, math.sqrt(sol[2] + cx*cx + cy*cy)

def _ray_edges(gray, cx, cy, r0):
    """Strongest outward radial gradient per ray within r0±MAX_DR."""
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    pts = []
    for k in range(N_RAYS):
        th = 2*math.pi*k/N_RAYS
        ux, uy = math.cos(th), math.sin(th)
        best, best_mag = None, 40.0          # soglia minima gradiente
        for dr in np.arange(-MAX_DR, MAX_DR + 0.5, 0.5):
            x, y = cx + (r0+dr)*ux, cy + (r0+dr)*uy
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < gray.shape[0] and 0 <= xi < gray.shape[1]): continue
            mag = abs(sob_x[yi, xi]*ux + sob_y[yi, xi]*uy)   # gradiente RADIALE
            if mag > best_mag: best, best_mag = (x, y), mag
        if best: pts.append(best)
    return pts

def validate(img, cx, cy, r0):
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 100, 100)
    pts = _ray_edges(gray, cx, cy, r0)
    support = len(pts) / N_RAYS
    if len(pts) < MIN_RAYS or support < MIN_SUPPORT:
        return {"accept": False, "cx": cx, "cy": cy, "r": r0,
                "support": support, "edge_pts": pts}
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    for _ in range(2):                                   # 2 iterazioni con rigetto MAD
        ncx, ncy, nr = _kasa(xs, ys)
        res = np.abs(np.hypot(xs - ncx, ys - ncy) - nr)
        mad = np.median(res) + 1e-6
        keep = res < 3 * mad + 1.0
        if keep.sum() < MIN_RAYS: break
        xs, ys = xs[keep], ys[keep]
    ncx, ncy, nr = _kasa(xs, ys)
    if abs(nr - r0) > MAX_DR:                            # clamp anti-deriva
        ncx, ncy, nr = cx, cy, r0
    return {"accept": True, "cx": float(ncx), "cy": float(ncy), "r": float(nr),
            "support": support, "edge_pts": list(zip(xs.tolist(), ys.tolist()))}
