"""Coin detection: permissive Hough candidate generation, radial-edge
validation with robust circle refit, and the three physical gates.

Shared verbatim by giacomo and tiziano (identical detector in both). See
giacomo/README.md S1-S5 or tiziano/README.md for the reasoning behind each step.
"""
import cv2
import numpy as np

from . import config as C


def hough_pass(gray, param1, param2, min_dist, seen):
    """One HoughCircles pass; keeps only circles whose centre is new (dedup vs `seen`)."""
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1.2, min_dist, param1=param1, param2=param2,
        minRadius=C.R_MIN, maxRadius=C.R_MAX,
    )
    new = []
    if circles is not None:
        for cx, cy, r in circles[0]:
            if all((cx - x) ** 2 + (cy - y) ** 2 > (0.5 * min(r, rr)) ** 2
                   for x, y, rr in seen + new):
                new.append((float(cx), float(cy), float(r)))
    return new


def candidates(img):
    """Two strict Hough passes plus a weak-edge rescue pass, trusted only if parsimonious."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    base = []
    for param2, min_dist in ((38, 85), (30, 70)):
        base += hough_pass(gray, 100, param2, min_dist, base)
    rescue = hough_pass(gray, 60, 30, 70, base)
    if len(rescue) > C.K_MAX:
        rescue = []
    return base + rescue


def kasa_fit(xs, ys):
    """Algebraic least-squares circle fit."""
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    sol, *_ = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    return cx, cy, np.sqrt(sol[2] + cx * cx + cy * cy)


def ray_edges(gray, cx, cy, r0):
    """Strongest outward radial gradient per ray within r0 +/- MAX_DR."""
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    pts = []
    for k in range(C.N_RAYS):
        th = 2 * np.pi * k / C.N_RAYS
        ux, uy = np.cos(th), np.sin(th)
        best, best_mag = None, 40.0
        for dr in np.arange(-C.MAX_DR, C.MAX_DR + 0.5, 0.5):
            x, y = cx + (r0 + dr) * ux, cy + (r0 + dr) * uy
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < gray.shape[0] and 0 <= xi < gray.shape[1]):
                continue
            mag = abs(sx[yi, xi] * ux + sy[yi, xi] * uy)
            if mag > best_mag:
                best, best_mag = (x, y), mag
        if best:
            pts.append(best)
    return pts


def validate(img, cx, cy, r0):
    """Accepts a candidate only if enough rays found a radial edge, then refits the circle."""
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 100, 100)
    pts = ray_edges(gray, cx, cy, r0)
    support = len(pts) / C.N_RAYS
    if len(pts) < C.MIN_RAYS or support < C.MIN_SUPPORT:
        return None
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    for _ in range(2):
        ncx, ncy, nr = kasa_fit(xs, ys)
        res = np.abs(np.hypot(xs - ncx, ys - ncy) - nr)
        keep = res < 3 * np.median(res) + 1.0
        if keep.sum() < C.MIN_RAYS:
            break
        xs, ys = xs[keep], ys[keep]
    ncx, ncy, nr = kasa_fit(xs, ys)
    if abs(nr - r0) > C.MAX_DR:
        ncx, ncy, nr = cx, cy, r0
    return {"cx": float(ncx), "cy": float(ncy), "r": float(nr), "support": float(support)}


def drop_contained(dets):
    """A centre lying inside a larger detected circle is internal structure, keep the larger."""
    return [
        d for d in dets
        if not any(
            o is not d and o["r"] > d["r"]
            and (d["cx"] - o["cx"]) ** 2 + (d["cy"] - o["cy"]) ** 2 < o["r"] ** 2
            for o in dets
        )
    ]


def drop_background_like(img, dets, thr=C.BG_Z_THR):
    """Disc colour within the background's own dispersion -> texture, not a coin."""
    if not dets:
        return dets
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.ones((h, w), bool)
    for d in dets:
        bg &= (xx - d["cx"]) ** 2 + (yy - d["cy"]) ** 2 > (1.15 * d["r"]) ** 2
    px = lab[bg].reshape(-1, 3)
    if len(px) < 100:  # almost no background visible: keep everything
        return dets
    med_bg = np.median(px, axis=0)
    mad_bg = np.median(np.abs(px - med_bg), axis=0) + 1e-6
    keep = []
    for d in dets:
        disc = (xx - d["cx"]) ** 2 + (yy - d["cy"]) ** 2 < (0.85 * d["r"]) ** 2
        med_d = np.median(lab[disc].reshape(-1, 3), axis=0)
        z = float(np.linalg.norm((med_d - med_bg) / (1.4826 * mad_bg)))
        if z >= thr:
            keep.append({**d, "bg_z": round(z, 2)})
    return keep


def drop_scale_outliers(dets, max_ratio=C.SCALE_MAX_RATIO):
    """Keeps the largest consistent radius band; acts only with a clear majority."""
    if len(dets) < 3:
        return dets
    best = None
    for lo in sorted(d["r"] for d in dets):
        inside = [d for d in dets if lo <= d["r"] <= lo * max_ratio]
        score = (len(inside), sum(d["support"] for d in inside))
        if best is None or score > best[0]:
            best = (score, inside)
    return best[1] if len(best[1]) * 2 > len(dets) else dets


def detect_coins(img):
    dets = [v for cx, cy, r in candidates(img) if (v := validate(img, cx, cy, r))]
    dets = drop_scale_outliers(drop_background_like(img, drop_contained(dets)))
    return sorted(dets, key=lambda d: (d["cy"], d["cx"]))
