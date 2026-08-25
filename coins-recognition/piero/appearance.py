"""Utility functions for the S11.3-bis operating classifier (Section 4 of
solution.ipynb): colour de-cast, canonical crops, the per-class reference
edge filter/blur, shape scoring, and the colour/bimetal cues.

The notebook calls this section "SELF-CONTAINED": it deliberately redefines
its own copy of helpers (decast, crop, Hough-locate-the-reference-coin, ...)
independent from Section 2's GHT machinery, rather than reusing it. This
module preserves that separation intentionally -- see piero/README.md.
"""
import cv2
import numpy as np

from . import config as C


def decast_11(bgr):
    out = bgr.astype(np.float64).copy()
    for ch in range(3):
        out[:, :, ch] = C.CASTG11[ch] * out[:, :, ch] + C.CASTO11[ch]
    return np.clip(out, 0, 255).astype(np.uint8)


def dn_post_11(img, h, sig):
    if h > 0:
        img = cv2.fastNlMeansDenoisingColored(
            img, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
        )
    if sig > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sig)
    return img


def croprz_11(bgr, cx, cy, r):
    R = int(r)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = max(0, cx - R), max(0, cy - R), min(w, cx + R), min(h, cy + R)
    patch = bgr[y0:y1, x0:x1]
    return None if patch.size == 0 else cv2.resize(patch, (C.CANON11, C.CANON11), interpolation=cv2.INTER_CUBIC)


def normref_11(bgr, cx, cy, r):
    d = croprz_11(bgr, cx, cy, r)
    return None if d is None else dn_post_11(d, C.NLM_REF11, C.GSIG_REF11)


def href_11(gray):
    """Locate the single coin in a clean reference image."""
    c = cv2.HoughCircles(
        cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, 1.2, 300,
        param1=150, param2=30, minRadius=40, maxRadius=300,
    )
    return tuple(map(int, np.round(c[0][0])))


def rot_11(img, theta):
    M = cv2.getRotationMatrix2D((C.CANON11 / 2.0, C.CANON11 / 2.0), theta, 1.0)
    return cv2.warpAffine(
        img, M, (C.CANON11, C.CANON11), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def grad_11(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    return np.arctan2(gy, gx)


def autocanny_11(gray, s=0.33):
    v = np.median(gray)
    return cv2.Canny(gray, int(max(0, (1 - s) * v)), int(min(255, (1 + s) * v))) > 0


def adaptive_11(gray, mask, target, lo_ratio=0.5, aperture=3, L2=False, iters=22, hi=(5.0, 400.0)):
    lo_t, hi_t, best = hi[0], hi[1], None
    for _ in range(iters):
        mid = 0.5 * (lo_t + hi_t)
        e = (cv2.Canny(gray, int(lo_ratio * mid), int(mid), apertureSize=aperture, L2gradient=L2) > 0) & mask
        n = int(e.sum())
        best = (e, n, mid)
        if n > target:
            lo_t = mid
        else:
            hi_t = mid
    return best


def chroma_11(canon):
    lab = cv2.cvtColor(canon, cv2.COLOR_BGR2LAB).astype(np.float32)
    cg = np.zeros((C.CANON11, C.CANON11), np.float32)
    for ch in (1, 2):
        gx = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 0, 1, 3)
        cg += np.hypot(gx, gy)
    rs = np.arange(0.1, 0.95, 0.05)
    prof = np.array([cg[(C.RHO11 >= t - 0.025) & (C.RHO11 < t + 0.025)].mean() for t in rs])
    prof = prof / (prof.max() + 1e-6)
    return float(prof[(rs >= 0.60) & (rs <= 0.78)].max() - np.median(prof[(rs >= 0.25) & (rs <= 0.55)]))


def ab_sig_11(canon):
    lab = cv2.cvtColor(canon, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.array([np.median(lab[:, :, 1][C.DMASK11]), np.median(lab[:, :, 2][C.DMASK11])])


def colorcue_11(crop, group_ab):
    ab = ab_sig_11(crop)
    d = {g: float(np.linalg.norm(ab - group_ab[g])) for g in C.GROUPS11}
    return min(d, key=d.get), d


def favour_11(shp, crop, group_ab):
    cr = chroma_11(crop)
    reg = "strong-bi" if cr >= C.TH_BI_S11 else ("weak-mono" if cr < C.TH_BI_W11 else "borderline")
    cg, cd = colorcue_11(crop, group_ab)
    spread = float(np.std(list(shp.values()))) + 1e-6
    bi = float(np.clip((cr - C.TH_BI_W11) / (C.TH_BI_S11 - C.TH_BI_W11), 0.0, 1.0))
    fav = {}
    for n in C.REF_NAMES:
        b = C.COLOR_FAV11 * spread if C.MATERIAL11[n] == cg else 0.0
        if C.MATERIAL11[n] == "bimetal":
            b += C.BI_FAV11 * bi * spread
        fav[n] = b
    score = {n: shp[n] + fav[n] for n in C.REF_NAMES}
    pred = max(score, key=score.get)
    return dict(
        pred=pred, value=C.VALUE_EUR[pred], chroma=cr, regime=reg,
        color_group=cg, cdist=cd, bi_ev=bi, fav=fav,
    )


def epfilter3b(canon, nlm_h=7):
    f = C.REF_FILTER3b
    if f == "bilateral":
        return cv2.bilateralFilter(canon, C.BILAT_D3b, C.BILAT_SC3b, C.BILAT_SS3b)
    if f == "edgepreserve":
        return cv2.edgePreservingFilter(canon, flags=cv2.RECURS_FILTER, sigma_s=C.EP_SS3b, sigma_r=C.EP_SR3b)
    if f == "median":
        return cv2.medianBlur(canon, C.MED_K3b | 1)
    return canon if nlm_h <= 0 else cv2.fastNlMeansDenoisingColored(
        canon, None, h=float(nlm_h), hColor=float(nlm_h), templateWindowSize=7, searchWindowSize=21
    )


def edgesfor3b(name, gray, mask, default_target):
    p = C.EDGE_PARAMS3b.get(name, {})
    ap, L2, lor = p.get("aperture", 3), p.get("L2", False), p.get("lo_ratio", 0.5)
    if "low" in p and "high" in p:
        lo, hi = int(p["low"]), int(p["high"])
        e = (cv2.Canny(gray, lo, hi, apertureSize=ap, L2gradient=L2) > 0) & mask
        return e, int(e.sum()), lo, hi, "manual"
    e, n, hi = adaptive_11(gray, mask, p.get("target", default_target), lo_ratio=lor, aperture=ap, L2=L2)
    return e, n, int(lor * hi), int(hi), "auto"


def blurmask3b(regions):
    m = np.zeros((C.CANON11, C.CANON11), bool)
    yy, xx = np.ogrid[:C.CANON11, :C.CANON11]
    c = C.CANON11 / 2.0
    for reg in regions:
        k = reg[0]
        if k == "ring":
            rho = np.hypot(xx - c, yy - c) / (C.CANON11 / 2.0)
            m |= (rho >= reg[1]) & (rho < reg[2])
        elif k == "disc":
            m |= ((xx - reg[1]) ** 2 + (yy - reg[2]) ** 2) <= reg[3] ** 2
        elif k == "box":
            m[max(0, reg[2]):reg[4], max(0, reg[1]):reg[3]] = True
    return m & C.DMASK11


def apply_blur3b(disc, name):
    spec = C.BLUR3b.get(name)
    if not spec or not spec.get("regions"):
        return disc, np.zeros((C.CANON11, C.CANON11), bool)
    m = blurmask3b(spec["regions"])
    if not m.any():
        return disc, m
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), C.BLUR_FEATHER3b)[..., None]
    b = cv2.GaussianBlur(disc, (0, 0), float(spec.get("sigma", C.BLUR_SIGMA3b)))
    out = disc.astype(np.float32) * (1 - alpha) + b.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8), m


def scores3b(crop, atlas):
    angs = [grad_11(cv2.cvtColor(rot_11(crop, th), cv2.COLOR_BGR2GRAY)) for th in C.ANGLES11]
    out = {}
    for n in C.REF_NAMES:
        cp, ra = atlas[n]["cp"], atlas[n]["ang"]
        ntot = int(cp.sum())
        bf, bi = -1.0, 0
        for k, a in enumerate(angs):
            frac = float((np.cos(2 * (a[cp] - ra[cp])) >= C.TAU11).mean()) if ntot else 0.0
            if frac > bf:
                bf, bi = frac, k
        out[n] = (bf, ntot, C.ANGLES11[bi])
    return out


def softmax11(score):
    v = np.array([score[n] for n in C.REF_NAMES], float)
    v -= v.max()
    e = np.exp(v / C.SOFTMAX_T11)
    p = e / e.sum()
    return {n: float(p[i]) for i, n in enumerate(C.REF_NAMES)}
