"""Per-coin appearance cues: canonical crops, colour normalisation, shape-based
matching, and the colour/bimetal tone cues used by classification.

Shared verbatim by giacomo and tiziano (identical cues in both). See
giacomo/README.md S6.1-S6.3 or tiziano/README.md for the reasoning behind each
function.
"""
import cv2
import numpy as np

from . import config as C


def decast(bgr):
    """Fixed affine colour de-cast (gain/offset per BGR channel, calibrated on the references)."""
    out = bgr.astype(np.float64).copy()
    for ch in range(3):
        out[:, :, ch] = C.CAST_GAIN[ch] * out[:, :, ch] + C.CAST_OFF[ch]
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_disc(bgr, cx, cy, r):
    """Square crop around the coin, resized to the canonical size (removes scale, S2.3)."""
    R = int(r)
    h, w = bgr.shape[:2]
    patch = bgr[max(0, cy - R):min(h, cy + R), max(0, cx - R):min(w, cx + R)]
    return None if patch.size == 0 else cv2.resize(
        patch, (C.CANON, C.CANON), interpolation=cv2.INTER_CUBIC
    )


def nlm(img, h):
    """Non-local means (S3.4): cleans gradient directions for matching."""
    return img if h <= 0 else cv2.fastNlMeansDenoisingColored(
        img, None, h=float(h), hColor=float(h), templateWindowSize=7, searchWindowSize=21
    )


def rot(img, theta):
    M = cv2.getRotationMatrix2D((C.CANON / 2.0, C.CANON / 2.0), theta, 1.0)
    return cv2.warpAffine(
        img, M, (C.CANON, C.CANON), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def grad_dirs(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    return np.arctan2(gy, gx)


def shape_scores(crop, atlas):
    """S6.2: fraction of control points whose gradient direction agrees (polarity-invariant),
    maximised over the quantised rotation hypotheses (S6.4)."""
    angs = [grad_dirs(cv2.cvtColor(rot(crop, th), cv2.COLOR_BGR2GRAY)) for th in C.ANGLES]
    out = {}
    for name in C.CLASSES:
        cp, ra = atlas[name]["cp"], atlas[name]["ang"]
        out[name] = max(float((np.cos(2 * (a[cp] - ra[cp])) >= C.TAU).mean()) for a in angs)
    return out


def lnorm(disc):
    """Per-disc luminance 2-98% stretch (L* only): restores dynamics without touching chroma."""
    lab = cv2.cvtColor(disc, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    lo, hi = np.percentile(L[C.DMASK], [2, 98])
    lab[:, :, 0] = np.clip((L - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def coin_tone(disc):
    lab = cv2.cvtColor(lnorm(disc), cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = C.RHO < 0.6  # inner disc avoids the bimetal ring
    return np.array([np.median(lab[:, :, 1][inner]), np.median(lab[:, :, 2][inner])])


def bg_tone(dcx, cx, cy, r):
    """Median (a*, b*) of the peri-coin annulus: the relative-colour anchor."""
    h, w = dcx.shape[:2]
    Y, X = np.ogrid[:h, :w]
    d = np.hypot(X - cx, Y - cy)
    ann = (d >= 1.10 * r) & (d <= 1.35 * r)
    if ann.sum() < 50:
        return None
    lab = cv2.cvtColor(dcx, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.array([np.median(lab[:, :, 1][ann]), np.median(lab[:, :, 2][ann])])


def bimetal_step(disc):
    """Core/ring chroma step: a 1-D edge along the radial colour profile (S4.1)."""
    lab = cv2.cvtColor(lnorm(disc), cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1], lab[:, :, 2]
    core, ring = C.RHO < 0.55, (C.RHO >= 0.55) & C.DMASK
    return float(np.hypot(
        np.median(a[core]) - np.median(a[ring]),
        np.median(b[core]) - np.median(b[ring]),
    ))


def ring_sign(disc):
    lab = cv2.cvtColor(disc, cv2.COLOR_BGR2LAB).astype(np.float32)
    core_b = float(lab[C.RHO < 0.45][:, 2].mean())
    ring_b = float(lab[(C.RHO > 0.75) & (C.RHO < 0.95)][:, 2].mean())
    return "gold-ring" if ring_b > core_b else "silver-ring"


def dev_tone(dcx, cx, cy, r):
    """Background-anchored coin tone deviation, plus the bimetal chroma step."""
    disc = crop_disc(dcx, cx, cy, r)
    if disc is None:
        return None, None
    bg = bg_tone(dcx, cx, cy, r)
    return coin_tone(disc) - (bg if bg is not None else 0.0), bimetal_step(disc)


def two_tone_family(dcx, cx, cy, r, proto, proto_sep):
    """Nearest-of-two-prototypes on the background-anchored tone; abstains on a likely bimetal."""
    dev, step = dev_tone(dcx, cx, cy, r)
    if dev is None or step >= C.BIMETAL_ABSTAIN_STEP:
        return None, 0.0
    dc = float(np.hypot(*(dev - proto["copper"])))
    dg = float(np.hypot(*(dev - proto["gold"])))
    conf = float(np.clip(abs(dc - dg) / (proto_sep + 1e-6), 0, 1))
    return ("copper" if dc < dg else "gold"), conf


def ring_chroma(crop):
    """Normalised chroma-gradient radial profile: a bimetal ring leaves a bump at rho 0.60-0.78."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    cg = np.zeros((C.CANON, C.CANON), np.float32)
    for ch in (1, 2):
        gx = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(lab[:, :, ch], cv2.CV_32F, 0, 1, 3)
        cg += np.hypot(gx, gy)
    rs = np.arange(0.1, 0.95, 0.05)
    prof = np.array([cg[(C.RHO >= t - 0.025) & (C.RHO < t + 0.025)].mean() for t in rs])
    prof = prof / (prof.max() + 1e-6)
    return float(
        prof[(rs >= 0.60) & (rs <= 0.78)].max()
        - np.median(prof[(rs >= 0.25) & (rs <= 0.55)])
    )


def first_pass(shp, crop, group_ab):
    """Seed prediction (shape + soft colour favour): only initialises the scene scale (S6.4)."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = np.array([np.median(lab[:, :, 1][C.DMASK]), np.median(lab[:, :, 2][C.DMASK])])
    group = min(group_ab, key=lambda g: float(np.linalg.norm(ab - group_ab[g])))
    spread = float(np.std(list(shp.values()))) + 1e-6
    bi = float(np.clip((ring_chroma(crop) - 0.10) / 0.30, 0.0, 1.0))
    fav = {
        n: (0.6 * spread if C.FAMILY[n] == group else 0.0)
        + (1.2 * bi * spread if C.FAMILY[n] == "bimetal" else 0.0)
        for n in C.CLASSES
    }
    return max(C.CLASSES, key=lambda n: shp[n] + fav[n])
