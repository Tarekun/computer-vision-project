"""Generic scene-level classification helpers shared by giacomo and tiziano:
z-scoring, bimetal confidence, pairwise size-ratio consistency, the absolute
bimetal scale anchor, and the joint beam-search labelling (S6.3-S6.5).

These depend only on constants confirmed identical between the two packages
(CLASSES, DIAMETER_MM, RATIO_TOL, SAME_TOL, REL_W, BEAM_K, BIM_START, BIM_FULL,
ANCHOR_CONF). The actual per-coin cue fusion (`classify_scene`) differs between
the two packages -- giacomo combines shape/family/bimetal, tiziano also folds
in a SIFT cue and uses its own W_ORDER/W_SAME -- so it stays local to each
package's classify.py, which imports these helpers from here.
"""
import numpy as np

from . import config as C


def zscore(d):
    v = np.array([d[c] for c in C.CLASSES], np.float64)
    s = float(v.std())
    if s < 1e-6:
        return {c: 0.0 for c in C.CLASSES}
    return {c: float((d[c] - v.mean()) / s) for c in C.CLASSES}


def bm_conf(step):
    return float(np.clip((step - C.BIM_START) / (C.BIM_FULL - C.BIM_START), 0, 1))


def pair_score(ri, rj, li, lj):
    """Pairwise size consistency (S2.3: radius ratios are metric within a scene)."""
    obs = np.log(max(1e-6, ri / rj))
    exp = np.log(C.DIAMETER_MM[li] / C.DIAMETER_MM[lj])
    s = -1.15 * min(abs(obs - exp) / C.RATIO_TOL, 1.8)
    if li == lj:
        s += 0.55 * max(0.0, 1.0 - abs(obs) / C.SAME_TOL)
    elif abs(obs) < C.SAME_TOL:
        s -= 0.35
    if (obs > C.SAME_TOL and exp < -C.SAME_TOL) or (obs < -C.SAME_TOL and exp > C.SAME_TOL):
        s -= 0.9  # hard ordering violation
    return s


def abs_anchor(coins):
    """Absolute scene scale from a confirmed bimetal (S6.4-6.5)."""
    best = None
    for c in coins:
        if bm_conf(c["step"]) < C.ANCHOR_CONF:
            continue
        for cls in ("1euro", "2euro"):
            s = c["r"] / C.DIAMETER_MM[cls]
            cost = sum(
                min(abs(np.log((cc["r"] / s) / C.DIAMETER_MM[k])) for k in C.CLASSES)
                for cc in coins
            )
            if best is None or cost < best[0]:
                best = (cost, s)
    return None if best is None else best[1]


def beam_assign(unaries, radii):
    beam = [([], 0.0)]
    for i, unary in enumerate(unaries):
        nxt = []
        for labels, score in beam:
            for lab in C.CLASSES:
                p = sum(
                    C.REL_W * pair_score(radii[i], radii[j], lab, prev)
                    for j, prev in enumerate(labels)
                )
                nxt.append((labels + [lab], score + unary[lab] + p))
        nxt.sort(key=lambda x: x[1], reverse=True)
        beam = nxt[:C.BEAM_K]
    return beam[0][0]
