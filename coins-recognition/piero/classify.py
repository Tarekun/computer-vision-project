"""Per-scene classification (S11.3-bis, DEC-15): whole-disc edge-orientation
shape match-% + material/bi-metal colour favour + relative-size re-rank.
"""
import numpy as np

from . import config as C
from .appearance import croprz_11, epfilter3b, scores3b, favour_11, softmax11, decast_11


def crop_filt3b(dcx, cx, cy, r):
    d = croprz_11(dcx, cx, cy, r)
    return None if d is None else epfilter3b(d, C.NLM_ZOOM11)


def classify_scene3b(circ, dcx, model):
    atlas, group_ab = model["atlas"], model["group_ab"]

    coins = []  # stage 1 -- zoom + shape + colour favour
    for (cx, cy, r) in circ:
        crop = crop_filt3b(dcx, cx, cy, r)
        if crop is None:
            continue
        sa = scores3b(crop, atlas)
        shp = {n: sa[n][0] for n in C.REF_NAMES}
        m0 = favour_11(shp, crop, group_ab)
        base = {n: shp[n] + m0["fav"][n] for n in C.REF_NAMES}
        coins.append(dict(cx=cx, cy=cy, r=r, crop=crop, sa=sa, base=base, shp=shp, pred0=m0["pred"]))
    if not coins:
        return [], None

    s = float(np.median([c["r"] / C.DIAMETER_MM[c["pred0"]] for c in coins]))  # stage 2 -- scene scale px/mm
    rad = {}
    for c in coins:
        rad.setdefault(c["pred0"], []).append(c["r"])
    rclass = {n: float(np.median(v)) for n, v in rad.items()}

    for c in coins:  # stage 3 -- two soft size penalties, re-rank
        d_obs = c["r"] / s
        final = {}
        for n in C.REF_NAMES:
            p_order = min(abs(np.log(d_obs / C.DIAMETER_MM[n])) / C.LOGMAX11, 1.0)
            p_same = min(abs(np.log(c["r"] / rclass[n])) / C.LOGMAX11, 1.0) if n in rclass else 0.0
            final[n] = c["base"][n] - C.W_ORDER11 * p_order - C.W_SAME11 * p_same
        c["d_obs"] = d_obs
        c["final"] = final
        c["prob"] = softmax11(final)
        c["pred"] = max(final, key=final.get)
        c["value"] = C.VALUE_EUR[c["pred"]]
        c["theta"] = c["sa"][c["pred"]][2]
        c["changed"] = c["pred"] != c["pred0"]
    return coins, s


def classify(bgr, coins, model):
    """Public interface: one denomination per detected coin, in input order."""
    if not coins:
        return []
    scene, _ = classify_scene3b(coins, decast_11(bgr), model)
    pred_by_coin = {(c["cx"], c["cy"], c["r"]): c["pred"] for c in scene}
    return [pred_by_coin.get(tuple(c), "1cent") for c in coins]  # default lone-failure -> conservative
