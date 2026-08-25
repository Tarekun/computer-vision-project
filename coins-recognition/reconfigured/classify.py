"""Scene-level classification: per-coin cue scoring, pairwise size consistency,
and the joint beam-search labelling (S6.3-S6.5), with the fixes from ANALYSIS.md:

- The SIFT cue (now built from the canonical crop, see sift_cue.py) is folded in via
  the same z-scoring `template` already uses, instead of tiziano's hand-picked
  `2*score - 0.25` remap -- that remap assumed scores summed to 1 across classes, which
  is no longer true now that `sift_match_scores` returns a per-class rate.
- `TEMPLATE_OFFDIAG` comes from the reference model (`model["offdiag"]`), not a config
  constant -- see reference.py for how it's derived from the 8 reference images.
- The absolute bimetal scale anchor (`abs_anchor`) is now checked for consistency
  against the independently-seeded scene scale before being trusted; a single
  confidently-bimetal-but-wrong coin used to be able to silently distort every other
  coin's size term with no fallback.

The generic helpers (z-scoring, bimetal confidence, pairwise size scoring, the anchor
search itself, and the beam search) are unchanged from giacomo/tiziano and shared via
common/classify.py.

Feature extraction (`extract_features`) and fusion (`fuse_scene`) are split into two
functions: extraction is the expensive part (crops, 360-rotation shape search, SIFT)
and does not depend on any of the tunable fusion weights, while fusion is cheap
arithmetic + beam search and reads every tunable weight in config.py. This split is
also what tune.py's search relies on: it caches `extract_features` once per image and
re-runs only `fuse_scene` for each candidate weight vector.
"""
import cv2
import numpy as np

from . import config as C
from .appearance import crop_disc, nlm, decast, shape_scores, two_tone_family, first_pass, bimetal_step, ring_sign
from .sift_cue import sift_extract_with_nms, sift_match_scores, sift_confidence
from common.classify import zscore, bm_conf, abs_anchor, beam_assign


def extract_features(img, dets, model):
    """Per-coin cue extraction for every detection in `dets`. Returns (coins, circles)
    where `coins[i]` is None for an off-frame crop, else a dict of raw cues -- nothing
    here depends on the tunable fusion weights in config.py."""
    if not dets:
        return [], []

    atlas, group_ab = model["atlas"], model["group_ab"]
    proto, proto_sep = model["proto"], model["proto_sep"]
    sift_cache = model["sift_cache"]

    dcx = decast(img)
    circles = [(int(round(d["cx"])), int(round(d["cy"])), int(round(d["r"]))) for d in dets]
    coins = []
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    for (cx, cy, r), det in zip(circles, dets):
        crop = crop_disc(dcx, cx, cy, r)
        crop = None if crop is None else nlm(crop, C.TGT_NLM)
        if crop is None:
            coins.append(None)
            continue
        shp = shape_scores(crop, atlas)
        fam, fam_conf = two_tone_family(dcx, cx, cy, r, proto, proto_sep)
        disc = crop_disc(dcx, cx, cy, r)

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        sift_des = sift_extract_with_nms(crop_gray)
        sift_scores, sift_raw = sift_match_scores(sift_des, bf_matcher, sift_cache)
        sift_conf = sift_confidence(sift_scores)

        coins.append({
            "r": r, "bg_z": float(det.get("bg_z", 9.9)), "shape": shp,
            "pred0": first_pass(shp, crop, group_ab), "fam": fam, "fam_conf": fam_conf,
            "step": bimetal_step(disc), "ring": ring_sign(disc),
            "sift_scores": sift_scores, "sift_conf": sift_conf,
        })
    return coins, circles


def fuse_scene(coins, circles, model):
    """Labels every coin in `coins` (as produced by extract_features), jointly, using
    the current tunable weights in config.py. This is the only function tune.py's
    search needs to re-run per candidate weight vector."""
    if not coins:
        return []

    offdiag = model["offdiag"]
    full = [c for c in coins if c]
    if not full:
        return ["20cent"] * len(coins)

    # scene scale seeded by the first pass; per-class radius medians for the
    # same-class consistency term
    sscale = float(np.median([c["r"] / C.DIAMETER_MM[c["pred0"]] for c in full]))
    rclass = {}
    for c in full:
        rclass.setdefault(c["pred0"], []).append(c["r"])
    rclass = {k: float(np.median(v)) for k, v in rclass.items()}

    s_abs = abs_anchor(full)
    if s_abs is not None and abs(np.log(s_abs / sscale)) > C.ANCHOR_CONSISTENCY_TOL:
        # the "confirmed bimetal" anchor disagrees too much with the independently
        # seeded scale to be trusted scene-wide -- fall back to the seed alone.
        s_abs = None

    unaries, radii = [], []
    for coin in coins:
        if coin is None:
            continue
        template = {
            c: C.TEMPLATE_W * v for c, v in zscore(
                {c: coin["shape"][c] - C.TEMPLATE_ALPHA * offdiag[c] for c in C.CLASSES}
            ).items()
        }
        fam_sc = {c: 0.0 for c in C.CLASSES}
        if coin["fam"]:
            for c in C.CLASSES:
                fam_sc[c] = (C.FAM_POS if C.FAMILY[c] == coin["fam"] else -C.FAM_NEG) * coin["fam_conf"]
        bm = bm_conf(coin["step"])
        bim = {c: (C.BIM_POS if C.FAMILY[c] == "bimetal" else -C.BIM_NEG) * bm for c in C.CLASSES}
        if bm > 0:
            hi, lo2 = (("1euro", "2euro") if coin["ring"] == "gold-ring" else ("2euro", "1euro"))
            bim[hi] += C.BIM_RING * bm
            bim[lo2] -= 0.5 * C.BIM_RING * bm
        # family and bimetal are competing material explanations: mutual damping
        if coin["fam"] and bm > 0:
            fs = max(0.05, 1.0 - C.FAM_BIM_SUP * bm)
            bs = max(C.BIM_FAM_MIN, 1.0 - C.BIM_FAM_SUP * coin["fam_conf"] * (1.0 - 0.5 * bm))
            fam_sc = {c: v * fs for c, v in fam_sc.items()}
            bim = {c: v * bs for c, v in bim.items()}
        sift_sc = {c: C.W_SIFT * v for c, v in zscore(coin["sift_scores"]).items()} if any(
            coin["sift_scores"].values()
        ) else {c: 0.0 for c in C.CLASSES}
        sift_norm = C.SIFT_FLOOR + (1 - C.SIFT_FLOOR) * coin["sift_conf"]
        sift_sc = {c: v * sift_norm for c, v in sift_sc.items()}
        # bg_z legibility damping of the appearance cues (shape + family + sift)
        damp = C.C1_FLOOR + (1 - C.C1_FLOOR) * float(
            np.clip((coin["bg_z"] - C.C1_ZLO) / (C.C1_ZHI - C.C1_ZLO), 0, 1)
        )
        d_obs = coin["r"] / sscale
        unary = {}
        for c in C.CLASSES:
            u = (
                damp * (fam_sc[c] + sift_sc[c] + template[c]) + bim[c]
                - C.W_ORDER * min(abs(np.log(d_obs / C.DIAMETER_MM[c])) / C.LOGMAX, 1.0)
                - C.W_SAME * (
                    min(abs(np.log(coin["r"] / rclass[c])) / C.LOGMAX, 1.0) if c in rclass else 0.0
                )
            )
            if s_abs is not None:
                u -= C.ANCHOR_W * min(abs(np.log((coin["r"] / s_abs) / C.DIAMETER_MM[c])) / C.LOGMAX, 1.0)
            unary[c] = u
        unaries.append(unary)
        radii.append(coin["r"])

    beam_labels = beam_assign(unaries, radii)
    labels, k = [], 0
    for i, coin in enumerate(coins):
        if coin is None:  # off-frame crop: nearest legal diameter
            d_obs = circles[i][2] / sscale
            labels.append(min(C.CLASSES, key=lambda c: abs(np.log(d_obs / C.DIAMETER_MM[c]))))
        else:
            labels.append(beam_labels[k])
            k += 1
    return labels


def classify_scene(img, dets, model):
    """Labels every detection in `dets`, jointly, using the reference `model`."""
    coins, circles = extract_features(img, dets, model)
    return fuse_scene(coins, circles, model)
