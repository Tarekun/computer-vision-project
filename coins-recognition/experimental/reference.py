"""Builds the classification model from the reference_set images: the shape atlas,
the two colour prototypes, the data-derived per-class shape bias (`offdiag`), and the
SIFT descriptor cache -- everything is built from these same 8 reference images only.

Forked from reconfigured/reference.py. EXPERIMENTS.md item 1 ("multi-exemplar
robustness from the single reference photo") tried building the shape atlas from a
small jittered ensemble of each reference photo (`_jittered_atlas`) instead of one
raw read (`_single_crop_atlas`) -- measured WORSE (43.55% vs. 53.43% mean held-out
amount_acc) and reverted; `C.JITTERED_ATLAS` (off by default) switches between the
two, kept for the write-up.

Two fixes carried over from reconfigured/ (one kept, one attempted-and-reverted
there, see ANALYSIS.md S3/S4 and reconfigured/README.md's "what we tried and undid"
section):

- `TEMPLATE_OFFDIAG` (how much each class's atlas tends to false-match a *different*
  denomination) used to be 8 hand-typed constants. Here it's computed directly: every
  reference coin's own canonical crop is scored against every OTHER class's atlas, and
  the average of those cross-class scores becomes that class's offdiag bias. This is
  exactly what the bias term is meant to represent, derived from data instead of guessed,
  and it only uses the 8 reference images -- no additional data. Measured improvement
  (combined with the aspect-preserving crop fix): 48.59% -> 49.30% full-set amount
  accuracy (see reconfigured/ablate.py).

- We ALSO tried building the shape atlas from `decast(ref_bgr)` instead of the raw
  photo, to fix the asymmetry ANALYSIS.md flags (giacomo decasts the target before
  shape-scoring it against the atlas, and already decasts the reference photo for the
  colour-prototype cue, but never decasts the atlas-building crop itself). Measured in
  isolation, this made things WORSE (49.30% -> 45.77%): `REF_EDGE`'s per-class Canny
  thresholds were implicitly calibrated against the *raw* reference photo's contrast, and
  decasting shifts that contrast enough to degrade the atlas without also retuning
  `REF_EDGE`. So this fix is reverted here -- the colour-prototype crop still uses
  `decast(ref_bgr)` (matching giacomo), but the atlas-building crop uses the raw photo,
  same as giacomo/tiziano. Jointly retuning `REF_EDGE` alongside a decast atlas is flagged
  as follow-up work rather than attempted here (it requires rebuilding the atlas -- and
  therefore re-running shape-score feature extraction over all 142 images -- per
  candidate threshold, which is much more expensive than tuning a fusion weight).
"""
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .appearance import crop_disc, nlm, grad_dirs, decast, dev_tone, shape_scores, photometric_jitter
from .sift_cue import calibrate_sift_templates


def find_ref_coin(gray):
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, 1.2, 300,
        param1=150, param2=30, minRadius=40, maxRadius=300,
    )
    return tuple(map(int, np.round(circles[0][0])))


def apply_ref_blur(disc, name):
    # EXPERIMENTS.md item 6b: DISABLE_REF_BLUR lets ref_blur_ablation.py measure this
    # hardcoded per-photo pixel patch's actual effect. Off (blur enabled) by default.
    if C.DISABLE_REF_BLUR or name not in C.REF_BLUR:
        return disc
    sigma, (bx, by, br) = C.REF_BLUR[name]
    mask = (((C.GRID_X - bx) ** 2 + (C.GRID_Y - by) ** 2) <= br ** 2) & C.DMASK
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 6)[..., None]
    blurred = cv2.GaussianBlur(disc, (0, 0), float(sigma))
    return (
        disc.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)


def _single_crop_atlas(raw_crop, name, ref_edge):
    """Original single-instance atlas construction (giacomo/tiziano/reconfigured):
    Canny + gradient directions on the one raw reference crop, no jitter."""
    disc = apply_ref_blur(nlm(raw_crop, C.REF_NLM[name]), name)
    g = cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY)
    lo, hi = ref_edge[name]
    return {"cp": (cv2.Canny(g, lo, hi) > 0) & C.DMASK, "ang": grad_dirs(g)}


def _jittered_atlas(raw_crop, name, ref_edge):
    """EXPERIMENTS.md item 1 (measured WORSE, off by default -- see JITTERED_ATLAS):
    builds one class's shape atlas as a robust vote across small photometric jitters
    of its single reference crop, instead of reading one raw instance. A control
    point is kept if present in >= ATLAS_VOTE_FRAC of the jittered Canny maps; its
    direction is the circular mean of the jittered angle maps (weighted by which
    jitters actually produced that control point)."""
    lo, hi = ref_edge[name]
    votes = np.zeros((C.CANON, C.CANON), np.float32)
    sin_sum = np.zeros((C.CANON, C.CANON), np.float32)
    cos_sum = np.zeros((C.CANON, C.CANON), np.float32)
    for b, k, s in C.REF_JITTERS:
        variant = photometric_jitter(raw_crop, b, k, s)
        disc = apply_ref_blur(nlm(variant, C.REF_NLM[name]), name)
        g = cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY)
        cp = (cv2.Canny(g, lo, hi) > 0) & C.DMASK
        ang = grad_dirs(g)
        votes += cp.astype(np.float32)
        sin_sum += np.sin(ang) * cp
        cos_sum += np.cos(ang) * cp
    cp_final = (votes / len(C.REF_JITTERS)) >= C.ATLAS_VOTE_FRAC
    ang_final = np.arctan2(sin_sum, cos_sum)
    return {"cp": cp_final, "ang": ang_final}


def build_reference_model(reference_dir):
    """Builds {atlas, group_ab, proto, proto_sep, offdiag, sift_cache} from the 8
    reference coin images."""
    reference_dir = Path(reference_dir)
    atlas = {}
    ref_ab = {}
    ref_discs = {}  # target-like processed canonical crop per class, reused for offdiag + SIFT
    protos_raw = {"copper": [], "gold": []}

    for name in C.CLASSES:
        ref_bgr = cv2.imread(str(reference_dir / f"{name}.jpg"))
        rcx, rcy, rr = find_ref_coin(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
        dcx_ref = decast(ref_bgr)  # colour-prototype side only (matches giacomo; see module docstring)

        # EXPERIMENTS.md item 4: the atlas-building crop optionally uses a decasted
        # source image + retuned per-class Canny thresholds (C.DECAST_ATLAS, off by
        # shipped default); ref_discs (offdiag's "other class" side + SIFT domain)
        # always come from the RAW photo, decoupled from this choice.
        ref_edge = C.REF_EDGE_RETUNED if C.DECAST_ATLAS else C.REF_EDGE
        atlas_src = decast(ref_bgr) if C.DECAST_ATLAS else ref_bgr
        raw_crop_atlas = crop_disc(atlas_src, rcx, rcy, rr)
        atlas[name] = (
            _jittered_atlas(raw_crop_atlas, name, ref_edge) if C.JITTERED_ATLAS
            else _single_crop_atlas(raw_crop_atlas, name, ref_edge)
        )
        raw_crop = crop_disc(ref_bgr, rcx, rcy, rr)
        ref_discs[name] = nlm(raw_crop, C.TGT_NLM)

        # gentler chain for the COLOUR side of the references (tones, not edges)
        cd = cv2.GaussianBlur(nlm(raw_crop, 1), (0, 0), 0.3)
        lab = cv2.cvtColor(cd, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_ab[name] = np.array(
            [np.median(lab[:, :, 1][C.DMASK]), np.median(lab[:, :, 2][C.DMASK])]
        )

        if C.FAMILY[name] != "bimetal":
            dev, _ = dev_tone(dcx_ref, rcx, rcy, rr)
            protos_raw[C.FAMILY[name]].append(dev)

    group_ab = {
        g: np.mean([ref_ab[n] for n in C.CLASSES if C.FAMILY[n] == g], axis=0)
        for g in ("copper", "gold", "bimetal")
    }
    proto = {g: np.mean(v, axis=0) for g, v in protos_raw.items()}
    proto_sep = float(np.hypot(*(proto["copper"] - proto["gold"])))

    # Data-derived per-class shape bias: how much does class c's atlas match OTHER
    # reference coins' crops on average? Replaces the old hand-typed TEMPLATE_OFFDIAG.
    cross_scores = {other: shape_scores(ref_discs[other], atlas) for other in C.CLASSES}
    offdiag = {
        c: float(np.mean([cross_scores[other][c] for other in C.CLASSES if other != c]))
        for c in C.CLASSES
    }

    sift_cache = calibrate_sift_templates(ref_discs)

    return {
        "atlas": atlas, "group_ab": group_ab, "proto": proto, "proto_sep": proto_sep,
        "offdiag": offdiag, "sift_cache": sift_cache,
    }
