"""Builds the S11.3-bis reference model from reference_set: the colour atlas
(material centroids in Lab a*,b*) and the per-class blurred edge atlas.

This is this implementation's `initialize_ght` stand-in -- a one-off model
built from the reference images and reused for every target image, mirroring
giacomo/reference.py and tiziano/reference.py's role for their own designs.
"""
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .appearance import (
    href_11, croprz_11, normref_11, ab_sig_11, epfilter3b, apply_blur3b,
    grad_11, autocanny_11, edgesfor3b,
)


def build_reference_model(reference_dir):
    """Builds {group_ab, atlas, blur_mask} from the 8 reference images."""
    reference_dir = Path(reference_dir)

    # ---- colour atlas (material centroids in Lab a*,b*, from the RAW references) ----
    ref_ab = {}
    for name in C.REF_NAMES:
        bgr = cv2.imread(str(reference_dir / f"{name}.jpg"))
        cx, cy, r = href_11(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        ref_ab[name] = ab_sig_11(normref_11(bgr, cx, cy, r))
    group_ab = {
        g: np.mean([ref_ab[n] for n in C.REF_NAMES if C.MATERIAL11[n] == g], axis=0)
        for g in C.GROUPS11
    }

    # ---- blurred reference edge atlas: per-coin localized blur, then per-coin Canny ----
    ref_atlas, blur_mask = {}, {}
    for name in C.REF_NAMES:
        bgr = cv2.imread(str(reference_dir / f"{name}.jpg"))
        cx, cy, r = href_11(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        disc = epfilter3b(croprz_11(bgr, cx, cy, r), C.NLM_REF3b[name])
        disc, m = apply_blur3b(disc, name)
        blur_mask[name] = m
        ref_atlas[name] = {"disc": disc, "ang": grad_11(cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY))}

    before = {
        n: autocanny_11(cv2.cvtColor(ref_atlas[n]["disc"], cv2.COLOR_BGR2GRAY)) & C.DMASK11
        for n in C.REF_NAMES
    }
    n_target = int(np.median([int(before[n].sum()) for n in C.REF_NAMES]))

    atlas = {}
    for name in C.REF_NAMES:
        gray = cv2.cvtColor(ref_atlas[name]["disc"], cv2.COLOR_BGR2GRAY)
        e, cnt, lo, hi, mode = edgesfor3b(name, gray, C.DMASK11, n_target)
        atlas[name] = {"cp": e, "ang": ref_atlas[name]["ang"], "n": cnt, "lo": lo, "thr": hi, "mode": mode}

    return {"group_ab": group_ab, "atlas": atlas, "blur_mask": blur_mask}
