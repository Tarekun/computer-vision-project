"""Builds the base classification model from the reference_set images: the shape
atlas (control points + gradient directions per class, S6.2) and the two
colour prototypes used by the family cue (S6.1).

Shared by giacomo and tiziano (identical construction in both); tiziano wraps
this to additionally attach its SIFT descriptor cache (see tiziano/reference.py).

This is the `initialize_ght` stand-in: instead of a Hough R-table it produces a
Canny-control-point atlas, but it plays the same role -- a one-off model built
from the reference images and reused for every target image.
"""
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .appearance import crop_disc, nlm, grad_dirs, decast, dev_tone


def find_ref_coin(gray):
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, 1.2, 300,
        param1=150, param2=30, minRadius=40, maxRadius=300,
    )
    return tuple(map(int, np.round(circles[0][0])))


def apply_ref_blur(disc, name):
    if name not in C.REF_BLUR:
        return disc
    sigma, (bx, by, br) = C.REF_BLUR[name]
    mask = (((C.GRID_X - bx) ** 2 + (C.GRID_Y - by) ** 2) <= br ** 2) & C.DMASK
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 6)[..., None]
    blurred = cv2.GaussianBlur(disc, (0, 0), float(sigma))
    return (
        disc.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)


def build_reference_model(reference_dir):
    """Builds {atlas, group_ab, proto, proto_sep} from the 8 reference coin images."""
    reference_dir = Path(reference_dir)
    atlas = {}
    ref_ab = {}
    protos_raw = {"copper": [], "gold": []}

    for name in C.CLASSES:
        ref_bgr = cv2.imread(str(reference_dir / f"{name}.jpg"))
        rcx, rcy, rr = find_ref_coin(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))

        disc = apply_ref_blur(nlm(crop_disc(ref_bgr, rcx, rcy, rr), C.REF_NLM[name]), name)
        g = cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY)
        lo, hi = C.REF_EDGE[name]
        atlas[name] = {"cp": (cv2.Canny(g, lo, hi) > 0) & C.DMASK, "ang": grad_dirs(g)}

        # gentler chain for the COLOUR side of the references (tones, not edges)
        cd = cv2.GaussianBlur(nlm(crop_disc(ref_bgr, rcx, rcy, rr), 1), (0, 0), 0.3)
        lab = cv2.cvtColor(cd, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_ab[name] = np.array(
            [np.median(lab[:, :, 1][C.DMASK]), np.median(lab[:, :, 2][C.DMASK])]
        )

        if C.FAMILY[name] != "bimetal":
            dev, _ = dev_tone(decast(ref_bgr), rcx, rcy, rr)
            protos_raw[C.FAMILY[name]].append(dev)

    group_ab = {
        g: np.mean([ref_ab[n] for n in C.CLASSES if C.FAMILY[n] == g], axis=0)
        for g in ("copper", "gold", "bimetal")
    }
    proto = {g: np.mean(v, axis=0) for g, v in protos_raw.items()}
    proto_sep = float(np.hypot(*(proto["copper"] - proto["gold"])))

    return {"atlas": atlas, "group_ab": group_ab, "proto": proto, "proto_sep": proto_sep}
