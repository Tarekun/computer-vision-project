"""Ad hoc ablation: isolate which fix helps/hurts vs giacomo's original numbers
(48.59% full-set amount accuracy). Not part of the shipped pipeline.

Builds reference models under a few variants and evaluates each with GIACOMO'S
ORIGINAL weights (SIFT disabled) to isolate the shape/colour fixes from both the
SIFT cue and the weight re-tuning.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.detection import detect_coins
from reconfigured import config as C
from reconfigured.classify import extract_features, fuse_scene
from reconfigured.tune import load_ground_truth, DATA_DIR

GIACOMO_WEIGHTS = {
    "TEMPLATE_W": 0.5, "TEMPLATE_ALPHA": 0.2, "FAM_POS": 3.4, "FAM_NEG": 1.45,
    "BIM_POS": 4.0, "BIM_NEG": 1.0, "BIM_RING": 0.16,
    "FAM_BIM_SUP": 0.92, "BIM_FAM_SUP": 0.75, "BIM_FAM_MIN": 0.15,
    "W_ORDER": 0.25, "W_SAME": 0.15, "ANCHOR_CONF": 0.4,
    "ANCHOR_W": 4.0, "ANCHOR_CONSISTENCY_TOL": 999.0, "C1_FLOOR": 0.3,
    "W_SIFT": 0.0, "SIFT_FLOOR": 0.3,
}


def apply_weights(weights):
    for k, v in weights.items():
        setattr(C, k, v)


def build_model_variant(reference_dir, decast_atlas):
    """Like reconfigured.reference.build_reference_model, but with a flag to toggle
    the decast-atlas fix on/off so it can be isolated."""
    import cv2
    from reconfigured.appearance import crop_disc, nlm, grad_dirs, decast, dev_tone, shape_scores
    from reconfigured.reference import find_ref_coin, apply_ref_blur
    from reconfigured.sift_cue import calibrate_sift_templates

    reference_dir = Path(reference_dir)
    atlas, ref_ab, ref_discs = {}, {}, {}
    protos_raw = {"copper": [], "gold": []}

    for name in C.CLASSES:
        ref_bgr = cv2.imread(str(reference_dir / f"{name}.jpg"))
        rcx, rcy, rr = find_ref_coin(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
        atlas_src = decast(ref_bgr) if decast_atlas else ref_bgr
        dcx_ref = decast(ref_bgr)  # colour prototype side always uses decast, as in giacomo

        raw_crop_atlas = crop_disc(atlas_src, rcx, rcy, rr)
        disc = apply_ref_blur(nlm(raw_crop_atlas, C.REF_NLM[name]), name)
        g = cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY)
        lo, hi = C.REF_EDGE[name]
        atlas[name] = {"cp": (cv2.Canny(g, lo, hi) > 0) & C.DMASK, "ang": grad_dirs(g)}
        ref_discs[name] = nlm(raw_crop_atlas, C.TGT_NLM)

        raw_crop_colour = crop_disc(dcx_ref, rcx, rcy, rr)
        cd = cv2.GaussianBlur(nlm(raw_crop_colour, 1), (0, 0), 0.3)
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


def run_variant(name, decast_atlas):
    apply_weights(GIACOMO_WEIGHTS)  # reset before building (offdiag derivation uses shape_scores, weight-independent anyway)
    t0 = time.time()
    model = build_model_variant(DATA_DIR / "reference_set", decast_atlas=decast_atlas)
    gt = load_ground_truth()
    target_files = sorted(
        (p for p in (DATA_DIR / "target_set").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    apply_weights(GIACOMO_WEIGHTS)
    n = count_ok = amount_ok = 0
    abs_err_sum = 0.0
    for path in target_files:
        img = cv2.imread(str(path))
        dets = detect_coins(img)
        coins, circles = extract_features(img, dets, model)
        labels = fuse_scene(coins, circles, model)
        total = round(sum(C.VALUE[l] for l in labels), 2)
        g = gt[path.name]
        n += 1
        if len(labels) == g["coins"]:
            count_ok += 1
        err = abs(total - g["total"])
        abs_err_sum += err
        if err < 1e-6:
            amount_ok += 1
    print(f"[{name}] n={n} count_acc={count_ok/n:.4f} amount_acc={amount_ok/n:.4f} "
          f"mean_abs_err={abs_err_sum/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    run_variant("fixes: aspect-crop + offdiag-derived + decast-ATLAS, giacomo weights, SIFT off", decast_atlas=True)
    run_variant("fixes: aspect-crop + offdiag-derived + NO decast-atlas (giacomo-original atlas input), giacomo weights, SIFT off", decast_atlas=False)
