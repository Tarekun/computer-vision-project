"""EXPERIMENTS.md item 4: jointly search per-class REF_EDGE (Canny threshold) pairs
together with the decast-atlas fix, instead of assuming giacomo's original REF_EDGE
values (calibrated against the RAW reference photo's contrast) are still right once
the atlas-building crop is decasted. reconfigured/README.md flagged this as follow-up
work, not attempted there.

Same isolation methodology as reconfigured/ablate.py: GIACOMO_WEIGHTS (SIFT disabled)
throughout, so only the reference-model construction varies between trials -- this
keeps the comparison to reconfigured/ablate.py's own 49.30% (decast off, original
REF_EDGE) / 45.77% (decast on, original REF_EDGE) numbers apples-to-apples.

Each trial rebuilds the reference model (atlas + offdiag; group_ab/proto/sift_cache
are unaffected by REF_EDGE/decast_atlas and reused from a single shared `initialize()`
call) and re-scores all 142 target images against it -- there's no way to avoid
re-running the 360-rotation shape-score search per trial since the atlas itself
changes, which is exactly why reconfigured/README.md called this expensive.
"""
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.detection import detect_coins
from experimental import config as C
from experimental.appearance import crop_disc, nlm, grad_dirs, decast, dev_tone, shape_scores
from experimental.classify import extract_features, fuse_scene
from experimental.reference import find_ref_coin, apply_ref_blur
from experimental.sift_cue import calibrate_sift_templates
from experimental.tune import DATA_DIR, load_ground_truth

GIACOMO_WEIGHTS = {
    "TEMPLATE_W": 0.5, "TEMPLATE_ALPHA": 0.2, "FAM_POS": 3.4, "FAM_NEG": 1.45,
    "BIM_POS": 4.0, "BIM_NEG": 1.0, "BIM_RING": 0.16,
    "FAM_BIM_SUP": 0.92, "BIM_FAM_SUP": 0.75, "BIM_FAM_MIN": 0.15,
    "W_ORDER": 0.25, "W_SAME": 0.15, "ANCHOR_CONF": 0.4,
    "ANCHOR_W": 4.0, "ANCHOR_CONSISTENCY_TOL": 999.0, "C1_FLOOR": 0.3,
    "W_SIFT": 0.0, "SIFT_FLOOR": 0.3, "TWO_STAGE_SIZE_FIRST": False,
}

ORIGINAL_REF_EDGE = dict(C.REF_EDGE)  # giacomo's hand-typed thresholds, the search's anchor
RESULTS_PATH = Path(__file__).resolve().parent / "ref_edge_search_results.json"


def apply_weights(weights):
    for k, v in weights.items():
        setattr(C, k, v)


def build_model_variant(reference_dir, decast_atlas, ref_edge):
    """Like reconfigured/ablate.py's build_model_variant, parameterized by a full
    REF_EDGE dict instead of only a decast_atlas flag. ref_discs (used for offdiag's
    "other class" side and the SIFT cache) always come from the RAW reference photo,
    decoupled from decast_atlas/ref_edge -- matching experimental/reference.py's own
    shipped design, not ablate.py's ad hoc script."""
    reference_dir = Path(reference_dir)
    atlas, ref_ab, ref_discs = {}, {}, {}
    protos_raw = {"copper": [], "gold": []}

    for name in C.CLASSES:
        ref_bgr = cv2.imread(str(reference_dir / f"{name}.jpg"))
        rcx, rcy, rr = find_ref_coin(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
        atlas_src = decast(ref_bgr) if decast_atlas else ref_bgr
        dcx_ref = decast(ref_bgr)

        raw_crop_atlas = crop_disc(atlas_src, rcx, rcy, rr)
        disc = apply_ref_blur(nlm(raw_crop_atlas, C.REF_NLM[name]), name)
        g = cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY)
        lo, hi = ref_edge[name]
        atlas[name] = {"cp": (cv2.Canny(g, lo, hi) > 0) & C.DMASK, "ang": grad_dirs(g)}

        raw_crop = crop_disc(ref_bgr, rcx, rcy, rr)  # RAW, shared across every trial
        ref_discs[name] = nlm(raw_crop, C.TGT_NLM)

        raw_crop_colour = crop_disc(dcx_ref, rcx, rcy, rr)
        cd = cv2.GaussianBlur(nlm(raw_crop_colour, 1), (0, 0), 0.3)
        lab = cv2.cvtColor(cd, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_ab[name] = np.array([np.median(lab[:, :, 1][C.DMASK]), np.median(lab[:, :, 2][C.DMASK])])
        if C.FAMILY[name] != "bimetal":
            dev, _ = dev_tone(dcx_ref, rcx, rcy, rr)
            protos_raw[C.FAMILY[name]].append(dev)

    group_ab = {g: np.mean([ref_ab[n] for n in C.CLASSES if C.FAMILY[n] == g], axis=0)
                for g in ("copper", "gold", "bimetal")}
    proto = {g: np.mean(v, axis=0) for g, v in protos_raw.items()}
    proto_sep = float(np.hypot(*(proto["copper"] - proto["gold"])))

    cross_scores = {other: shape_scores(ref_discs[other], atlas) for other in C.CLASSES}
    offdiag = {c: float(np.mean([cross_scores[other][c] for other in C.CLASSES if other != c]))
               for c in C.CLASSES}
    sift_cache = calibrate_sift_templates(ref_discs)
    return {"atlas": atlas, "group_ab": group_ab, "proto": proto, "proto_sep": proto_sep,
            "offdiag": offdiag, "sift_cache": sift_cache}


def run_variant(model, target_files, gt):
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
        count_ok += int(len(labels) == g["coins"])
        err = abs(total - g["total"])
        abs_err_sum += err
        amount_ok += int(err < 1e-6)
    return {"n": n, "count_acc": count_ok / n, "amount_acc": amount_ok / n, "mean_abs_err": abs_err_sum / n}


def random_ref_edge(rng, spread=0.4):
    """Perturbs each class's (lo, hi) Canny thresholds by up to +/-spread (fraction
    of the original value), keeping lo < hi and both positive."""
    out = {}
    for name, (lo, hi) in ORIGINAL_REF_EDGE.items():
        nlo = max(1, lo * (1 + rng.uniform(-spread, spread)))
        nhi = max(nlo + 1, hi * (1 + rng.uniform(-spread, spread)))
        out[name] = (nlo, nhi)
    return out


def main(n_random_trials=10, seed=0):
    gt = load_ground_truth()
    target_files = sorted(
        (p for p in (DATA_DIR / "target_set").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    rng = random.Random(seed)
    results = []

    # controls, reproducing reconfigured/ablate.py's own numbers as a sanity check
    for label, decast_atlas, ref_edge in [
        ("control: decast=False, original REF_EDGE (reconfigured/ablate.py's 49.30%)", False, ORIGINAL_REF_EDGE),
        ("control: decast=True,  original REF_EDGE (reconfigured/ablate.py's 45.77%)", True, ORIGINAL_REF_EDGE),
    ]:
        t0 = time.time()
        model = build_model_variant(DATA_DIR / "reference_set", decast_atlas, ref_edge)
        m = run_variant(model, target_files, gt)
        print(f"[{label}] {m} ({time.time() - t0:.0f}s)", flush=True)
        results.append({"label": label, "decast_atlas": decast_atlas, "ref_edge": ref_edge, "metrics": m})

    # joint random search: decast=True with retuned per-class thresholds
    for t in range(1, n_random_trials + 1):
        ref_edge = random_ref_edge(rng)
        t0 = time.time()
        model = build_model_variant(DATA_DIR / "reference_set", True, ref_edge)
        m = run_variant(model, target_files, gt)
        print(f"[trial {t}/{n_random_trials}, decast=True, random REF_EDGE] {m} ({time.time() - t0:.0f}s)", flush=True)
        results.append({"label": f"random trial {t}", "decast_atlas": True, "ref_edge": ref_edge, "metrics": m})

    best = max(results, key=lambda r: r["metrics"]["amount_acc"])
    print("\n=== BEST ===")
    print(json.dumps(best, indent=2))
    with open(RESULTS_PATH, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"full results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
