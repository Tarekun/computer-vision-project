"""Standalone comparison: is ORB a viable, cheaper substitute for the SIFT cue, now
that both sides (reference cache domain, per-class score normalisation) are fixed
(EXPERIMENTS.md item 5)?

Reuses `classify.fuse_scene` and `tune.evaluate` completely unmodified: ORB match
scores are written into the same `coin["sift_scores"]`/`coin["sift_conf"]` slots SIFT
normally fills (both are already descriptor-agnostic per-class rate + top1-vs-top2
confidence), so the existing fusion math (z-scoring, W_SIFT, SIFT_FLOOR) is exercised
identically -- the descriptor choice is the only variable that changes between this
and the SIFT-based cache.
"""
import pickle
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.detection import detect_coins
from experimental import config as C
from experimental.appearance import (
    crop_disc, nlm, decast, shape_scores, two_tone_family, first_pass, bimetal_step, ring_sign,
)
from experimental.orb_cue import calibrate_orb_templates, orb_extract_with_nms, orb_match_scores
from experimental.pipeline import initialize
from experimental.reference import find_ref_coin
from experimental.sift_cue import sift_confidence
from experimental.tune import DATA_DIR, load_ground_truth, evaluate as tune_evaluate

CACHE_PATH = Path("/tmp/coins_recognition_experimental_orb_cache.pkl")


def extract_features_orb(img, dets, model, orb_cache):
    if not dets:
        return [], []
    atlas, group_ab = model["atlas"], model["group_ab"]
    proto, proto_sep = model["proto"], model["proto_sep"]
    dcx = decast(img)
    circles = [(int(round(d["cx"])), int(round(d["cy"])), int(round(d["r"]))) for d in dets]
    coins = []
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
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
        orb_des = orb_extract_with_nms(crop_gray)
        orb_scores, _ = orb_match_scores(orb_des, bf_matcher, orb_cache)
        orb_conf = sift_confidence(orb_scores)

        coins.append({
            "r": r, "bg_z": float(det.get("bg_z", 9.9)), "shape": shp,
            "pred0": first_pass(shp, crop, group_ab), "fam": fam, "fam_conf": fam_conf,
            "step": bimetal_step(disc), "ring": ring_sign(disc),
            "sift_scores": orb_scores, "sift_conf": orb_conf,  # ORB scores in SIFT's slot
        })
    return coins, circles


def build_orb_cache():
    gt = load_ground_truth()
    model = initialize(DATA_DIR / "reference_set")

    # ORB's own reference cache, from the same canonical-crop domain reference.py
    # already uses for SIFT (build_reference_model doesn't expose its ref_discs, so
    # this repeats that same crop_disc/nlm construction here, only for ORB).
    ref_discs = {}
    for name in C.CLASSES:
        ref_bgr = cv2.imread(str(DATA_DIR / "reference_set" / f"{name}.jpg"))
        rcx, rcy, rr = find_ref_coin(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
        raw_crop = crop_disc(ref_bgr, rcx, rcy, rr)
        ref_discs[name] = nlm(raw_crop, C.TGT_NLM)
    orb_cache = calibrate_orb_templates(ref_discs)

    target_files = sorted(
        (p for p in (DATA_DIR / "target_set").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    cache = []
    t0 = time.time()
    for i, path in enumerate(target_files):
        img = cv2.imread(str(path))
        dets = detect_coins(img)
        coins, circles = extract_features_orb(img, dets, model, orb_cache)
        cache.append({"filename": path.name, "coins": coins, "circles": circles, "gt": gt[path.name]})
        if (i + 1) % 20 == 0:
            print(f"  extracted {i + 1}/{len(target_files)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"cache": cache, "offdiag": model["offdiag"]}, f)
    print(f"ORB cache built: {len(cache)} images in {time.time() - t0:.0f}s -> {CACHE_PATH}")
    return cache, model["offdiag"]


def main():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            d = pickle.load(f)
        cache, offdiag = d["cache"], d["offdiag"]
    else:
        cache, offdiag = build_orb_cache()
    metrics = tune_evaluate(cache, offdiag)
    print("ORB-substituted-for-SIFT, current shipped fusion weights:", metrics)


if __name__ == "__main__":
    main()
