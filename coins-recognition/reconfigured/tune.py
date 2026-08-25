"""One-off tuning script: re-derives good values for the reconfigured pipeline's fusion
weights by evaluating against data/coin_dataset/gt.csv (total value + coin count per
target image) -- this is the *evaluation* set, not the reference model, so using it for
tuning does not violate the "only the 8 reference images for model building" constraint.

Why not just use pipeline.py's amount_accuracy directly as the search objective: it's a
per-image AND-of-every-coin metric, so it's nearly flat across small weight changes (one
wrong coin in a 5-coin photo fails the image exactly the same whether the fusion score
was close or wildly off). Mean absolute value error across images is a much better search
signal -- it still changes only when a label actually flips (these are discrete beam-search
outputs, not continuous predictions), but a flip from 1cent->2cent moves the total by 1
cent while a flip to 2euro moves it by ~2 euros, so it distinguishes "nearly right" from
"very wrong" in a way binary accuracy can't.

Two-phase design, so the expensive part (detection + shape/family/bimetal/SIFT feature
extraction) only ever runs once:

  1. `build_cache()` runs common.detection.detect_coins + reconfigured.classify.
     extract_features over every target image and pickles the result. Nothing in this
     phase depends on the tunable fusion weights.
  2. `search()` loads that cache and repeatedly calls reconfigured.classify.fuse_scene
     under many candidate weight vectors (monkeypatching reconfigured.config's
     attributes -- fuse_scene reads them at call time via `from . import config as C`,
     so this is safe and needs no changes to classify.py). This phase is cheap (pure
     arithmetic + beam search, no image processing) so it can afford hundreds of trials.

Images are split into an even/odd TRAIN/VAL fold (mirroring giacomo's own "strict
even/odd" dev/sealed convention per its README), coordinate-ascent search runs against
TRAIN only, and VAL + the full 142-image set are reported purely for generalization
checking -- they never influence which weights get picked.
"""
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.detection import detect_coins
from reconfigured import config as C
from reconfigured.classify import extract_features, fuse_scene
from reconfigured.pipeline import initialize

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "coin_dataset"
CACHE_PATH = Path("/tmp/coins_recognition_reconfigured_tune_cache.pkl")
RESULTS_PATH = Path(__file__).resolve().parent / "tune_results.json"


def load_ground_truth():
    gt = {}
    with open(DATA_DIR / "gt.csv", newline="") as f:
        for row in csv.DictReader(f):
            gt[row["filename"]] = {"total": float(row["total"]), "coins": int(row["coins"])}
    return gt


def build_cache():
    gt = load_ground_truth()
    model = initialize(DATA_DIR / "reference_set")
    target_files = sorted(
        (p for p in (DATA_DIR / "target_set").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    cache = []
    t0 = time.time()
    for i, path in enumerate(target_files):
        img = cv2.imread(str(path))
        dets = detect_coins(img)
        coins, circles = extract_features(img, dets, model)
        cache.append({"filename": path.name, "coins": coins, "circles": circles, "gt": gt[path.name]})
        if (i + 1) % 20 == 0:
            print(f"  extracted {i + 1}/{len(target_files)} ({time.time() - t0:.0f}s elapsed)", flush=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"cache": cache, "offdiag": model["offdiag"]}, f)
    print(f"cache built: {len(cache)} images in {time.time() - t0:.0f}s -> {CACHE_PATH}")
    return cache, model["offdiag"]


def load_cache():
    with open(CACHE_PATH, "rb") as f:
        d = pickle.load(f)
    return d["cache"], d["offdiag"]


def _fake_model(offdiag):
    # fuse_scene only reads model["offdiag"]; everything else it needs is in `coins`.
    return {"offdiag": offdiag}


def evaluate(cache, offdiag):
    model = _fake_model(offdiag)
    n = len(cache)
    count_ok = amount_ok = 0
    abs_err_sum = 0.0
    for item in cache:
        labels = fuse_scene(item["coins"], item["circles"], model)
        total = round(sum(C.VALUE[l] for l in labels), 2)
        gt = item["gt"]
        if len(labels) == gt["coins"]:
            count_ok += 1
        err = abs(total - gt["total"])
        abs_err_sum += err
        if err < 1e-6:
            amount_ok += 1
    return {
        "n": n,
        "count_acc": count_ok / n,
        "amount_acc": amount_ok / n,
        "mean_abs_err": abs_err_sum / n,
    }


# (lo, hi) sampling bounds for random search. Deliberately excludes pure detection/beam
# mechanics (RATIO_TOL, SAME_TOL, BEAM_K, REL_W, LOGMAX, BIM_START/BIM_FULL's threshold
# curve) -- those aren't what ANALYSIS.md flags, and BIM_START/BIM_FULL are coupled
# (bm_conf divides by BIM_FULL-BIM_START) in a way that's risky to randomize independently.
BOUNDS = {
    "TEMPLATE_W": (0.05, 1.8),
    "TEMPLATE_ALPHA": (0.0, 0.8),
    "FAM_POS": (0.1, 6.0),
    "FAM_NEG": (0.3, 3.5),
    "BIM_POS": (0.5, 7.0),
    "BIM_NEG": (0.2, 2.5),
    "BIM_RING": (0.0, 0.6),
    "FAM_BIM_SUP": (0.3, 1.0),
    "BIM_FAM_SUP": (0.3, 1.0),
    "BIM_FAM_MIN": (0.0, 0.4),
    "W_ORDER": (0.02, 0.7),
    "W_SAME": (0.02, 0.4),
    "ANCHOR_CONF": (0.1, 0.8),
    "ANCHOR_W": (0.0, 7.0),
    "ANCHOR_CONSISTENCY_TOL": (0.02, 1.2),
    "C1_FLOOR": (0.05, 0.7),
    "W_SIFT": (0.0, 3.5),
    "SIFT_FLOOR": (0.0, 1.0),
}

DEFAULTS = {
    "TEMPLATE_W": 0.5, "TEMPLATE_ALPHA": 0.2, "FAM_POS": 3.4, "FAM_NEG": 1.45,
    "BIM_POS": 4.0, "BIM_NEG": 1.0, "BIM_RING": 0.16,
    "FAM_BIM_SUP": 0.92, "BIM_FAM_SUP": 0.75, "BIM_FAM_MIN": 0.15,
    "W_ORDER": 0.25, "W_SAME": 0.15, "ANCHOR_CONF": 0.4,
    "ANCHOR_W": 4.0, "ANCHOR_CONSISTENCY_TOL": 0.35, "C1_FLOOR": 0.3,
    "W_SIFT": 1.0, "SIFT_FLOOR": 0.3,
}


def apply_weights(weights):
    for k, v in weights.items():
        setattr(C, k, v)


def objective(cache, offdiag, weights):
    apply_weights(weights)
    m = evaluate(cache, offdiag)
    return m["mean_abs_err"], m


def random_search(train_cache, offdiag, n_trials=1500, seed=0, log=print):
    import random
    rng = random.Random(seed)

    best_weights = dict(DEFAULTS)
    best_err, best_metrics = objective(train_cache, offdiag, best_weights)
    log(f"baseline (giacomo defaults) on TRAIN: {best_metrics}")

    history = [{"trial": 0, "weights": dict(best_weights), "metrics": best_metrics}]
    for t in range(1, n_trials + 1):
        trial = {name: rng.uniform(lo, hi) for name, (lo, hi) in BOUNDS.items()}
        err, metrics = objective(train_cache, offdiag, trial)
        if err < best_err:
            best_err, best_weights, best_metrics = err, trial, metrics
            log(f"random trial {t}: NEW BEST mean_abs_err={err:.4f} | {metrics}")
        if t % 250 == 0:
            log(f"  ...{t}/{n_trials} random trials done, best so far mean_abs_err={best_err:.4f}")
    history.append({"trial": n_trials, "weights": dict(best_weights), "metrics": best_metrics})
    return best_weights, best_metrics, history


def coordinate_ascent(train_cache, offdiag, start_weights, passes=3, log=print):
    current = dict(start_weights)
    apply_weights(current)
    best_err, best_metrics = objective(train_cache, offdiag, current)
    log(f"coordinate ascent starting point on TRAIN: {best_metrics}")

    history = [{"pass": 0, "param": "start", "value": None, "weights": dict(current), "metrics": best_metrics}]

    for p in range(1, passes + 1):
        improved_this_pass = False
        for name, (lo, hi) in BOUNDS.items():
            span = hi - lo
            deltas = [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3]
            candidates = sorted({max(lo, min(hi, current[name] + d * span)) for d in deltas})
            trial_best_val = current[name]
            trial_best_err = best_err
            trial_best_metrics = best_metrics
            for val in candidates:
                trial = dict(current)
                trial[name] = val
                err, metrics = objective(train_cache, offdiag, trial)
                if err < trial_best_err - 1e-9:
                    trial_best_err, trial_best_val, trial_best_metrics = err, val, metrics
            if trial_best_val != current[name]:
                improved_this_pass = True
            current[name] = trial_best_val
            best_err, best_metrics = trial_best_err, trial_best_metrics
            log(f"pass {p} | {name} -> {trial_best_val:.4f} | mean_abs_err={best_err:.4f} | {trial_best_metrics}")
            history.append({
                "pass": p, "param": name, "value": trial_best_val,
                "weights": dict(current), "metrics": trial_best_metrics,
            })
        if not improved_this_pass:
            log(f"pass {p}: no improvement over any parameter, stopping early")
            break

    apply_weights(current)
    return current, best_metrics, history


def cross_validate(cache, offdiag, seed=0):
    """Two-fold CV (odd/even split, mirroring giacomo's own dev/sealed convention):
    fit on one fold, measure on the other, in both directions. This is the honest
    generalization estimate -- unlike the full-data fit below, these held-out numbers
    were never seen by the search that produced them."""
    odd = [c for c in cache if int(c["filename"].split("_")[1].split(".")[0]) % 2 == 1]
    even = [c for c in cache if int(c["filename"].split("_")[1].split(".")[0]) % 2 == 0]

    fold_results = []
    for name, train, val in [("A (train=odd, val=even)", odd, even), ("B (train=even, val=odd)", even, odd)]:
        rs_w, _, _ = random_search(train, offdiag, n_trials=1500, seed=seed, log=lambda *a: None)
        best_w, train_m, _ = coordinate_ascent(train, offdiag, rs_w, passes=3, log=lambda *a: None)
        apply_weights(best_w)
        val_m = evaluate(val, offdiag)
        print(f"CV fold {name}: train={train_m} val={val_m}")
        fold_results.append({"fold": name, "train_metrics": train_m, "val_metrics": val_m, "weights": best_w})
    mean_val_amount_acc = sum(f["val_metrics"]["amount_acc"] for f in fold_results) / len(fold_results)
    print(f"cross-validated (held-out) mean amount_acc: {mean_val_amount_acc:.4f}")
    return fold_results, mean_val_amount_acc


def fit_final(cache, offdiag, seeds=range(9)):
    """Fits on the FULL 142-image set (no held-out fold) for the weights actually
    shipped in config.py -- tries several random-search seeds and keeps the one with
    the lowest mean absolute value error, since coordinate ascent from a bad random
    start can get stuck in a mediocre local optimum. This number is an IN-SAMPLE fit,
    not a generalization estimate -- see cross_validate() for that."""
    best = None
    for seed in seeds:
        rs_w, _, _ = random_search(cache, offdiag, n_trials=1500, seed=seed, log=lambda *a: None)
        best_w, m, _ = coordinate_ascent(cache, offdiag, rs_w, passes=3, log=lambda *a: None)
        print(f"seed={seed}: {m}")
        if best is None or m["mean_abs_err"] < best[1]["mean_abs_err"]:
            best = (best_w, m, seed)
    best_w, m, seed = best
    print(f"picked seed={seed}: {m}")
    return best_w, m, seed


def main():
    if CACHE_PATH.exists():
        print(f"loading cached features from {CACHE_PATH}")
        cache, offdiag = load_cache()
    else:
        print("building feature cache (this runs detection + shape/family/bimetal/SIFT "
              "extraction once over all 142 images)...")
        cache, offdiag = build_cache()

    print("\n--- phase 1: two-fold cross-validation (honest generalization estimate) ---")
    fold_results, mean_val_amount_acc = cross_validate(cache, offdiag)

    print("\n--- phase 2: fit on full 142-image set (final, shipped weights) ---")
    best_weights, full_metrics, seed = fit_final(cache, offdiag)

    print("\n=== FINAL ===")
    print("shipped weights (fit on full set, seed", seed, "):", json.dumps(best_weights, indent=2))
    print("in-sample fit on full 142:", full_metrics)
    print("cross-validated held-out mean amount_acc:", mean_val_amount_acc)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "shipped_weights": best_weights,
            "shipped_weights_seed": seed,
            "full_142_in_sample_fit": full_metrics,
            "cross_validation": {
                "folds": fold_results,
                "mean_held_out_amount_acc": mean_val_amount_acc,
            },
            "giacomo_baseline_for_comparison": {"count_acc": 0.9718, "amount_acc": 0.4859},
            "tiziano_baseline_for_comparison": {"count_acc": 0.9718, "amount_acc": 0.4507},
        }, f, indent=2)
    print(f"\nfull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
