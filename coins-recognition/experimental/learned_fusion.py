"""EXPERIMENTS.md item 2: a learned fusion classifier over the cue vector, instead of
(some of) fuse_scene's hand-set/searched additive weights -- ANALYSIS.md S6's second
"bigger bet".

`data/coin_dataset/gt.csv` has no per-coin labels, confirmed nowhere in this repo (see
EXPERIMENTS.md item 2) -- only per-image {filename, total, coins}. So "trained on a
labelled dev split" (ANALYSIS.md's original phrasing, referring to giacomo's own
one-off 99-image/227-coin manual count that was never saved to a file) isn't literally
available. The honest substitute here is SELF-TRAINING: pseudo-label individual coins
using images where a baseline fusion's whole-image prediction already matches gt.csv
exactly (count AND total) AND that (count, total) pair has only ONE feasible multiset
of the 8 denominations -- so the model's predicted label SET for that image must equal
the true one.

This does NOT fully rule out a within-image swap between two coins that both belong to
that one correct multiset (the model could exchange which specific coin gets which of
two distinct labels while the image-level total still checks out) -- a real, documented
limitation of this weak-supervision signal that this script does not eliminate. The
self-trained classifier is judged purely by its measured held-out CV amount_accuracy
against the hand-tuned/searched weights on the SAME folds: if swap noise were too large
to learn anything useful, the held-out number would show it.

The learned model keeps the exact same feature set fuse_scene already computes
(z-scored shape template, family/bimetal indicators, z-scored SIFT, the three
log-distance size-consistency penalties) and the same beam-search pairwise mechanics
(`common.classify.beam_assign`, untouched) -- only the per-class UNARY score becomes a
single shared linear combination (`feats[c] @ w`, fit by multinomial softmax regression
via cross-entropy) instead of config.py's several independently hand-set/searched
weights (TEMPLATE_W, FAM_POS/NEG, BIM_POS/NEG, W_SIFT, W_ORDER, W_SAME, ANCHOR_W).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experimental import config as C
from common.classify import zscore, bm_conf, abs_anchor, beam_assign
from experimental.tune import (
    DATA_DIR, load_cache, random_search, coordinate_ascent, apply_weights, evaluate as tune_evaluate,
)
from experimental.classify import fuse_scene

FEATURE_NAMES = ["template", "fam", "bim", "sift", "order_pen", "same_pen", "anchor_pen"]
N_FEATURES = len(FEATURE_NAMES)


def feasible_multisets(count, total, tol=0.006):
    """Every size-`count` multiset of C.CLASSES whose value sums to `total` (euros),
    within rounding tolerance -- used to check whether a correct image-level
    prediction could only have come from one specific set of denominations."""
    out = set()
    for combo in itertools.combinations_with_replacement(C.CLASSES, count):
        if abs(round(sum(C.VALUE[c] for c in combo), 2) - total) < tol:
            out.add(tuple(sorted(combo)))
    return out


def scene_context(full):
    sscale = float(np.median([c["r"] / C.DIAMETER_MM[c["pred0"]] for c in full]))
    rclass = {}
    for c in full:
        rclass.setdefault(c["pred0"], []).append(c["r"])
    rclass = {k: float(np.median(v)) for k, v in rclass.items()}
    s_abs = abs_anchor(full)
    if s_abs is not None and abs(np.log(s_abs / sscale)) > C.ANCHOR_CONSISTENCY_TOL:
        s_abs = None
    return sscale, rclass, s_abs


def coin_features(coin, offdiag, sscale, rclass, s_abs):
    """Per-class raw feature vector for one coin: the same intermediate quantities
    fuse_scene computes, kept un-weighted. Returns an (8, N_FEATURES) array, rows in
    C.CLASSES order."""
    template_z = zscore({c: coin["shape"][c] - C.TEMPLATE_ALPHA * offdiag[c] for c in C.CLASSES})
    fam_raw = {c: 0.0 for c in C.CLASSES}
    if coin["fam"]:
        for c in C.CLASSES:
            fam_raw[c] = (1.0 if C.FAMILY[c] == coin["fam"] else -1.0) * coin["fam_conf"]
    bm = bm_conf(coin["step"])
    bim_raw = {c: (1.0 if C.FAMILY[c] == "bimetal" else -1.0) * bm for c in C.CLASSES}
    sift_z = zscore(coin["sift_scores"]) if any(coin["sift_scores"].values()) else {c: 0.0 for c in C.CLASSES}
    d_obs = coin["r"] / sscale

    feats = np.zeros((len(C.CLASSES), N_FEATURES))
    for i, c in enumerate(C.CLASSES):
        order_pen = -min(abs(np.log(d_obs / C.DIAMETER_MM[c])) / C.LOGMAX, 1.0)
        same_pen = -(min(abs(np.log(coin["r"] / rclass[c])) / C.LOGMAX, 1.0) if c in rclass else 0.0)
        anchor_pen = 0.0
        if s_abs is not None:
            anchor_pen = -min(abs(np.log((coin["r"] / s_abs) / C.DIAMETER_MM[c])) / C.LOGMAX, 1.0)
        feats[i] = [template_z[c], fam_raw[c], bim_raw[c], sift_z[c], order_pen, same_pen, anchor_pen]
    return feats


def collect_pseudo_labeled_examples(cache, offdiag, weights):
    """Runs fuse_scene under `weights` over every image in `cache` (monkeypatches
    experimental.config, same mechanism tune.py uses), keeps only coins from images
    whose predicted (count, total) exactly matches gt AND whose gt (count, total) has
    a UNIQUE feasible multiset (see module docstring for the swap caveat this does not
    eliminate). Returns a list of (feats[8,N_FEATURES], true_class_index)."""
    apply_weights(weights)
    examples = []
    kept_images = 0
    for item in cache:
        coins, circles, gt = item["coins"], item["circles"], item["gt"]
        full = [c for c in coins if c]
        if not full:
            continue
        labels = fuse_scene(coins, circles, {"offdiag": offdiag})
        total = round(sum(C.VALUE[l] for l in labels), 2)
        if len(labels) != gt["coins"] or abs(total - gt["total"]) >= 1e-6:
            continue
        if len(feasible_multisets(gt["coins"], gt["total"])) != 1:
            continue
        sscale, rclass, s_abs = scene_context(full)
        k = 0
        for coin in coins:
            if coin is None:
                continue
            feats = coin_features(coin, offdiag, sscale, rclass, s_abs)
            examples.append((feats, C.CLASSES.index(labels[k])))
            k += 1
        kept_images += 1
    return examples, kept_images


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _loss_and_grad(w, examples, l2):
    loss = 0.0
    grad = np.zeros_like(w)
    for feats, true_idx in examples:
        p = _softmax(feats @ w)
        loss += -np.log(max(p[true_idx], 1e-12))
        onehot = np.zeros(len(p))
        onehot[true_idx] = 1.0
        grad += feats.T @ (p - onehot)
    n = len(examples)
    loss = loss / n + 0.5 * l2 * np.sum(w ** 2)
    grad = grad / n + l2 * w
    return loss, grad


def fit_linear_softmax(examples, l2=1e-3):
    w0 = np.zeros(N_FEATURES)
    res = minimize(_loss_and_grad, w0, args=(examples, l2), jac=True, method="L-BFGS-B")
    return res.x


def fuse_scene_learned(coins, circles, model, w):
    """Same structure as classify.fuse_scene, but the per-class unary score is the
    single learned linear combination `feats[c] @ w` instead of config.py's several
    independent weights. Beam-search pairwise mechanics are untouched."""
    if not coins:
        return []
    offdiag = model["offdiag"]
    full = [c for c in coins if c]
    if not full:
        return ["20cent"] * len(coins)
    sscale, rclass, s_abs = scene_context(full)

    unaries, radii = [], []
    for coin in coins:
        if coin is None:
            continue
        feats = coin_features(coin, offdiag, sscale, rclass, s_abs)
        unaries.append({c: float(feats[i] @ w) for i, c in enumerate(C.CLASSES)})
        radii.append(coin["r"])

    beam_labels = beam_assign(unaries, radii)
    labels, k = [], 0
    for i, coin in enumerate(coins):
        if coin is None:
            d_obs = circles[i][2] / sscale
            labels.append(min(C.CLASSES, key=lambda c: abs(np.log(d_obs / C.DIAMETER_MM[c]))))
        else:
            labels.append(beam_labels[k])
            k += 1
    return labels


def evaluate_learned(cache, offdiag, w):
    n = count_ok = amount_ok = 0
    abs_err_sum = 0.0
    for item in cache:
        labels = fuse_scene_learned(item["coins"], item["circles"], {"offdiag": offdiag}, w)
        total = round(sum(C.VALUE[l] for l in labels), 2)
        gt = item["gt"]
        n += 1
        count_ok += int(len(labels) == gt["coins"])
        err = abs(total - gt["total"])
        abs_err_sum += err
        amount_ok += int(err < 1e-6)
    return {"n": n, "count_acc": count_ok / n, "amount_acc": amount_ok / n, "mean_abs_err": abs_err_sum / n}


def run_cv(cache, offdiag, k=4, seed=0, n_trials=500, passes=2, log=print):
    """k-fold comparison: on each fold's TRAIN split, hand-tune weights (same search
    tune.py uses, just fewer trials for speed) to generate pseudo-labels, fit the
    learned linear softmax on those pseudo-labels, then compare both models' accuracy
    on the SAME held-out VAL split. Neither model ever sees VAL during fitting."""
    def fold_id(item):
        return int(item["filename"].split("_")[1].split(".")[0]) % k

    folds = [[c for c in cache if fold_id(c) == i] for i in range(k)]
    results = []
    for i in range(k):
        val = folds[i]
        train = [c for j in range(k) if j != i for c in folds[j]]

        rs_w, _, _ = random_search(train, offdiag, n_trials=n_trials, seed=seed, log=lambda *a: None)
        hand_w, hand_train_m, _ = coordinate_ascent(train, offdiag, rs_w, passes=passes, log=lambda *a: None)
        apply_weights(hand_w)
        hand_val_m = tune_evaluate(val, offdiag)

        examples, kept_images = collect_pseudo_labeled_examples(train, offdiag, hand_w)
        w = fit_linear_softmax(examples)
        learned_val_m = evaluate_learned(val, offdiag, w)

        log(f"fold {i}/{k}: pseudo-labeled {kept_images}/{len(train)} train images "
            f"({len(examples)} coin examples) | hand-tuned val={hand_val_m} | learned val={learned_val_m}")
        results.append({
            "fold": i, "kept_images": kept_images, "n_examples": len(examples),
            "hand_val_metrics": hand_val_m, "learned_val_metrics": learned_val_m,
            "learned_weights": dict(zip(FEATURE_NAMES, w.tolist())),
        })

    hand_mean = sum(r["hand_val_metrics"]["amount_acc"] for r in results) / len(results)
    learned_mean = sum(r["learned_val_metrics"]["amount_acc"] for r in results) / len(results)
    log(f"\nmean held-out amount_acc: hand-tuned={hand_mean:.4f} learned={learned_mean:.4f}")
    return results, hand_mean, learned_mean


if __name__ == "__main__":
    import json
    cache, offdiag = load_cache()
    results, hand_mean, learned_mean = run_cv(cache, offdiag)
    with open(Path(__file__).resolve().parent / "learned_fusion_results.json", "w") as f:
        json.dump({"folds": results, "hand_mean_amount_acc": hand_mean, "learned_mean_amount_acc": learned_mean}, f, indent=2)
