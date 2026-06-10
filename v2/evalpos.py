# v2/evalpos.py
# Positional evaluation: circle-IoU matching, detection + classification metrics,
# per-stage error attribution (detection -> famiglia -> taglia/shape).
import json, math, os, sys
import numpy as np
from v2.paths import PROJECT, FAMILY, VALUES, is_dev

def circle_iou(a, b):
    (x1, y1, r1), (x2, y2, r2) = a, b
    d = math.hypot(x2 - x1, y2 - y1)
    if d >= r1 + r2: return 0.0
    if d <= abs(r1 - r2):
        inter = math.pi * min(r1, r2) ** 2
    else:
        a1 = r1*r1*math.acos((d*d + r1*r1 - r2*r2) / (2*d*r1))
        a2 = r2*r2*math.acos((d*d + r2*r2 - r1*r1) / (2*d*r2))
        a3 = 0.5*math.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
        inter = a1 + a2 - a3
    union = math.pi*(r1*r1 + r2*r2) - inter
    return inter / union

def match_scene(gt, pred, thr=0.5):
    cands = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = circle_iou((g["cx"],g["cy"],g["r"]), (p["cx"],p["cy"],p["r"]))
            if iou > thr: cands.append((iou, i, j))
    used_g, used_p, pairs = set(), set(), []
    for iou, i, j in sorted(cands, reverse=True):
        if i in used_g or j in used_p: continue
        pairs.append((i, j)); used_g.add(i); used_p.add(j)
    return {"pairs": pairs,
            "miss": [i for i in range(len(gt)) if i not in used_g],
            "fp": [j for j in range(len(pred)) if j not in used_p]}

def scene_metrics(gt, pred, m):
    n_eval = n_correct = 0
    stage, cerr, rerr = [], [], []
    multi = len(gt) > 1
    for i, j in m["pairs"]:
        g, p = gt[i], pred[j]
        cerr.append(math.hypot(g["cx"]-p["cx"], g["cy"]-p["cy"]))
        rerr.append(abs(p["r"]-g["r"]) / g["r"] * 100)
        if g["label"] == "unknown": continue
        n_eval += 1
        if p["pred"] == g["label"]: n_correct += 1
        elif FAMILY[p["pred"]] != FAMILY[g["label"]]: stage.append("famiglia")
        else: stage.append("taglia" if multi else "shape_single")
    stage += ["detection_miss"] * sum(1 for i in m["miss"] if gt[i]["label"] != "unknown")
    amt_gt = sum(VALUES.get(g["label"], 0) for g in gt)
    amt_pr = sum(VALUES[p["pred"]] for p in pred)
    # NOTE: an image with unknown coins can never be exact (n_eval == len(gt) fails) — acceptable, documented behavior.
    return {"n_eval": n_eval, "n_correct": n_correct, "stage": stage,
            "center_err": cerr, "radius_errpc": rerr,
            "n_miss": len(m["miss"]), "n_fp": len(m["fp"]),
            "count_ok": len(gt) == len(pred), "amount_ae": abs(amt_gt - amt_pr),
            "exact": n_eval == len(gt) and n_correct == n_eval and not m["fp"] and not m["miss"]}

def evaluate(pred_path, half="dev"):
    gt = json.load(open(os.path.join(PROJECT, "v2", "gt_pos.json")))
    pred = json.load(open(pred_path))
    tot = {"n_eval":0,"n_correct":0,"n_miss":0,"n_fp":0,"count_ok":0,"exact":0,
           "imgs":0,"amount_ae":[], "center_err":[], "radius_errpc":[], "stage":{}}
    for name, entry in gt.items():
        if half == "dev" and not is_dev(name): continue
        if half == "sealed" and is_dev(name): continue
        m = match_scene(entry["coins"], pred.get(name, []))
        s = scene_metrics(entry["coins"], pred.get(name, []), m)
        tot["imgs"] += 1
        for k in ("n_eval","n_correct","n_miss","n_fp"): tot[k] += s[k]
        tot["count_ok"] += s["count_ok"]; tot["exact"] += s["exact"]
        tot["amount_ae"].append(s["amount_ae"])
        tot["center_err"] += s["center_err"]; tot["radius_errpc"] += s["radius_errpc"]
        for st in s["stage"]: tot["stage"][st] = tot["stage"].get(st, 0) + 1
    r = {"half": half, "images": tot["imgs"],
         "coin_acc": round(tot["n_correct"]/max(1,tot["n_eval"]), 3),
         "n_eval": tot["n_eval"], "miss": tot["n_miss"], "fp": tot["n_fp"],
         "count_acc": round(tot["count_ok"]/max(1,tot["imgs"]), 3),
         "exact_rate": round(tot["exact"]/max(1,tot["imgs"]), 3),
         "amount_mae": round(float(np.mean(tot["amount_ae"])), 3),
         "center_err_med": round(float(np.median(tot["center_err"])), 2) if tot["center_err"] else None,
         "radius_errpc_med": round(float(np.median(tot["radius_errpc"])), 2) if tot["radius_errpc"] else None,
         "error_stages": tot["stage"]}
    return r

if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dev"), indent=1))
