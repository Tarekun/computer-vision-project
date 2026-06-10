# Re-ingegnerizzazione pipeline monete — Piano Fase 0 + Fase 1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Costruire GT posizionale validabile sui pixel + error budget del campione e12, poi il nuovo fronte detection (prefilter misurato, generatore+validatore, rettifica prospettica), come da spec `docs/superpowers/specs/2026-06-10-reingegnerizzazione-pipeline-design.md`.

**Architecture:** Moduli `v2/` (un file per stadio, importabili dal futuro `solution_2.ipynb`), GT etichettato da agenti che GUARDANO le immagini, eval posizionale con matching IoU, error budget per stadio. M2+ (classifier DP) è ESCLUSO da questo piano: la spec lo gates sull'error budget (Task 12 scrive il piano di Fase 2).

**Tech Stack:** Python `"/home/gianmarco/envs/IP&CV/bin/python"`, solo cv2 (4.13) + numpy + matplotlib. **NO pytest** (assente nell'env): test con `unittest` stdlib. **NO skimage** (vincolo utente).

**Working dir di TUTTI i comandi:** `/home/gianmarco/projects/Image Processing  and Computer Vision/Part 1/Project` (citato sotto come `$P`). Le 142 target sono `coin_dataset/target_set/image_N.jpg`, gli 8 ref `coin_dataset/reference_set/<denom>.jpg`.

**Convenzioni:** split GT = numero immagine PARI → metà sviluppo, DISPARI → sigillata (non guardarla mai fino a M4). Token classi: `1cent 2cent 5cent 10cent 20cent 50cent 1euro 2euro` + `unknown` (solo GT). Famiglie: copper={1,2,5cent}, gold={10,20,50cent}, bimetal={1euro,2euro}.

---

### Task 1: Scaffold pacchetto v2/ + infrastruttura test

**Files:**
- Create: `v2/__init__.py` (vuoto)
- Create: `v2/tests/__init__.py` (vuoto)
- Create: `v2/paths.py`
- Test: `v2/tests/test_paths.py`

- [ ] **Step 1: Scrivere il test fallente**

```python
# v2/tests/test_paths.py
import unittest, os
from v2.paths import TARGET_DIR, REF_DIR, target_images, image_num, is_dev

class TestPaths(unittest.TestCase):
    def test_dirs_exist(self):
        self.assertTrue(os.path.isdir(TARGET_DIR))
        self.assertTrue(os.path.isdir(REF_DIR))
    def test_target_list(self):
        imgs = target_images()
        self.assertEqual(len(imgs), 142)
        self.assertTrue(all(i.startswith("image_") for i in imgs))
    def test_image_num_and_split(self):
        self.assertEqual(image_num("image_85.jpg"), 85)
        self.assertFalse(is_dev("image_85.jpg"))   # dispari = sigillata
        self.assertTrue(is_dev("image_4.jpg"))     # pari = sviluppo

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Eseguire il test e verificarne il fallimento**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m unittest v2.tests.test_paths -v`
Expected: FAIL/ERROR (`No module named 'v2.paths'`)

- [ ] **Step 3: Implementazione minima**

```python
# v2/paths.py
# Project-relative paths and the frozen dev/sealed split (even=dev, odd=sealed).
import os, re

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(PROJECT, "coin_dataset", "target_set")
REF_DIR = os.path.join(PROJECT, "coin_dataset", "reference_set")
PY = "/home/gianmarco/envs/IP&CV/bin/python"
CLASSES = ["1cent","2cent","5cent","10cent","20cent","50cent","1euro","2euro"]
VALUES = {"1cent":0.01,"2cent":0.02,"5cent":0.05,"10cent":0.10,
          "20cent":0.20,"50cent":0.50,"1euro":1.00,"2euro":2.00}
FAMILY = {**{c:"copper" for c in ["1cent","2cent","5cent"]},
          **{c:"gold" for c in ["10cent","20cent","50cent"]},
          **{c:"bimetal" for c in ["1euro","2euro"]}}
# Real euro diameters (mm) — physical constants, see spec §1 (Q1 may replace with ref-derived grid)
DIAM_MM = {"1cent":16.25,"2cent":18.75,"10cent":19.75,"5cent":21.25,
           "20cent":22.25,"1euro":23.25,"50cent":24.25,"2euro":25.75}

def target_images():
    return sorted(os.listdir(TARGET_DIR), key=lambda n: image_num(n))

def image_num(name):
    return int(re.search(r"image_(\d+)\.jpg", name).group(1))

def is_dev(name):
    return image_num(name) % 2 == 0
```

- [ ] **Step 4: Eseguire il test e verificare il PASS**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m unittest v2.tests.test_paths -v`
Expected: `OK` (3 test)

- [ ] **Step 5: Commit**

```bash
cd "$P" && git add v2/ && git commit -m "feat(v2): scaffold package, paths, dev/sealed split"
```

---

### Task 2: Schema GT posizionale + merge/validazione (`v2/gtschema.py`)

**Files:**
- Create: `v2/gtschema.py`
- Test: `v2/tests/test_gtschema.py`

- [ ] **Step 1: Scrivere i test fallenti**

```python
# v2/tests/test_gtschema.py
import unittest
from v2.gtschema import validate_entry, merge_agent_output, cross_check_multiset

GOOD = {"coins":[{"cx":424,"cy":477,"r":85,"label":"5cent","sure":True}],
        "tilt":"none","note":""}

class TestSchema(unittest.TestCase):
    def test_validate_ok(self):
        self.assertEqual(validate_entry("image_2.jpg", GOOD), [])
    def test_validate_bad_label_and_tilt(self):
        bad = {"coins":[{"cx":1,"cy":1,"r":10,"label":"3cent","sure":True}],
               "tilt":"verystrong","note":""}
        errs = validate_entry("image_2.jpg", bad)
        self.assertEqual(len(errs), 2)
    def test_merge_agent_output(self):
        seeds = [{"i":0,"cx":100,"cy":100,"r":50},{"i":1,"cx":300,"cy":300,"r":60}]
        agent = {"image":"image_2.jpg","tilt":"mild",
                 "seeds":[{"i":0,"label":"5cent","sure":True,"circle_ok":True},
                          {"i":1,"label":"FP","sure":True,"circle_ok":True}],
                 "extra":[{"cx":500,"cy":500,"r":55,"label":"1euro","sure":False}]}
        entry = merge_agent_output(agent, seeds)
        self.assertEqual(len(entry["coins"]), 2)          # seed FP escluso, extra incluso
        self.assertEqual(entry["coins"][1]["label"], "1euro")
        self.assertFalse(entry["coins"][1]["sure"])
    def test_cross_check(self):
        entry = {"coins":[{"cx":1,"cy":1,"r":9,"label":"5cent","sure":True}],
                 "tilt":"none","note":""}
        out = cross_check_multiset(entry, ["1cent"])       # GT utente dice 1cent
        self.assertFalse(out["coins"][0]["sure"])          # mismatch → sure:false
        self.assertIn("mismatch", out["note"])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Eseguire i test, verificarne il fallimento**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m unittest v2.tests.test_gtschema -v`
Expected: ERROR (`No module named 'v2.gtschema'`)

- [ ] **Step 3: Implementazione**

```python
# v2/gtschema.py
# Positional-GT schema: validation, agent-output merge, user-GT cross-check.
from collections import Counter
from v2.paths import CLASSES

LABELS = set(CLASSES) | {"unknown"}
TILTS = {"none","mild","strong"}

def validate_entry(name, entry):
    errs = []
    if entry.get("tilt") not in TILTS:
        errs.append(f"{name}: bad tilt {entry.get('tilt')!r}")
    for k, c in enumerate(entry.get("coins", [])):
        if c.get("label") not in LABELS:
            errs.append(f"{name} coin{k}: bad label {c.get('label')!r}")
        if not all(isinstance(c.get(f), (int, float)) for f in ("cx","cy","r")):
            errs.append(f"{name} coin{k}: bad geometry")
    return errs

def merge_agent_output(agent, seeds):
    """agent: labeling-agent JSON; seeds: [{'i','cx','cy','r'}] from baseline detector.
    Seed labelled 'FP' is dropped; 'extra' coins (missed by detector) are appended."""
    by_i = {s["i"]: s for s in seeds}
    coins = []
    for s in agent.get("seeds", []):
        if s["label"] == "FP":
            continue
        seed = by_i[s["i"]]
        c = {"cx": s.get("cx", seed["cx"]), "cy": s.get("cy", seed["cy"]),
             "r": s.get("r", seed["r"]), "label": s["label"], "sure": bool(s["sure"])}
        coins.append(c)
    for e in agent.get("extra", []):
        coins.append({"cx": e["cx"], "cy": e["cy"], "r": e["r"],
                      "label": e["label"], "sure": bool(e["sure"])})
    return {"coins": coins, "tilt": agent.get("tilt", "none"),
            "note": agent.get("note", "")}

def cross_check_multiset(entry, user_labels):
    """user_labels: list from fable/gt_labels.json (may carry trailing '?').
    On multiset mismatch every coin of the image is demoted to sure:false."""
    mine = Counter(c["label"] for c in entry["coins"] if c["label"] != "unknown")
    theirs = Counter(l.rstrip("?") for l in user_labels)
    if mine != theirs:
        for c in entry["coins"]:
            c["sure"] = False
        entry["note"] = (entry["note"] + " | " if entry["note"] else "") + \
            f"mismatch vs user GT: mine={dict(mine)} user={dict(theirs)}"
    return entry
```

- [ ] **Step 4: Eseguire i test, verificare PASS**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m unittest v2.tests.test_gtschema -v`
Expected: `OK` (4 test)

- [ ] **Step 5: Commit**

```bash
cd "$P" && git add v2/gtschema.py v2/tests/test_gtschema.py && git commit -m "feat(v2): positional-GT schema, agent merge, user-GT cross-check"
```

---

### Task 3: Preparazione etichettatura (`v2/label_prep.py`)

Genera per ogni target un'immagine con i cerchi seed NUMERATI (dal detector frozen) che gli agenti etichettatori useranno come riferimento visivo, più il JSON dei seed.

**Files:**
- Create: `v2/label_prep.py`
- Output: `v2/label_prep/image_N.jpg` (142), `v2/label_prep/seeds.json`

- [ ] **Step 1: Implementazione (script, niente TDD: output visivo verificato a campione)**

```python
# v2/label_prep.py
# Render numbered baseline seed circles on each target image for labeling agents.
import cv2, json, os
from v2.paths import PROJECT, TARGET_DIR, target_images

OUT = os.path.join(PROJECT, "v2", "label_prep")

def main():
    os.makedirs(OUT, exist_ok=True)
    base = json.load(open(os.path.join(PROJECT, "fable", "baseline", "pred.json")))
    seeds = {}
    for name in target_images():
        img = cv2.imread(os.path.join(TARGET_DIR, name))
        ss = [{"i": k, "cx": c["cx"], "cy": c["cy"], "r": c["r"]}
              for k, c in enumerate(base.get(name, []))]
        seeds[name] = ss
        for s in ss:
            cv2.circle(img, (s["cx"], s["cy"]), s["r"], (0, 255, 0), 3)
            cv2.putText(img, str(s["i"]), (s["cx"] - 15, s["cy"] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        cv2.imwrite(os.path.join(OUT, name), img)
    json.dump(seeds, open(os.path.join(OUT, "seeds.json"), "w"), indent=1)
    print("done", len(seeds))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Eseguire e verificare**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m v2.label_prep`
Expected: `done 142`; aprire (Read) `v2/label_prep/image_85.jpg` e verificare cerchi numerati leggibili.

- [ ] **Step 3: Commit**

```bash
cd "$P" && git add v2/label_prep.py && git commit -m "feat(v2): labeling prep overlays with numbered seeds"
```

---

### Task 4: M0a — Etichettatura multi-agente del GT posizionale

**ORCHESTRAZIONE, non codice.** L'orchestratore (sessione principale) lancia agenti VISIVI in parallelo; ogni agente etichetta un lotto di ~10 immagini GUARDANDO i file (tool Read sui jpg). Autorizzazione multi-agente: esplicita dall'utente.

**Files:**
- Create: `v2/gt_pos.json` (artefatto finale)
- Create: `v2/merge_labels.py` (merge + cross-check)

- [ ] **Step 1: Scrivere il merge script**

```python
# v2/merge_labels.py
# Merge labeling-agent JSON batches into v2/gt_pos.json, cross-checking vs user GT.
import json, os, sys
from v2.paths import PROJECT
from v2.gtschema import validate_entry, merge_agent_output, cross_check_multiset

def main(batch_dir):
    seeds = json.load(open(os.path.join(PROJECT, "v2", "label_prep", "seeds.json")))
    user_gt = json.load(open(os.path.join(PROJECT, "fable", "gt_labels.json")))
    gt, errors = {}, []
    for fn in sorted(os.listdir(batch_dir)):
        if not fn.endswith(".json"):
            continue
        for agent in json.load(open(os.path.join(batch_dir, fn))):
            name = agent["image"]
            entry = merge_agent_output(agent, seeds[name])
            if name in user_gt and user_gt[name].get("coins"):
                entry = cross_check_multiset(entry, user_gt[name]["coins"])
            errors += validate_entry(name, entry)
            gt[name] = entry
    if errors:
        print("VALIDATION ERRORS:\n" + "\n".join(errors)); sys.exit(1)
    out = os.path.join(PROJECT, "v2", "gt_pos.json")
    json.dump(gt, open(out, "w"), indent=1)
    n = sum(len(e["coins"]) for e in gt.values())
    unsure = sum(1 for e in gt.values() for c in e["coins"] if not c["sure"])
    unk = sum(1 for e in gt.values() for c in e["coins"] if c["label"] == "unknown")
    print(f"images={len(gt)} coins={n} unsure={unsure} unknown={unk}")

if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Lanciare l'etichettatura multi-agente (Workflow o Agent paralleli, ~15 lotti × 10 img)**

Prompt-tipo per ogni agente (passare la lista immagini del lotto):

> Sei un etichettatore di monete euro. Per ogni immagine del lotto: (1) Read `coin_dataset/target_set/<name>` (originale) e `v2/label_prep/<name>` (cerchi numerati). (2) Per OGNI cerchio numerato decidi: label ∈ {1cent,2cent,5cent,10cent,20cent,50cent,1euro,2euro,unknown,FP} — FP se il cerchio NON contiene una moneta; unknown se c'è una moneta ma illeggibile anche per te. Se il cerchio è mal centrato/scalato fornisci cx,cy,r corretti. (3) Cerca monete SENZA cerchio → lista `extra` con cx,cy,r stimati e label. (4) Classifica `tilt`: none/mild/strong (monete ellittiche?). (5) `sure:false` su ogni decisione incerta. Indizi: bimetallo = due tonalità concentriche (1euro: anello e nucleo di colore DIVERSO tra loro — riporta quale); rame scuro = 1/2/5c; oro = 10/20/50c; taglia relativa nella scena. Restituisci SOLO JSON: `[{"image":..., "tilt":..., "seeds":[{"i":0,"label":...,"sure":true,"circle_ok":true,"cx":...,"cy":...,"r":...}], "extra":[...], "note":""}]`

Salvare ogni risposta in `v2/label_batches/batch_NN.json`.

- [ ] **Step 3: Merge e validazione**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m v2.merge_labels v2/label_batches`
Expected: `images=142 coins=~320±20 unsure=... unknown=...` senza VALIDATION ERRORS. Se `unknown` > 10% delle monete → segnalare nel report finale (Q4, checkpoint spec).

- [ ] **Step 4: Commit**

```bash
cd "$P" && git add v2/merge_labels.py v2/gt_pos.json v2/label_batches/ && git commit -m "feat(v2): M0a positional GT labelled by visual agents"
```

---

### Task 5: Renderer GT per revisione utente (`v2/gt_overlay.py`)

**Files:**
- Create: `v2/gt_overlay.py`
- Output: `v2/gt_overlays/image_N.jpg` (142) + `v2/gt_overlays/_doubts_sheet.jpg`

- [ ] **Step 1: Implementazione**

```python
# v2/gt_overlay.py
# Render GT circles on every target (green=sure, yellow=doubt, red=unknown)
# plus a contact sheet of doubtful crops for fast human review.
import cv2, json, os
import numpy as np
from v2.paths import PROJECT, TARGET_DIR, target_images

OUT = os.path.join(PROJECT, "v2", "gt_overlays")
COLOR = {"sure": (0, 200, 0), "doubt": (0, 215, 255), "unknown": (0, 0, 255)}

def state(c):
    if c["label"] == "unknown": return "unknown"
    return "sure" if c["sure"] else "doubt"

def main():
    os.makedirs(OUT, exist_ok=True)
    gt = json.load(open(os.path.join(PROJECT, "v2", "gt_pos.json")))
    crops = []
    for name in target_images():
        img = cv2.imread(os.path.join(TARGET_DIR, name))
        for k, c in enumerate(gt[name]["coins"]):
            st = state(c)
            cv2.circle(img, (int(c["cx"]), int(c["cy"])), int(c["r"]), COLOR[st], 3)
            cv2.putText(img, c["label"], (int(c["cx"] - c["r"]), int(c["cy"] - c["r"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR[st], 2)
            if st != "sure":
                x0 = max(0, int(c["cx"] - 1.2 * c["r"])); y0 = max(0, int(c["cy"] - 1.2 * c["r"]))
                x1 = min(img.shape[1], int(c["cx"] + 1.2 * c["r"]))
                y1 = min(img.shape[0], int(c["cy"] + 1.2 * c["r"]))
                crop = cv2.imread(os.path.join(TARGET_DIR, name))[y0:y1, x0:x1]
                crop = cv2.resize(crop, (180, 180))
                cv2.putText(crop, f"{name.replace('image_','').replace('.jpg','')}:{k} {c['label']}",
                            (3, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR[st], 1)
                crops.append(crop)
        cv2.imwrite(os.path.join(OUT, name), img)
    if crops:
        cols = 8
        rows = (len(crops) + cols - 1) // cols
        sheet = np.zeros((rows * 180, cols * 180, 3), np.uint8)
        for k, cr in enumerate(crops):
            r0, c0 = divmod(k, cols)
            sheet[r0 * 180:(r0 + 1) * 180, c0 * 180:(c0 + 1) * 180] = cr
        cv2.imwrite(os.path.join(OUT, "_doubts_sheet.jpg"), sheet)
    print(f"overlays=142 doubts={len(crops)}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Eseguire e verificare visivamente**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m v2.gt_overlay`
Expected: `overlays=142 doubts=N`. Read su `v2/gt_overlays/_doubts_sheet.jpg` + 2 overlay a campione: etichette leggibili, colori corretti.

- [ ] **Step 3: Commit**

```bash
cd "$P" && git add v2/gt_overlay.py && git commit -m "feat(v2): GT overlay renderer + doubts contact sheet (user review on pixels)"
```

---

### Task 6: Eval posizionale (`v2/evalpos.py`)

**Files:**
- Create: `v2/evalpos.py`
- Test: `v2/tests/test_evalpos.py`

- [ ] **Step 1: Test fallenti (fixture sintetiche)**

```python
# v2/tests/test_evalpos.py
import unittest
from v2.evalpos import circle_iou, match_scene, scene_metrics

class TestEval(unittest.TestCase):
    def test_iou_identical(self):
        self.assertAlmostEqual(circle_iou((100,100,50),(100,100,50)), 1.0, places=3)
    def test_iou_disjoint(self):
        self.assertEqual(circle_iou((0,0,10),(100,100,10)), 0.0)
    def test_match_and_metrics(self):
        gt = [{"cx":100,"cy":100,"r":50,"label":"5cent","sure":True},
              {"cx":300,"cy":300,"r":60,"label":"unknown","sure":True}]
        pred = [{"cx":102,"cy":101,"r":51,"pred":"10cent"},
                {"cx":600,"cy":600,"r":40,"pred":"1euro"}]
        m = match_scene(gt, pred)                      # IoU>0.5
        self.assertEqual(m["pairs"], [(0,0)])
        self.assertEqual(m["miss"], [1])
        self.assertEqual(m["fp"], [1])
        s = scene_metrics(gt, pred, m)
        self.assertEqual(s["n_eval"], 1)               # unknown fuori denominatore
        self.assertEqual(s["n_correct"], 0)
        self.assertEqual(s["stage"], ["famiglia"])     # 5cent(copper) vs 10cent(gold)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Eseguire, verificare il fallimento**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m unittest v2.tests.test_evalpos -v`
Expected: ERROR (modulo mancante)

- [ ] **Step 3: Implementazione**

```python
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
```

- [ ] **Step 4: Eseguire i test, verificare PASS**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m unittest v2.tests.test_evalpos -v`
Expected: `OK` (3 test)

- [ ] **Step 5: Commit**

```bash
cd "$P" && git add v2/evalpos.py v2/tests/test_evalpos.py && git commit -m "feat(v2): positional eval with IoU matching and stage attribution"
```

---

### Task 7: M0b — Error budget di e12 + affidabilità per-cue

**Files:**
- Create: `v2/reliability.py`
- Create: `v2/reports/M0b_error_budget.md` (generato + redatto)

- [ ] **Step 1: Error budget di e12**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m v2.evalpos fable/e12_size_constraints/pred.json dev`
Expected: JSON con `coin_acc`, `error_stages` (detection_miss / famiglia / taglia / shape_single), `miss`, `fp`. Salvare l'output nel report.

- [ ] **Step 2: Affidabilità per-cue**

```python
# v2/reliability.py
# Per-cue reliability measured on dev-half GT (MEASUREMENT ONLY, never calibration).
# famiglia: e13 tone chain; taglia: same-size cluster purity; shape: shape_only pred.
import json, math, os
import cv2
import numpy as np
from v2.paths import PROJECT, TARGET_DIR, FAMILY, is_dev

import sys
sys.path.insert(0, os.path.join(PROJECT, "fable", "e13_two_tone_family"))
from run import decast_11, lnorm13, tone13, PROTO  # riusa la catena e13 e i prototipi dai ref

def family_reliability(gt):
    ok = tot = 0
    for name, entry in gt.items():
        if not is_dev(name): continue
        img = cv2.imread(os.path.join(TARGET_DIR, name))
        dc = decast_11(img)
        for c in entry["coins"]:
            if c["label"] == "unknown" or FAMILY[c["label"]] == "bimetal": continue
            dev = tone13(lnorm13(dc), int(c["cx"]), int(c["cy"]), int(c["r"]))
            fam = min(PROTO, key=lambda f: np.hypot(*(np.array(dev) - np.array(PROTO[f]))))
            tot += 1; ok += (fam == FAMILY[c["label"]])
    return ok, tot

def size_cluster_purity(gt, tol=0.030):
    pure = tot = 0
    for name, entry in gt.items():
        if not is_dev(name) or len(entry["coins"]) < 2: continue
        cs = [c for c in entry["coins"] if c["label"] != "unknown"]
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                same_r = abs(math.log(cs[a]["r"] / cs[b]["r"])) <= tol
                same_l = cs[a]["label"] == cs[b]["label"]
                tot += 1; pure += (same_r == same_l)
    return pure, tot

def shape_reliability(gt):
    sh = json.load(open(os.path.join(PROJECT, "fable", "shape_only", "pred.json")))
    from v2.evalpos import match_scene
    ok = tot = okfam = 0
    for name, entry in gt.items():
        if not is_dev(name): continue
        m = match_scene(entry["coins"], sh.get(name, []))
        for i, j in m["pairs"]:
            g, p = entry["coins"][i], sh[name][j]
            if g["label"] == "unknown": continue
            tot += 1; ok += (p["pred"] == g["label"])
            okfam += (FAMILY[p["pred"]] == FAMILY[g["label"]])
    return ok, okfam, tot

if __name__ == "__main__":
    gt = json.load(open(os.path.join(PROJECT, "v2", "gt_pos.json")))
    fo, ft = family_reliability(gt)
    sp, st = size_cluster_purity(gt)
    so, sf, stt = shape_reliability(gt)
    print(f"famiglia-colore: {fo}/{ft} = {fo/max(1,ft):.3f}")
    print(f"same-size purity (tol .030): {sp}/{st} = {sp/max(1,st):.3f}")
    print(f"shape top-1: {so}/{stt} = {so/max(1,stt):.3f}  famiglia-da-shape: {sf}/{stt} = {sf/max(1,stt):.3f}")
```

NOTA per l'esecutore: i nomi `PROTO`/firme in `fable/e13_two_tone_family/run.py` vanno verificati a inizio task (`grep -n "PROTO\|def tone13\|def lnorm13" fable/e13_two_tone_family/run.py`); se differiscono, adattare l'import mantenendo la catena di misura identica (de-cast → stretch L* → tono interno − background).

- [ ] **Step 3: Eseguire**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m v2.reliability`
Expected: tre righe con frazioni e percentuali; nessuna eccezione.

- [ ] **Step 4: Redigere `v2/reports/M0b_error_budget.md`**

Contenuto obbligatorio: (1) tabella error_stages di e12 su metà sviluppo ("dove vivono i punti mancanti"); (2) tabella affidabilità per-cue; (3) **checkpoint 0.85**: se `detection_miss + quota unknown > 10 punti` scrivere esplicitamente che il target va rinegoziato e con quali numeri; (4) raccomandazione su dove investire in M1/M2.

- [ ] **Step 5: Commit**

```bash
cd "$P" && git add v2/reliability.py v2/reports/ && git commit -m "feat(v2): M0b error budget + per-cue reliability report"
```

---

### Task 8: Questioni aperte Q1–Q4 (`v2/q_open.py`)

**Files:**
- Create: `v2/q_open.py`
- Modify: `v2/reports/M0b_error_budget.md` (append sezione "Questioni aperte")

- [ ] **Step 1: Implementazione**

```python
# v2/q_open.py
# Q1 ref scale sharing, Q2 bimetal ring sign, Q3 tilt frequency, Q4 unknown rate.
import json, os
import cv2
import numpy as np
from v2.paths import PROJECT, REF_DIR, DIAM_MM, CLASSES

def q1_ref_scale():
    rows = []
    for cls in CLASSES:
        img = cv2.imread(os.path.join(REF_DIR, f"{cls}.jpg"))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (9, 9), 2)
        cc = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 200, param1=100,
                              param2=30, minRadius=60, maxRadius=min(g.shape)//2)
        r = float(cc[0][0][2]) if cc is not None else float("nan")
        rows.append((cls, r, r / DIAM_MM[cls]))   # px/mm: costante ⇔ scala condivisa
    return rows

def q2_ring_sign():
    out = {}
    for cls in ("1euro", "2euro"):
        img = cv2.imread(os.path.join(REF_DIR, f"{cls}.jpg"))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (9, 9), 2)
        cc = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 200, param1=100,
                              param2=30, minRadius=60, maxRadius=min(g.shape)//2)
        cx, cy, r = cc[0][0]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        rho = np.hypot(xx - cx, yy - cy) / r
        core = lab[rho < 0.45][:, 2].mean()        # b*: oro alto, argento basso
        ring = lab[(rho > 0.75) & (rho < 0.95)][:, 2].mean()
        out[cls] = {"core_b": round(float(core), 1), "ring_b": round(float(ring), 1),
                    "sign": "gold-ring" if ring > core else "silver-ring"}
    return out

def q3_q4():
    gt = json.load(open(os.path.join(PROJECT, "v2", "gt_pos.json")))
    tilt = {"none": 0, "mild": 0, "strong": 0}
    unk = tot = 0
    for e in gt.values():
        tilt[e["tilt"]] += 1
        for c in e["coins"]:
            tot += 1; unk += (c["label"] == "unknown")
    return tilt, unk, tot

if __name__ == "__main__":
    print("Q1 ref px/mm:", *[f"{c}:{px:.1f}/{ratio:.2f}" for c, px, ratio in q1_ref_scale()])
    print("Q2 ring sign:", q2_ring_sign())
    t, u, n = q3_q4()
    print(f"Q3 tilt: {t}   Q4 unknown: {u}/{n} = {u/max(1,n):.1%}")
```

- [ ] **Step 2: Eseguire e interpretare**

Run: `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m v2.q_open`
Interpretazione da scrivere nel report: Q1 — se i px/mm degli 8 ref hanno CV < 5% la scala è condivisa → griglia dai ref; Q2 — chiude la questione del verso (BCE vs memoria utente) CON I NUMERI; Q3 — dimensiona M1c; Q4 — alimenta il checkpoint.

- [ ] **Step 3: Commit**

```bash
cd "$P" && git add v2/q_open.py v2/reports/ && git commit -m "feat(v2): open questions Q1-Q4 measured"
```

---

### Task 9: M1a — Prefilter misurato (`v2/prefilter.py` + bench)

**Files:**
- Create: `v2/prefilter.py`
- Test: `v2/tests/test_prefilter.py`

- [ ] **Step 1: Test fallente**

```python
# v2/tests/test_prefilter.py
import unittest
import numpy as np
from v2.prefilter import VARIANTS, apply

class TestPrefilter(unittest.TestCase):
    def test_variants_run_and_preserve_shape(self):
        img = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
        for v in VARIANTS:
            out = apply(img, v)
            self.assertEqual(out.shape, img.shape, v)
            self.assertEqual(out.dtype, np.uint8, v)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Eseguire, verificare fallimento** — `cd "$P" && "/home/gianmarco/envs/IP&CV/bin/python" -m unittest v2.tests.test_prefilter -v` → ERROR

- [ ] **Step 3: Implementazione**

```python
# v2/prefilter.py
# Candidate prefilters (from fable/legibility sweep + e11 recipe). Chosen ONE
# by measured detection metrics (M1a bench), not by eye.
import cv2
import numpy as np

VARIANTS = ["identity", "nlm7", "bilateral", "median5", "e11"]

def _stretch(img, lo=2, hi=98):
    g = img.astype(np.float32)
    a, b = np.percentile(g, lo), np.percentile(g, hi)
    return np.clip((g - a) * 255.0 / max(1e-6, b - a), 0, 255).astype(np.uint8)

def apply(img, variant):
    if variant == "identity": return img.copy()
    if variant == "nlm7": return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    if variant == "bilateral": return cv2.bilateralFilter(img, 9, 75, 75)
    if variant == "median5": return cv2.medianBlur(img, 5)
    if variant == "e11":   # global percentile stretch + NLM (per-disc version arrives in M2 cues)
        st = _stretch(img)
        return cv2.fastNlMeansDenoisingColored(st, None, 7, 7, 7, 21)
    raise ValueError(variant)
```

- [ ] **Step 4: Test PASS** — stesso comando → `OK`

- [ ] **Step 5: Bench su metà sviluppo (dopo Task 10: usa detect+validate).** Segnaposto consapevole: il bench M1a si esegue nello Step 6 del Task 10, perché richiede il detector. Ordine reale di esecuzione: Task 9 (modulo) → Task 10 (detector) → bench congiunto.

- [ ] **Step 6: Commit**

```bash
cd "$P" && git add v2/prefilter.py v2/tests/test_prefilter.py && git commit -m "feat(v2): prefilter candidates module"
```

---

### Task 10: M1b — Detection generatore+validatore (`v2/detect.py`, `v2/validate.py`)

**Files:**
- Create: `v2/detect.py`, `v2/validate.py`
- Test: `v2/tests/test_detect.py`
- Output bench: `v2/reports/M1_detection.md`

- [ ] **Step 1: Test fallenti (immagine sintetica: 2 dischi pieni + 1 candidato fantasma)**

```python
# v2/tests/test_detect.py
import unittest
import numpy as np, cv2
from v2.detect import candidates
from v2.validate import validate

def synth():
    img = np.full((400, 600, 3), 30, np.uint8)
    cv2.circle(img, (150, 200), 70, (180, 160, 140), -1)
    cv2.circle(img, (430, 180), 55, (190, 170, 150), -1)
    return img

class TestDetect(unittest.TestCase):
    def test_candidates_find_both(self):
        cs = candidates(synth())
        self.assertGreaterEqual(len(cs), 2)
        self.assertTrue(any(abs(c[0]-150) < 8 and abs(c[1]-200) < 8 for c in cs))
    def test_validator_accepts_real_rejects_ghost(self):
        img = synth()
        ok = validate(img, 150, 200, 70)
        self.assertTrue(ok["accept"])
        self.assertLess(abs(ok["r"] - 70), 5)
        ghost = validate(img, 300, 320, 60)     # nessun bordo coerente lì
        self.assertFalse(ghost["accept"])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Eseguire, verificare fallimento** — `... -m unittest v2.tests.test_detect -v` → ERROR

- [ ] **Step 3: Implementazione `detect.py` (generatore permissivo)**

```python
# v2/detect.py
# Hough DEMOTED to permissive candidate generator: two passes, dedup by center.
import cv2
import numpy as np

def candidates(img, r_min=30, r_max=130):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (9, 9), 2)
    out = []
    for p2, mind in ((38, 85), (30, 70)):
        cc = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, mind, param1=100,
                              param2=p2, minRadius=r_min, maxRadius=r_max)
        if cc is None: continue
        for cx, cy, r in cc[0]:
            if all((cx-x)**2 + (cy-y)**2 > (0.5*min(r, rr))**2 for x, y, rr in out):
                out.append((float(cx), float(cy), float(r)))
    return out
```

- [ ] **Step 4: Implementazione `validate.py` (porting e4: selezione bordi per-raggio + Kasa + edge-support)**

Fonte da consultare: `fable/e4_precise_radius/run.py` righe 49–113 (`_kasa`, `refine_radial`, `edge_support`). Implementazione autonoma equivalente:

```python
# v2/validate.py
# Geometric validator (port of e4): per-ray strongest radial edge, MAD outlier
# rejection, Kasa LSQ circle fit, edge-support accept gate. Also exports the
# selected edge points for the ellipse fit (rectify.py).
import math
import cv2
import numpy as np

N_RAYS, MAX_DR, MIN_SUPPORT, MIN_RAYS = 72, 14, 0.45, 20

def _kasa(xs, ys):
    A = np.c_[2*xs, 2*ys, np.ones(len(xs))]
    b = xs**2 + ys**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    return cx, cy, math.sqrt(sol[2] + cx*cx + cy*cy)

def _ray_edges(gray, cx, cy, r0):
    """Strongest outward radial gradient per ray within r0±MAX_DR."""
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    pts = []
    for k in range(N_RAYS):
        th = 2*math.pi*k/N_RAYS
        ux, uy = math.cos(th), math.sin(th)
        best, best_mag = None, 40.0          # soglia minima gradiente
        for dr in np.arange(-MAX_DR, MAX_DR + 0.5, 0.5):
            x, y = cx + (r0+dr)*ux, cy + (r0+dr)*uy
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < gray.shape[0] and 0 <= xi < gray.shape[1]): continue
            mag = abs(sob_x[yi, xi]*ux + sob_y[yi, xi]*uy)   # gradiente RADIALE
            if mag > best_mag: best, best_mag = (x, y), mag
        if best: pts.append(best)
    return pts

def validate(img, cx, cy, r0):
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 100, 100)
    pts = _ray_edges(gray, cx, cy, r0)
    support = len(pts) / N_RAYS
    if len(pts) < MIN_RAYS or support < MIN_SUPPORT:
        return {"accept": False, "cx": cx, "cy": cy, "r": r0,
                "support": support, "edge_pts": pts}
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    for _ in range(2):                                   # 2 iterazioni con rigetto MAD
        ncx, ncy, nr = _kasa(xs, ys)
        res = np.abs(np.hypot(xs - ncx, ys - ncy) - nr)
        mad = np.median(res) + 1e-6
        keep = res < 3 * mad + 1.0
        if keep.sum() < MIN_RAYS: break
        xs, ys = xs[keep], ys[keep]
    ncx, ncy, nr = _kasa(xs, ys)
    if abs(nr - r0) > MAX_DR:                            # clamp anti-deriva
        ncx, ncy, nr = cx, cy, r0
    return {"accept": True, "cx": float(ncx), "cy": float(ncy), "r": float(nr),
            "support": support, "edge_pts": list(zip(xs.tolist(), ys.tolist()))}
```

- [ ] **Step 5: Test PASS** — `... -m unittest v2.tests.test_detect -v` → `OK` (2 test). Se il ghost passa, alzare `MIN_SUPPORT` a 0.50 e ri-testare.

- [ ] **Step 6: Bench congiunto M1a+M1b su metà sviluppo**

Scrivere ed eseguire uno script usa-e-getta `v2/bench_m1.py`: per ogni variante prefilter → `candidates` → `validate` → pred.json provvisorio (pred="1cent" fittizio, conta solo la geometria) → `evalpos.evaluate(..., "dev")` ma SOLO metriche detection (miss/fp/center/radius). Tabella in `v2/reports/M1_detection.md`: righe = varianti, colonne = miss, fp, center_err_med, radius_errpc_med + confronto con riga "frozen baseline" (`fable/baseline/pred.json`).
**Criterio d'uscita spec:** miss+FP dimezzati vs baseline, center_err_med ≤ 2px. Se non raggiunto: documentare nel report il miglior compromesso raggiunto e NON forzare le soglie (decisione rimandata alla revisione utente).

- [ ] **Step 7: Commit**

```bash
cd "$P" && git add v2/detect.py v2/validate.py v2/tests/test_detect.py v2/bench_m1.py v2/reports/M1_detection.md && git commit -m "feat(v2): M1 detection generator+validator with measured bench"
```

---

### Task 11: M1c — Rettifica prospettica (`v2/rectify.py`)

**Files:**
- Create: `v2/rectify.py`
- Test: `v2/tests/test_rectify.py`

- [ ] **Step 1: Test fallenti (ellisse sintetica → dopo rettifica torna cerchio)**

```python
# v2/tests/test_rectify.py
import unittest
import numpy as np, cv2
from v2.rectify import fit_tilt, rectify_disc

class TestRectify(unittest.TestCase):
    def _ellipse_pts(self, cx, cy, a, b, angle_deg):
        th = np.linspace(0, 2*np.pi, 72, endpoint=False)
        x, y = a*np.cos(th), b*np.sin(th)
        ar = math_rad = np.deg2rad(angle_deg)
        xr = cx + x*np.cos(ar) - y*np.sin(ar)
        yr = cy + x*np.sin(ar) + y*np.cos(ar)
        return list(zip(xr.tolist(), yr.tolist()))
    def test_fit_detects_tilt(self):
        pts = self._ellipse_pts(200, 200, 100, 80, 30)
        t = fit_tilt(pts)
        self.assertIsNotNone(t)
        self.assertLess(abs(t["ratio"] - 0.8), 0.05)
        self.assertLess(min(abs(t["angle"] - 30), abs(t["angle"] - 210)) % 180, 8)
    def test_fit_circle_returns_none(self):
        pts = self._ellipse_pts(200, 200, 100, 99, 0)   # quasi cerchio
        self.assertIsNone(fit_tilt(pts))
    def test_rectify_makes_circle(self):
        img = np.zeros((400, 400, 3), np.uint8)
        cv2.ellipse(img, (200, 200), (100, 80), 30, 0, 360, (255, 255, 255), -1)
        pts = self._ellipse_pts(200, 200, 100, 80, 30)
        disc, new_r = rectify_disc(img, 200, 200, fit_tilt(pts))
        self.assertAlmostEqual(new_r, 100, delta=3)
        g = cv2.cvtColor(disc, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(g > 128)
        h = ys.max() - ys.min(); w = xs.max() - xs.min()
        self.assertLess(abs(h - w) / max(h, w), 0.05)    # tornato tondo

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Eseguire, verificare fallimento** — `... -m unittest v2.tests.test_rectify -v` → ERROR

- [ ] **Step 3: Implementazione**

```python
# v2/rectify.py
# Perspective rectification: ellipse fit on validator edge points; if eccentric,
# anisotropic unwarp back to a circle. True size radius = semi-MAJOR axis.
import math
import cv2
import numpy as np

MIN_PTS, RATIO_THR = 12, 0.97   # sotto 12 punti niente fit; ratio>0.97 = già tondo

def fit_tilt(edge_pts):
    if len(edge_pts) < MIN_PTS: return None
    pts = np.array(edge_pts, np.float32).reshape(-1, 1, 2)
    (ecx, ecy), (d1, d2), ang = cv2.fitEllipse(pts)
    major, minor = max(d1, d2) / 2, min(d1, d2) / 2
    ratio = minor / major
    if ratio > RATIO_THR: return None
    if d1 < d2: ang = (ang + 90) % 180        # ang = direzione asse MAGGIORE
    return {"cx": ecx, "cy": ecy, "major": major, "ratio": ratio, "angle": ang}

def rectify_disc(img, cx, cy, tilt, pad=1.35):
    """Crop around (cx,cy), rotate so minor axis is vertical, stretch it to major.
    Returns (rectified BGR crop centered on the coin, true radius = major)."""
    R = tilt["major"]
    s = int(pad * R)
    a = math.radians(tilt["angle"])
    # rotation to axis frame + anisotropic scale + back, around the coin center
    M1 = cv2.getRotationMatrix2D((cx, cy), tilt["angle"], 1.0)
    M1 = np.vstack([M1, [0, 0, 1]])
    S = np.array([[1, 0, 0], [0, 1/tilt["ratio"], 0], [0, 0, 1]], np.float32)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float32)
    Tb = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float32)
    M2 = cv2.getRotationMatrix2D((cx, cy), -tilt["angle"], 1.0)
    M2 = np.vstack([M2, [0, 0, 1]])
    M = M2 @ Tb @ S @ T @ M1
    warped = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x0, y0 = int(cx - s), int(cy - s)
    x0c, y0c = max(0, x0), max(0, y0)
    crop = warped[y0c:int(cy + s), x0c:int(cx + s)]
    return crop, float(R)

def scene_tilt(tilts):
    """Shared scene tilt: median ratio/angle if >=2 coins agree, else None."""
    ts = [t for t in tilts if t is not None]
    if len(ts) < 2: return None
    angs = np.array([t["angle"] for t in ts])
    rats = np.array([t["ratio"] for t in ts])
    if np.ptp(angs) > 25 and np.ptp((angs + 90) % 180) > 25: return None
    return {"angle": float(np.median(angs)), "ratio": float(np.median(rats))}
```

- [ ] **Step 4: Test PASS** — `... -m unittest v2.tests.test_rectify -v` → `OK` (3 test)

- [ ] **Step 5: Misura su scene tilt≠none (dal GT, Q3)**

Estendere `v2/bench_m1.py` con flag `--rectify`: per le immagini GT con `tilt != "none"`, confrontare `radius_errpc_med` con e senza rettifica (raggio = semiasse maggiore). Append risultato a `v2/reports/M1_detection.md`. Criterio spec: errore raggio ridotto su tilt≠none, NESSUNA regressione sulle piatte.

- [ ] **Step 6: Commit**

```bash
cd "$P" && git add v2/rectify.py v2/tests/test_rectify.py v2/bench_m1.py v2/reports/M1_detection.md && git commit -m "feat(v2): M1c perspective rectification measured on tilted scenes"
```

---

### Task 12: Chiusura — report per l'utente + piano Fase 2

**Files:**
- Create: `v2/reports/FASE0-1_SUMMARY.md`
- Create: `docs/superpowers/plans/2026-06-10-reingegnerizzazione-fase2-DRAFT.md`

- [ ] **Step 1: Redigere `v2/reports/FASE0-1_SUMMARY.md`** — in italiano, per l'utente al ritorno: (1) stato GT + **dove guardare**: `v2/gt_overlays/_doubts_sheet.jpg` e overlay; (2) error budget e verdetto checkpoint 0.85; (3) risposte Q1–Q4 (incluso il verdetto sul verso anello/nucleo); (4) numeri detection nuova vs frozen; (5) decisioni che restano all'utente.

- [ ] **Step 2: Stesura piano Fase 2 (DRAFT)** — struttura task per M2 (cues.py, refs.py, scene.py DP famiglia×taglia), M3 (sensibilità pesi, dossier template), M4 (solution_2.ipynb), coi pesi/priorità AGGIORNATI dall'error budget misurato. È un draft: si finalizza con l'utente.

- [ ] **Step 3: Commit finale e push se remoto configurato (NO push se assente)**

```bash
cd "$P" && git add -A v2/ && git add -f docs/superpowers/plans/ && git commit -m "docs(v2): fase 0-1 summary report + fase 2 draft plan"
```

---

## Self-review (eseguita)

- **Copertura spec:** M0a→T3/T4, revisione pixel→T5, M0b→T7, Q1–Q4→T8, M1a→T9, M1b→T10, M1c→T11, gating M2→T12 (piano separato come da spec §8). Formati §7→T2/T6. Fallback runtime §5b → dentro validate (clamp/reject) e rectify (MIN_PTS); il resto matura in Fase 2.
- **Placeholder:** l'unico rinvio dichiarato è il bench M1a spostato nello Step 6 del Task 10 (dipendenza reale dal detector), esplicitato in entrambi i task.
- **Coerenza tipi/firme:** `validate()` ritorna `edge_pts` consumati da `fit_tilt()`; `evaluate(path, half)` usato in T7/T10; `VARIANTS/apply` usati nel bench. Import e13 in T7 marcato con verifica firme obbligatoria a inizio task.
