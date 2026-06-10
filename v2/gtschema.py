# Positional-GT schema: validation, agent-output merge, user-GT cross-check.
from collections import Counter
from v2.paths import CLASSES

LABELS = set(CLASSES) | {"unknown"}
TILTS = {"none", "mild", "strong"}


def validate_entry(name, entry):
    errs = []
    if entry.get("tilt") not in TILTS:
        errs.append(f"{name}: bad tilt {entry.get('tilt')!r}")
    for k, c in enumerate(entry.get("coins", [])):
        if c.get("label") not in LABELS:
            errs.append(f"{name} coin{k}: bad label {c.get('label')!r}")
        if not all(isinstance(c.get(f), (int, float)) for f in ("cx", "cy", "r")):
            errs.append(f"{name} coin{k}: bad geometry")
    return errs


def merge_agent_output(agent, seeds):
    """agent: labeling-agent JSON; seeds: [{'i','cx','cy','r'}] from baseline detector.
    Seed labelled 'FP' is dropped; 'extra' coins (missed by detector) are appended.
    Precondition: agent seed indices must exist in seeds; raises KeyError otherwise (fail-fast)."""
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
