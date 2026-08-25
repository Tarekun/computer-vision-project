"""EXPERIMENTS.md item 3: is size-first two-stage classification (restrict a
confidently-resolved coin's beam-search candidates to its family) worth it, once the
two new gate thresholds (FAMILY_GATE_CONF, FAMILY_GATE_BIM) get to be searched
alongside the other 18 fusion weights?

Sets `C.TWO_STAGE_SIZE_FIRST = True` for the whole run (a structural switch, not a
continuous weight) and temporarily extends tune.py's module-level BOUNDS/DEFAULTS
with the two gate thresholds, so the standard k=4 CV harness (random_search +
coordinate_ascent, unchanged) searches 20 parameters instead of 18. Does not touch
tune.py's BOUNDS/DEFAULTS on disk -- only monkeypatches the module attributes for
this process, exactly like classify.py's weights are monkeypatched at evaluation
time.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experimental.tune as T
from experimental import config as C

T.BOUNDS = {**T.BOUNDS, "FAMILY_GATE_CONF": (0.3, 0.95), "FAMILY_GATE_BIM": (0.3, 0.95)}
T.DEFAULTS = {**T.DEFAULTS, "FAMILY_GATE_CONF": 0.6, "FAMILY_GATE_BIM": 0.6}

RESULTS_PATH = Path(__file__).resolve().parent / "two_stage_search_results.json"


def main():
    cache, offdiag = T.load_cache()
    C.TWO_STAGE_SIZE_FIRST = True
    fold_results, mean_val = T.cross_validate(cache, offdiag, seed=0, k=4, n_trials=500, passes=2)
    print(f"\nTWO_STAGE_SIZE_FIRST=True, mean held-out amount_acc = {mean_val:.4f}")
    with open(RESULTS_PATH, "w") as f:
        json.dump({"fold_results": fold_results, "mean_val_amount_acc": mean_val}, f, indent=2)


if __name__ == "__main__":
    main()
