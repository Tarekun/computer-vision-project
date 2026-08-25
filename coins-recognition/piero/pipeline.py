"""End-to-end entrypoint for the piero pipeline: detection + classification,
wired to the same interface as coins-recognition/pipeline.py's
initialize_ght/evaluate.
"""
from pathlib import Path

import cv2

from . import config as C
from .detection import detect_coins
from .classify import classify
from .reference import build_reference_model


def initialize(reference_dir):
    """Builds the model (colour atlas + blurred edge atlas) from the reference_set images."""
    return build_reference_model(reference_dir)


def process_image(img, model):
    """Detects and classifies every coin in `img`. Returns one dict per coin with
    cx/cy/r (pixel geometry), pred (denomination) and value (euros)."""
    coins = detect_coins(img)
    if not coins:
        return []
    labels = classify(img, coins, model)
    return [
        {"cx": cx, "cy": cy, "r": r, "pred": lab, "value": C.VALUE_EUR[lab]}
        for (cx, cy, r), lab in zip(coins, labels)
    ]


def initialize_ght(reference_dir):
    """Alias of `initialize`, matching pipeline.py's initialize_ght(reference_dir) name."""
    return initialize(reference_dir)


def evaluate(image, ght_models):
    """Matches pipeline.py's evaluate(image, ght_models) contract: returns a list of
    (denomination, center_x, center_y, radius) tuples."""
    coins = process_image(image, ght_models)
    return [(c["pred"], c["cx"], c["cy"], c["r"]) for c in coins]


def _main():
    """Standalone run over the whole target set, in the notebook's original output format."""
    data_dir = Path(__file__).resolve().parents[2] / "data" / "coin_dataset"
    model = initialize(data_dir / "reference_set")

    total = 0.0
    target_dir = data_dir / "target_set"
    for path in sorted(target_dir.iterdir(), key=lambda p: int(p.stem.split("_")[1])):
        coins = process_image(cv2.imread(str(path)), model)
        print(f"{path.name} - {len(coins)} coin(s) found:")
        for i, c in enumerate(coins, 1):
            print(f"  Coin {i} {{value: {c['value']:.3f}}}")
        partial = sum(c["value"] for c in coins)
        total += partial
        print(f"  Partial Amount {{value: {partial:.3f}}}")
    print(f"\nTotal Amount: {total:.2f} EUR")


if __name__ == "__main__":
    _main()
