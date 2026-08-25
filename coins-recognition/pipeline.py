import csv
from pathlib import Path

import cv2

from giacomo.pipeline import initialize_ght as _giacomo_initialize_ght
from giacomo.pipeline import evaluate as _giacomo_evaluate
from tiziano.pipeline import initialize_ght as _tiziano_initialize_ght
from tiziano.pipeline import evaluate as _tiziano_evaluate
from piero.pipeline import initialize_ght as _piero_initialize_ght
from piero.pipeline import evaluate as _piero_evaluate
from reconfigured.pipeline import initialize_ght as _reconfigured_initialize_ght
from reconfigured.pipeline import evaluate as _reconfigured_evaluate
from experimental.pipeline import initialize_ght as _experimental_initialize_ght
from experimental.pipeline import evaluate as _experimental_evaluate

# which implementation initialize_ght/evaluate delegate to: "giacomo", "tiziano", "piero", "reconfigured" or "experimental"
IMPLEMENTATION = "piero"

_IMPLEMENTATIONS = {
    "giacomo": (_giacomo_initialize_ght, _giacomo_evaluate),
    "tiziano": (_tiziano_initialize_ght, _tiziano_evaluate),
    "piero": (_piero_initialize_ght, _piero_evaluate),
    "reconfigured": (_reconfigured_initialize_ght, _reconfigured_evaluate),
    "experimental": (_experimental_initialize_ght, _experimental_evaluate),
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "coin_dataset"
REFERENCE_DIR = DATA_DIR / "reference_set"
TARGET_DIR = DATA_DIR / "target_set"
PROCESSED_DIR = DATA_DIR / "processed"
GT_PATH = DATA_DIR / "gt.csv"

# value in euros for each reference image (filename stem -> value)
COIN_VALUES = {
    "1cent": 0.01,
    "2cent": 0.02,
    "5cent": 0.05,
    "10cent": 0.10,
    "20cent": 0.20,
    "50cent": 0.50,
    "1euro": 1.00,
    "2euro": 2.00,
}

# distinct BGR color per denomination (evenly spaced hues, full saturation), so
# detections are easy to tell apart in the annotated output images
COIN_COLORS = {
    "1cent": (0, 0, 255),
    "2cent": (0, 187, 255),
    "5cent": (0, 255, 127),
    "10cent": (60, 255, 0),
    "20cent": (255, 255, 0),
    "50cent": (255, 68, 0),
    "1euro": (255, 0, 119),
    "2euro": (195, 0, 255),
}
DEFAULT_COLOR = (0, 255, 0)


def initialize_ght(reference_dir=REFERENCE_DIR):
    """Initializes the GHT model using the reference_set directory"""
    init_fn, _ = _IMPLEMENTATIONS[IMPLEMENTATION]
    return init_fn(reference_dir)


def evaluate(image, ght_models):
    """With the GHT initialized, detects coins in a single target image.

    Returns a list of detections, one per detected coin, each a tuple
    (denomination, center_x, center_y, radius).
    """
    _, evaluate_fn = _IMPLEMENTATIONS[IMPLEMENTATION]
    return evaluate_fn(image, ght_models)


def load_ground_truth(gt_path=GT_PATH):
    ground_truth = {}
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth[row["filename"]] = {
                "total": float(row["total"]),
                "coins": int(row["coins"]),
            }
    return ground_truth


def draw_detections(image, detections):
    annotated = image.copy()
    for denomination, x, y, radius in detections:
        center = (int(round(x)), int(round(y)))
        radius = int(round(radius))
        color = COIN_COLORS.get(denomination, DEFAULT_COLOR)
        cv2.circle(annotated, center, radius, color, 2)
        cv2.putText(
            annotated,
            denomination,
            (center[0] - radius, center[1] - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth()
    ght_models = initialize_ght()

    target_files = sorted(
        p for p in TARGET_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    count_matches = 0
    amount_matches = 0
    mistakes = []

    for image_path in target_files:
        filename = image_path.name
        gt = ground_truth.get(filename)
        if gt is None:
            continue

        image = cv2.imread(str(image_path))
        detections = evaluate(image, ght_models) or []

        detected_count = len(detections)
        detected_total = round(sum(COIN_VALUES.get(d, 0.0) for d, *_ in detections), 2)

        count_ok = detected_count == gt["coins"]
        amount_ok = abs(detected_total - gt["total"]) < 1e-6

        if count_ok:
            count_matches += 1
        if amount_ok:
            amount_matches += 1
        if not (count_ok and amount_ok):
            mistakes.append(filename)

        annotated = draw_detections(image, detections)
        cv2.imwrite(str(PROCESSED_DIR / filename), annotated)

    total_images = len(target_files)
    count_accuracy = count_matches / total_images if total_images else 0.0
    amount_accuracy = amount_matches / total_images if total_images else 0.0

    print(f"Processed {total_images} images")
    print(f"Coin count accuracy: {count_accuracy:.2%} ({count_matches}/{total_images})")
    print(
        f"Total amount accuracy: {amount_accuracy:.2%} ({amount_matches}/{total_images})"
    )

    if mistakes:
        print("Images with mistakes:")
        for filename in mistakes:
            print(f"  {filename}")


if __name__ == "__main__":
    pipeline()
