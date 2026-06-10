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
