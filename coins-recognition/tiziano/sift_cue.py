"""The SIFT matching cue this variant adds on top of giacomo's shape/colour
cues: reference descriptors are extracted from the raw reference photos
(not the cropped canonical disc), and a target crop is scored against every
class's cached descriptors via ratio-test kNN matching.

This is a straight port of solution_2_T.ipynb's SIFT cell -- including its
choice to detect keypoints across the whole reference photo (coin *and*
background) rather than the canonical disc crop the shape cue (appearance.py)
uses. That is a real difference in approach from giacomo's atlas, not a bug
fixed during porting; see tiziano/README.md.
"""
from pathlib import Path

import cv2
import numpy as np

from . import config as C


def preprocess_for_sift(gray):
    """CLAHE + bilateral filter to enhance edges before SIFT keypoint detection."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)


def calibrate_sift_templates(template_paths):
    """Builds {class_name: descriptor array} from the reference photos.

    This is the SIFT half of the reference model (paired with the shape
    atlas built in reference.py): a one-off computation over reference_set,
    reused for every target image.
    """
    sift_cache = {}
    sift = cv2.SIFT_create(nfeatures=C.SIFT_NFEATURES, contrastThreshold=C.SIFT_CONTRAST)

    for cls, path in template_paths.items():
        if not Path(path).exists():
            print(f"  [WARN] Template NON trovato: {path}")
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = preprocess_for_sift(img)
        kp, des = sift.detectAndCompute(img, None)

        if des is not None and len(kp) > 0:
            sorted_idx = sorted(range(len(kp)), key=lambda i: kp[i].response, reverse=True)
            keep, kept_pos = [], []
            for idx in sorted_idx:
                pt = kp[idx].pt
                too_close = False
                for p in kept_pos:
                    d = np.sqrt((pt[0] - p[0]) ** 2 + (pt[1] - p[1]) ** 2)
                    if d < C.SIFT_NMS_DIST:
                        too_close = True
                        break
                if not too_close:
                    keep.append(idx)
                    kept_pos.append(pt)
            des = des[keep] if keep else None

        n_des = 0 if des is None else len(des)
        sift_cache[cls] = des
        print(f"  OK {cls:8s}: {n_des} descrittori (dopo NMS)")

    print(f"SIFT cache: {len(sift_cache)} entries")
    return sift_cache


def sift_extract_with_nms(gray_roi):
    """Extracts SIFT descriptors from a target crop and applies spatial NMS."""
    gray_proc = preprocess_for_sift(gray_roi)
    sift = cv2.SIFT_create(nfeatures=C.SIFT_NFEATURES, contrastThreshold=C.SIFT_CONTRAST)
    kp, des = sift.detectAndCompute(gray_proc, None)
    if des is None or len(kp) == 0:
        print("  [DEBUG] ⚠ NESSUN keypoint estratto!")
        return None
    sorted_idx = sorted(range(len(kp)), key=lambda i: kp[i].response, reverse=True)
    keep, kept_pos = [], []
    for i in sorted_idx:
        pt = kp[i].pt
        if all(np.hypot(pt[0] - p[0], pt[1] - p[1]) >= C.SIFT_NMS_DIST for p in kept_pos):
            keep.append(i)
            kept_pos.append(pt)
    return des[keep] if keep else None


def sift_match_scores(des_query, bf_matcher, sift_cache):
    """Per-class match fraction from ratio-test kNN matching against `sift_cache`."""
    raw_counts = {c: 0 for c in C.CLASSES}
    scores = {c: 0.0 for c in C.CLASSES}

    if des_query is None or len(des_query) < 3:
        return scores, raw_counts

    for cls, tmpl_des in sift_cache.items():
        if tmpl_des is None or len(tmpl_des) < 2:
            continue
        matches = bf_matcher.knnMatch(tmpl_des, des_query, k=2)
        good = 0
        for pair in matches:
            if pair is not None and len(pair) == 2:
                m, n = pair
                if m.distance < C.LOWE_RATIO * n.distance:
                    good += 1
        raw_counts[cls] = good

    total = sum(raw_counts.values())
    if total > 0:
        for c in C.CLASSES:
            scores[c] = raw_counts[c] / total

    return scores, raw_counts


def sift_confidence(raw_counts):
    """1 - (top2 / top1) match count: high when one class clearly dominates."""
    sorted_counts = sorted(raw_counts.values(), reverse=True)
    if len(sorted_counts) < 2 or sorted_counts[0] == 0:
        return 0.0
    top1, top2 = sorted_counts[0], sorted_counts[1]
    if top2 == 0:
        return 1.0
    return 1.0 - (top2 / top1)
