"""The SIFT matching cue, fixed from tiziano's version (see ANALYSIS.md S2):

1. The reference cache is now built from the same canonical 150x150 crop_disc
   output the shape cue uses, not the raw, uncropped reference photo. This
   removes the train/query domain mismatch AND the physical-size bias it
   caused (a photo's raw pixel scale used to make bimetal coins occupy more
   image area than small ones, so `SIFT_NMS_DIST` -- an absolute pixel radius
   -- let them keep proportionally more keypoints). Every class's canonical
   crop is exactly the same size, so `SIFT_NMS_DIST` no longer favours large
   coins.
2. `sift_match_scores` now normalises each class's match count by that
   class's OWN template descriptor count, not the total across all classes.
   tiziano's `raw_counts[c] / total` meant a class with more cached
   descriptors structurally won a bigger share of the match budget regardless
   of true visual similarity; this makes the score a same-scale rate (0-1)
   per class, independent of how many descriptors any other class has.
"""
import cv2
import numpy as np

from . import config as C


def preprocess_for_sift(gray):
    """CLAHE + bilateral filter to enhance edges before SIFT keypoint detection."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)


def _nms_keypoints(kp, des):
    if des is None or len(kp) == 0:
        return None
    sorted_idx = sorted(range(len(kp)), key=lambda i: kp[i].response, reverse=True)
    keep, kept_pos = [], []
    for i in sorted_idx:
        pt = kp[i].pt
        if all(np.hypot(pt[0] - p[0], pt[1] - p[1]) >= C.SIFT_NMS_DIST for p in kept_pos):
            keep.append(i)
            kept_pos.append(pt)
    return des[keep] if keep else None


def calibrate_sift_templates(template_crops):
    """Builds {class_name: descriptor array} from the reference model's canonical crops
    (the same 150x150 discs the shape atlas is built from, one per class, BGR or gray).

    This is the SIFT half of the reference model (paired with the shape atlas built in
    reference.py): a one-off computation over the 8 reference images, reused for every
    target image.
    """
    sift_cache = {}
    sift = cv2.SIFT_create(nfeatures=C.SIFT_NFEATURES, contrastThreshold=C.SIFT_CONTRAST)

    for cls, crop in template_crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = preprocess_for_sift(gray)
        kp, des = sift.detectAndCompute(gray, None)
        sift_cache[cls] = _nms_keypoints(kp, des)

    return sift_cache


def sift_extract_with_nms(gray_roi):
    """Extracts SIFT descriptors from a target crop and applies spatial NMS."""
    gray_proc = preprocess_for_sift(gray_roi)
    sift = cv2.SIFT_create(nfeatures=C.SIFT_NFEATURES, contrastThreshold=C.SIFT_CONTRAST)
    kp, des = sift.detectAndCompute(gray_proc, None)
    return _nms_keypoints(kp, des)


def sift_match_scores(des_query, bf_matcher, sift_cache):
    """Per-class match RATE (good matches / that class's own template descriptor count)
    from ratio-test kNN matching against `sift_cache`. Unlike tiziano's version, this does
    not divide by the total match count across classes, so a class with few cached
    descriptors isn't structurally penalised relative to one with many."""
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
        scores[cls] = good / len(tmpl_des)

    return scores, raw_counts


def sift_confidence(scores):
    """Normalised top1-vs-top2 gap: high only when one class's match rate clearly
    dominates the runner-up."""
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) < 2 or sorted_scores[0] == 0:
        return 0.0
    top1, top2 = sorted_scores[0], sorted_scores[1]
    return float(np.clip((top1 - top2) / (top1 + 1e-6), 0, 1))
