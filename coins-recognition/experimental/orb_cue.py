"""ORB alternative to the SIFT cue (EXPERIMENTS.md item 5): a cheaper binary
descriptor, built with the same domain-correct design sift_cue.py already uses --
the reference cache comes from the canonical 150x150 crop (not a raw photo), and
each class's match count is normalised by that class's OWN cached descriptor count,
not the total across classes. Only used by orb_compare.py to measure whether ORB is
a viable, cheaper substitute for SIFT now that the domain mismatch is fixed on both
sides -- not wired into classify.py unless EXPERIMENTS.md's measurement says to.

ORB descriptors are binary (not float, like SIFT's), so matching needs a Hamming-
distance BFMatcher, not the L2 one sift_cue.py uses.
"""
import cv2
import numpy as np

from . import config as C


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


def calibrate_orb_templates(template_crops):
    """Builds {class_name: descriptor array} from the reference model's canonical
    crops -- the ORB half of an alternative reference model, mirroring
    sift_cue.calibrate_sift_templates."""
    orb = cv2.ORB_create(nfeatures=C.SIFT_NFEATURES)
    cache = {}
    for cls, crop in template_crops.items():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        kp, des = orb.detectAndCompute(gray, None)
        cache[cls] = _nms_keypoints(kp, des)
    return cache


def orb_extract_with_nms(gray_roi):
    orb = cv2.ORB_create(nfeatures=C.SIFT_NFEATURES)
    kp, des = orb.detectAndCompute(gray_roi, None)
    return _nms_keypoints(kp, des)


def orb_match_scores(des_query, bf_matcher, orb_cache):
    """Per-class match RATE (good matches / that class's own template descriptor
    count) via Hamming-distance ratio-test kNN matching -- mirrors
    sift_cue.sift_match_scores exactly, just with a Hamming matcher for binary
    descriptors."""
    raw_counts = {c: 0 for c in C.CLASSES}
    scores = {c: 0.0 for c in C.CLASSES}

    if des_query is None or len(des_query) < 3:
        return scores, raw_counts

    for cls, tmpl_des in orb_cache.items():
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
