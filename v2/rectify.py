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
