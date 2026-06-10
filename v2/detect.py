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
