# Render GT circles on every target (green=sure, yellow=doubt, red=unknown)
# plus a contact sheet of doubtful crops for fast human review.
import cv2, json, os
import numpy as np
from v2.paths import PROJECT, TARGET_DIR, target_images

OUT = os.path.join(PROJECT, "v2", "gt_overlays")
COLOR = {"sure": (0, 200, 0), "doubt": (0, 215, 255), "unknown": (0, 0, 255)}

def state(c):
    if c["label"] == "unknown": return "unknown"
    return "sure" if c["sure"] else "doubt"

def main(gt_path=None):
    os.makedirs(OUT, exist_ok=True)
    with open(gt_path or os.path.join(PROJECT, "v2", "gt_pos.json")) as f:
        gt = json.load(f)
    crops = []
    for name in target_images():
        if name not in gt:
            continue
        img = cv2.imread(os.path.join(TARGET_DIR, name))
        if img is None:
            raise FileNotFoundError(name)
        for k, c in enumerate(gt[name]["coins"]):
            st = state(c)
            cv2.circle(img, (int(c["cx"]), int(c["cy"])), int(c["r"]), COLOR[st], 3)
            cv2.putText(img, c["label"], (int(c["cx"] - c["r"]), int(c["cy"] - c["r"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR[st], 2)
            if st != "sure":
                x0 = max(0, int(c["cx"] - 1.2 * c["r"])); y0 = max(0, int(c["cy"] - 1.2 * c["r"]))
                x1 = min(img.shape[1], int(c["cx"] + 1.2 * c["r"]))
                y1 = min(img.shape[0], int(c["cy"] + 1.2 * c["r"]))
                crop = cv2.imread(os.path.join(TARGET_DIR, name))[y0:y1, x0:x1]
                crop = cv2.resize(crop, (180, 180))
                cv2.putText(crop, f"{name.replace('image_','').replace('.jpg','')}:{k} {c['label']}",
                            (3, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR[st], 1)
                crops.append(crop)
        cv2.imwrite(os.path.join(OUT, name), img)
    if crops:
        cols = 8
        rows = (len(crops) + cols - 1) // cols
        sheet = np.zeros((rows * 180, cols * 180, 3), np.uint8)
        for k, cr in enumerate(crops):
            r0, c0 = divmod(k, cols)
            sheet[r0 * 180:(r0 + 1) * 180, c0 * 180:(c0 + 1) * 180] = cr
        cv2.imwrite(os.path.join(OUT, "_doubts_sheet.jpg"), sheet)
    print(f"overlays={sum(1 for n in target_images() if n in gt)} doubts={len(crops)}")

if __name__ == "__main__":
    main()
