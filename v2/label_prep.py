# v2/label_prep.py
# Render numbered baseline seed circles on each target image for labeling agents.
import cv2, json, os
from v2.paths import PROJECT, TARGET_DIR, target_images

OUT = os.path.join(PROJECT, "v2", "label_prep")

def main():
    os.makedirs(OUT, exist_ok=True)
    base = json.load(open(os.path.join(PROJECT, "fable", "baseline", "pred.json")))
    seeds = {}
    for name in target_images():
        img = cv2.imread(os.path.join(TARGET_DIR, name))
        ss = [{"i": k, "cx": int(c["cx"]), "cy": int(c["cy"]), "r": int(c["r"])}
              for k, c in enumerate(base.get(name, []))]
        seeds[name] = ss
        for s in ss:
            cv2.circle(img, (s["cx"], s["cy"]), s["r"], (0, 255, 0), 3)
            cv2.putText(img, str(s["i"]), (s["cx"] - 15, s["cy"] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        cv2.imwrite(os.path.join(OUT, name), img)
    json.dump(seeds, open(os.path.join(OUT, "seeds.json"), "w"), indent=1)
    print("done", len(seeds))

if __name__ == "__main__":
    main()
