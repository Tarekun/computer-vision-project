# Project-relative paths and the frozen dev/sealed split (even=dev, odd=sealed).
import os, re

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(PROJECT, "coin_dataset", "target_set")
REF_DIR = os.path.join(PROJECT, "coin_dataset", "reference_set")
PY = "/home/gianmarco/envs/IP&CV/bin/python"
CLASSES = ["1cent","2cent","5cent","10cent","20cent","50cent","1euro","2euro"]
VALUES = {"1cent":0.01,"2cent":0.02,"5cent":0.05,"10cent":0.10,
          "20cent":0.20,"50cent":0.50,"1euro":1.00,"2euro":2.00}
FAMILY = {**{c:"copper" for c in ["1cent","2cent","5cent"]},
          **{c:"gold" for c in ["10cent","20cent","50cent"]},
          **{c:"bimetal" for c in ["1euro","2euro"]}}
# Real euro diameters (mm) — physical constants, see spec §1 (Q1 may replace with ref-derived grid)
DIAM_MM = {"1cent":16.25,"2cent":18.75,"10cent":19.75,"5cent":21.25,
           "20cent":22.25,"1euro":23.25,"50cent":24.25,"2euro":25.75}

def target_images():
    return sorted(os.listdir(TARGET_DIR), key=lambda n: image_num(n))

def image_num(name):
    return int(re.search(r"image_(\d+)\.jpg", name).group(1))

def is_dev(name):
    return image_num(name) % 2 == 0
