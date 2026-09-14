#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [67]).
# Logic is unchanged from the run that produced the reported results.
# Paths below are the original Kaggle absolute paths; see README "Reproducing".

import os
import re
import glob
import json
import csv
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Extracts individual vehicle crops from your 40 pose-integrated images so you can
hand-label coarse facing direction on a small subset (8-10 is enough). Saves
numbered crop images + a CSV template with a blank 'facing_label' column for
you to fill in.

Run AFTER pose_integrated_pipeline.py (needs all_pose_records.json to know
which boxes exist).
"""

import os
import json
import csv
import shutil
import cv2

POSE_DIR = "/kaggle/working/pose_integrated"
MAPILLARY_DIR = "/kaggle/input/datasets/coconotchanel/mapillary-subset-usage"
RECORDS_PATH = os.path.join(POSE_DIR, "all_pose_records.json")
CROPS_OUT_DIR = "/kaggle/working/crops_for_labeling"
LABELS_CSV_PATH = "/kaggle/working/crops_for_labeling/labels_template.csv"  # <-- now saved INSIDE the crops folder, so it's bundled in the zip automatically
ZIP_PATH = "/kaggle/working/crops_for_labeling_bundle"  # shutil appends .zip

os.makedirs(CROPS_OUT_DIR, exist_ok=True)

# Coarse label -> target alpha (radians), matching the KITTI convention used in training:
#   0 = facing away (same direction as camera), pi = facing toward camera,
#   +-pi/2 = perpendicular (crossing left/right)
LABEL_TO_ALPHA_DEG = {
    "away": 0, "toward": 180, "left": 90, "right": -90,
}


def main():
    with open(RECORDS_PATH, "r") as f:
        all_records = json.load(f)

    rows = []
    saved = 0
    for name, records in all_records.items():
        img_path = os.path.join(MAPILLARY_DIR, f"{name}.jpg")
        if not os.path.exists(img_path) or len(records) == 0:
            continue
        rgb = cv2.imread(img_path)

        for i, r in enumerate(records):
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            crop = rgb[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crop_filename = f"{name}_veh{i}.jpg"
            cv2.imwrite(os.path.join(CROPS_OUT_DIR, crop_filename), crop)

            rows.append({
                "crop_filename": crop_filename,
                "image_name": name,
                "vehicle_idx": i,
                "predicted_global_yaw_deg": round(r["global_yaw_deg"], 1),
                "facing_label": "",   # <-- YOU FILL THIS IN: away / toward / left / right
            })
            saved += 1

    with open(LABELS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_filename", "image_name", "vehicle_idx",
                                                 "predicted_global_yaw_deg", "facing_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {saved} vehicle crops to {CROPS_OUT_DIR}")
    print(f"Saved label template ({len(rows)} rows) to {LABELS_CSV_PATH}")

    shutil.make_archive(ZIP_PATH, "zip", CROPS_OUT_DIR)
    print(f"\nZipped everything to {ZIP_PATH}.zip")
    print("Find it in the 'Output' tab (right sidebar) and download it, then upload "
          "that zip to Claude -- it'll walk through each crop with you and fill in "
          "the CSV based on what you say, so you don't have to touch the CSV yourself.")


if __name__ == "__main__":
    main()
