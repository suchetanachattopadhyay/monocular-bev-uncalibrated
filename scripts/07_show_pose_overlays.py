#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [63]).
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
Displays the pose-integrated BEV overlays inline in the notebook (instead of
digging through the file browser), alongside the original photo and the
readable pose records for that image. Run this AFTER pose_integrated_pipeline.py
has finished and produced files in /kaggle/working/pose_integrated/.

No pip installs needed -- matplotlib ships with Kaggle notebooks by default.
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

POSE_DIR = "/kaggle/working/pose_integrated"
MAPILLARY_DIR = "/kaggle/input/datasets/coconotchanel/mapillary-subset-usage"
RECORDS_PATH = os.path.join(POSE_DIR, "all_pose_records.json")

N_TO_SHOW = 5   # how many images to display -- bump this up once the first few look right


def main():
    if not os.path.exists(RECORDS_PATH):
        print(f"'{RECORDS_PATH}' not found -- did pose_integrated_pipeline.py finish? "
              f"Check for a 'Done.' message in that script's output before running this.")
        return

    with open(RECORDS_PATH, "r") as f:
        all_records = json.load(f)

    overlay_paths = sorted(glob.glob(os.path.join(POSE_DIR, "*_pose_overlay.png")))
    print(f"Found {len(overlay_paths)} pose overlays out of an expected 40.")
    if len(overlay_paths) == 0:
        print("0 overlays found -- the pipeline likely didn't complete. Re-check its output for errors.")
        return

    for overlay_path in overlay_paths[:N_TO_SHOW]:
        name = os.path.basename(overlay_path).replace("_pose_overlay.png", "")
        original_path = os.path.join(MAPILLARY_DIR, f"{name}.jpg")

        records = all_records.get(name, [])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        if os.path.exists(original_path):
            axes[0].imshow(mpimg.imread(original_path))
        axes[0].set_title(f"Original: {name}")
        axes[0].axis("off")

        overlay_img = mpimg.imread(overlay_path)
        axes[1].imshow(overlay_img)
        axes[1].set_title(f"BEV occupancy + oriented boxes ({len(records)} vehicles)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        print(f"--- Pose records for {name} ---")
        if len(records) == 0:
            print("  (no vehicles with valid pose in this image)")
        for i, r in enumerate(records):
            print(f"  vehicle {i}: conf={r['conf']:.2f}  Z={r['Z']:.1f}m  X={r['X']:.1f}m  "
                  f"alpha={r['alpha_deg']:.1f}deg  theta_ray={r['theta_ray_deg']:.1f}deg  "
                  f"global_yaw={r['global_yaw_deg']:.1f}deg")
        print()


if __name__ == "__main__":
    main()
