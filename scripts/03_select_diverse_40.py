#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [24]).
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
Re-runs the paper's Section 5.1 diversity-sampling protocol: embed all images
with CLIP ViT-B/32, k-means cluster into 40 groups, select the image closest
to each cluster centroid. Produces the list of 40 filenames to use for both
inference and VLM re-scoring, so the comparison matches your already-reported
Section 5.2 results.

pip install: git+https://github.com/openai/CLIP.git scikit-learn torch pillow
"""

import os
import glob
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans

MAPILLARY_DIR = "/kaggle/input/datasets/coconotchanel/mapillary-subset-usage"
N_CLUSTERS = 40
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_all_images():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    paths = sorted(glob.glob(os.path.join(MAPILLARY_DIR, "*.jpg")) +
                    glob.glob(os.path.join(MAPILLARY_DIR, "*.png")))
    print(f"Found {len(paths)} images to embed.")

    embeddings = []
    valid_paths = []
    with torch.no_grad():
        for p in paths:
            try:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize, standard for CLIP
                embeddings.append(feat.cpu().numpy().squeeze())
                valid_paths.append(p)
            except Exception as e:
                print(f"skip {p}: {e}")

    return np.stack(embeddings), valid_paths


def select_diverse_40(embeddings, paths, n_clusters=N_CLUSTERS, seed=SEED):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(embeddings)

    selected_paths = []
    for c in range(n_clusters):
        cluster_mask = cluster_ids == c
        if cluster_mask.sum() == 0:
            continue
        cluster_embeds = embeddings[cluster_mask]
        cluster_paths = [p for p, m in zip(paths, cluster_mask) if m]
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
        closest_idx = np.argmin(dists)
        selected_paths.append(cluster_paths[closest_idx])

    return selected_paths


def main():
    embeddings, paths = embed_all_images()
    selected = select_diverse_40(embeddings, paths)
    print(f"\nSelected {len(selected)} diverse images (target was {N_CLUSTERS}):")
    for p in selected:
        print(" ", os.path.basename(p))

    # Save the filename list so the inference script can filter to just these 40
    out_path = "/kaggle/working/selected_40_filenames.txt"
    with open(out_path, "w") as f:
        for p in selected:
            f.write(os.path.basename(p) + "\n")
    print(f"\nSaved filename list to {out_path}")
    return selected


if __name__ == "__main__":
    selected_filenames = main()
