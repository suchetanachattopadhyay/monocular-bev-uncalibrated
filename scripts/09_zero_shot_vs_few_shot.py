#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [68]).
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
Loads your hand-filled labels_template.csv (after you've labeled 8-10 rows),
splits those labeled rows into a tiny train/test split, then compares:
  - ZERO-SHOT: the KITTI-trained yaw head, unmodified, evaluated on the test rows
  - FEW-SHOT: the same head with only the last layers fine-tuned on the train
    rows, evaluated on the same test rows

Both are scored against YOUR hand labels (converted to target alpha), not the
model's own prior prediction -- this is the actual accuracy check the
zero-shot-only version couldn't give you.

Run AFTER extract_crops_for_labeling.py and after you've filled in at least
8 'facing_label' values in labels_template.csv.
"""

import os
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

CROPS_DIR = "/kaggle/working/crops_for_labeling"
LABELS_CSV_PATH = "/kaggle/working/crops_for_labeling/labels_template.csv"
CKPT_DIR = "/kaggle/working/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CROP_SIZE = 224
NUM_BINS = 2
FEW_SHOT_EPOCHS = 30
FEW_SHOT_LR = 1e-4

LABEL_TO_ALPHA_DEG = {"away": 0, "toward": 180, "left": 90, "right": -90}


# ----------------------------- Reused model + feature code ---------------------

def compute_edge_channels(rgb_crop):
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_norm = sobel_mag / (sobel_mag.max() + 1e-6)
    emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    emboss = cv2.filter2D(gray.astype(np.float32), -1, emboss_kernel)
    emboss_norm = np.abs(emboss) / (np.abs(emboss).max() + 1e-6)
    return canny, sobel_norm, emboss_norm


def load_crop_tensor(crop_path):
    crop = cv2.imread(crop_path)
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    canny, sobel, emboss = compute_edge_channels(crop)
    rgb_norm = crop.astype(np.float32) / 255.0
    stacked = np.dstack([rgb_norm, canny, sobel, emboss])
    return torch.from_numpy(np.transpose(stacked, (2, 0, 1)).astype(np.float32))


class YawHead(nn.Module):
    def __init__(self, num_bins=NUM_BINS, in_channels=6):
        super().__init__()
        backbone = models.resnet18(weights=None)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                              stride=old_conv.stride, padding=old_conv.padding, bias=False)
        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.bin_conf = nn.Linear(512, num_bins)
        self.bin_cossin = nn.Linear(512, num_bins * 2)
        self.num_bins = num_bins

    def forward(self, x):
        feat = self.backbone(x)
        conf = self.bin_conf(feat)
        cossin = self.bin_cossin(feat).view(-1, self.num_bins, 2)
        cossin = F.normalize(cossin, dim=-1)
        return conf, cossin


def angle_to_bin_target(alpha, num_bins=NUM_BINS):
    bin_centers = np.linspace(-np.pi, np.pi, num_bins, endpoint=False)
    diffs = np.abs(np.angle(np.exp(1j * (alpha - bin_centers))))
    bin_id = int(np.argmin(diffs))
    residual = np.angle(np.exp(1j * (alpha - bin_centers[bin_id])))
    return bin_id, np.cos(residual), np.sin(residual)


def decode_bin_prediction(bin_logits, cossin, num_bins=NUM_BINS):
    bin_centers = np.linspace(-np.pi, np.pi, num_bins, endpoint=False)
    bin_id = int(torch.argmax(bin_logits).item())
    cos_r, sin_r = cossin[bin_id]
    residual = float(torch.atan2(sin_r, cos_r))
    alpha = bin_centers[bin_id] + residual
    return float(np.angle(np.exp(1j * alpha)))


def load_pretrained_yaw_head():
    model = YawHead().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "yaw_head.pt"), map_location=DEVICE))
    return model


# ----------------------------- Load labeled rows -------------------------------

def load_labeled_rows():
    labeled = []
    with open(LABELS_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["facing_label"].strip().lower()
            if label in LABEL_TO_ALPHA_DEG:
                labeled.append({
                    "crop_path": os.path.join(CROPS_DIR, row["crop_filename"]),
                    "alpha_rad": np.radians(LABEL_TO_ALPHA_DEG[label]),
                    "crop_filename": row["crop_filename"],
                })
    return labeled


# ----------------------------- Evaluation --------------------------------------

@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    errs = []
    for row in rows:
        x = load_crop_tensor(row["crop_path"]).unsqueeze(0).to(DEVICE)
        conf, cossin = model(x)
        pred_alpha = decode_bin_prediction(conf[0].cpu(), cossin[0].cpu())
        err = abs(np.angle(np.exp(1j * (pred_alpha - row["alpha_rad"]))))
        errs.append(np.degrees(err))
    return errs


def finetune_last_layers(model, train_rows, epochs=FEW_SHOT_EPOCHS, lr=FEW_SHOT_LR):
    """Freezes the backbone, only trains bin_conf + bin_cossin (the final heads) --
    appropriate for an 8-10 sample dataset, where fine-tuning the whole ResNet
    would just overfit instantly."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    params_to_train = list(model.bin_conf.parameters()) + list(model.bin_cossin.parameters())
    opt = torch.optim.Adam(params_to_train, lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    crop_tensors = torch.stack([load_crop_tensor(r["crop_path"]) for r in train_rows]).to(DEVICE)
    bin_targets, cossin_targets = [], []
    for r in train_rows:
        bin_id, cos_r, sin_r = angle_to_bin_target(r["alpha_rad"])
        bin_targets.append(bin_id)
        cossin_targets.append([cos_r, sin_r])
    bin_targets = torch.tensor(bin_targets, dtype=torch.long).to(DEVICE)
    cossin_targets = torch.tensor(cossin_targets, dtype=torch.float32).to(DEVICE)

    model.train()
    for epoch in range(epochs):
        conf, cossin_pred = model(crop_tensors)
        cls_loss = ce_loss(conf, bin_targets)
        pred_for_target = cossin_pred[torch.arange(len(bin_targets)), bin_targets]
        reg_loss = F.smooth_l1_loss(pred_for_target, cossin_targets)
        loss = cls_loss + reg_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"  fine-tune epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    return model


def main():
    labeled = load_labeled_rows()
    print(f"Found {len(labeled)} hand-labeled rows.")
    if len(labeled) < 6:
        print("Need at least ~6-8 labeled rows for a meaningful train/test split. "
              "Label more rows in labels_template.csv and re-run.")
        return

    np.random.seed(42)
    np.random.shuffle(labeled)
    n_test = max(2, len(labeled) // 3)
    test_rows, train_rows = labeled[:n_test], labeled[n_test:]
    print(f"Split: {len(train_rows)} train (for few-shot fine-tuning) / {len(test_rows)} test (evaluation)")

    print("\n=== ZERO-SHOT (no fine-tuning) ===")
    zero_shot_model = load_pretrained_yaw_head()
    zero_shot_errs = evaluate(zero_shot_model, test_rows)
    print(f"Mean angular error on test set: {np.mean(zero_shot_errs):.1f} deg  "
          f"(per-sample: {[round(e,1) for e in zero_shot_errs]})")

    print("\n=== FEW-SHOT (fine-tuned on train rows) ===")
    few_shot_model = load_pretrained_yaw_head()
    few_shot_model = finetune_last_layers(few_shot_model, train_rows)
    few_shot_errs = evaluate(few_shot_model, test_rows)
    print(f"Mean angular error on test set: {np.mean(few_shot_errs):.1f} deg  "
          f"(per-sample: {[round(e,1) for e in few_shot_errs]})")

    print("\n" + "=" * 50)
    print(f"ZERO-SHOT mean error: {np.mean(zero_shot_errs):.1f} deg")
    print(f"FEW-SHOT  mean error: {np.mean(few_shot_errs):.1f} deg")
    print("=" * 50)
    print("\nNote: n is tiny here (a handful of hand-labeled samples), so treat "
          "this as a directional signal, not a statistically robust claim -- "
          "say so plainly if this goes in the paper.")


if __name__ == "__main__":
    main()
