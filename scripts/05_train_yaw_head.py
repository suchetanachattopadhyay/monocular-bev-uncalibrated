#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [44, 45, 46, 47, 48, 49, 50]).
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

KITTI_3D_ROOT = "/kaggle/input/datasets/klemenko/kitti-dataset"
IMAGE_2_DIR = os.path.join(KITTI_3D_ROOT, "data_object_image_2", "training", "image_2")
LABEL_2_DIR = os.path.join(KITTI_3D_ROOT, "data_object_label_2", "training", "label_2")
 
CKPT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
 
VALID_CLASSES = {"Car", "Van"}   # skip Pedestrian/Cyclist/Truck/DontCare/Misc for this head
CROP_SIZE = 224
NUM_BINS = 2
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4
VAL_FRACTION = 0.15

#label parsing
 
def parse_label_file(label_path):
    """Returns a list of dicts, one per valid object in the image."""
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            if obj_type not in VALID_CLASSES:
                continue
            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])
            bbox = tuple(float(x) for x in parts[4:8])       # x1, y1, x2, y2
            dimensions = tuple(float(x) for x in parts[8:11])  # h, w, l
            location = tuple(float(x) for x in parts[11:14])   # x, y, z (camera frame)
            rotation_y = float(parts[14])
 
            # Basic quality filter -- skip heavily truncated/occluded boxes, which
            # give unreliable orientation supervision (standard KITTI practice)
            if truncated > 0.5 or occluded > 2:
                continue
            x1, y1, x2, y2 = bbox
            if (x2 - x1) < 10 or (y2 - y1) < 10:  # skip tiny/degenerate boxes
                continue
 
            objects.append({
                "type": obj_type, "alpha": alpha, "bbox": bbox,
                "dimensions": dimensions, "location": location, "rotation_y": rotation_y,
            })
    return objects

# Edge channel extraction (reused pattern)
 
def compute_edge_channels(rgb_crop):
    """Returns 3 separate single-channel maps (canny, sobel, emboss) -- NOT
    combined into one weighted score like the fusion pipeline's edge_conf.
    Here we want them as distinct input channels so the network can learn
    which one matters for orientation, rather than pre-committing to fixed
    weights the way Section 3.3 does for the BEV fusion task."""
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
 
 
def crop_and_stack(rgb_full, bbox):
    """Crops the bbox region, resizes to CROP_SIZE, returns a 6-channel
    tensor: [R, G, B, canny, sobel, emboss], each in [0,1]."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h_img, w_img = rgb_full.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    crop = rgb_full[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
 
    canny, sobel, emboss = compute_edge_channels(crop)
    rgb_norm = crop.astype(np.float32) / 255.0  # (H,W,3), BGR order from cv2
 
    stacked = np.dstack([rgb_norm, canny, sobel, emboss])  # (H,W,6)
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)  # (6,H,W)

 
# MultiBin target encoding 
 
def angle_to_bin_target(alpha, num_bins=NUM_BINS):
    """Simplified single-target-bin MultiBin encoding (a practical simplification
    of Mousavian et al.'s overlapping-bin scheme): assign the bin whose center is
    angularly closest to alpha, and compute the residual (cos, sin) offset from
    that bin's center for the regression loss. bin_id is the classification target."""
    bin_centers = np.linspace(-np.pi, np.pi, num_bins, endpoint=False)  # e.g. [-pi, 0] for num_bins=2
    diffs = np.abs(np.angle(np.exp(1j * (alpha - bin_centers))))  # wrapped angular distance
    bin_id = int(np.argmin(diffs))
    residual = alpha - bin_centers[bin_id]
    residual = np.angle(np.exp(1j * residual))  # wrap to [-pi, pi]
    return bin_id, np.cos(residual), np.sin(residual)
 
 
def decode_bin_prediction(bin_logits, cossin, num_bins=NUM_BINS):
    """Inverse of angle_to_bin_target, for use at inference time."""
    bin_centers = np.linspace(-np.pi, np.pi, num_bins, endpoint=False)
    bin_id = int(torch.argmax(bin_logits).item())
    cos_r, sin_r = cossin[bin_id]
    residual = float(torch.atan2(sin_r, cos_r))
    alpha = bin_centers[bin_id] + residual
    return float(np.angle(np.exp(1j * alpha)))  # wrap to [-pi, pi]

# Dataset 
 
class KittiYawDataset(Dataset):
    def __init__(self, image_dir=IMAGE_2_DIR, label_dir=LABEL_2_DIR):
        self.samples = []  # list of (image_path, object_dict)
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        print(f"Scanning {len(label_paths)} label files...")
        for label_path in label_paths:
            frame_id = os.path.splitext(os.path.basename(label_path))[0]
            image_path = os.path.join(image_dir, f"{frame_id}.png")
            if not os.path.exists(image_path):
                continue
            objects = parse_label_file(label_path)
            for obj in objects:
                self.samples.append((image_path, obj))
        print(f"Built dataset with {len(self.samples)} valid Car/Van instances.")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"0 samples found -- check IMAGE_2_DIR/LABEL_2_DIR paths against "
                f"the actual dataset structure."
            )
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        image_path, obj = self.samples[idx]
        rgb_full = cv2.imread(image_path)
        crop_tensor = crop_and_stack(rgb_full, obj["bbox"])
        if crop_tensor is None:
            # fall back to a neighboring sample if the crop is degenerate
            return self.__getitem__((idx + 1) % len(self.samples))
 
        bin_id, cos_r, sin_r = angle_to_bin_target(obj["alpha"])
        return (
            torch.from_numpy(crop_tensor),
            torch.tensor(bin_id, dtype=torch.long),
            torch.tensor([cos_r, sin_r], dtype=torch.float32),
        )
 
 
def get_dataloaders():
    full_ds = KittiYawDataset()
    n_val = max(1, int(len(full_ds) * VAL_FRACTION))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Yaw dataset: {len(full_ds)} total | {n_train} train | {n_val} held-out val")
    return train_loader, val_loader

# Model: MultiBin yaw head 
 
class YawHead(nn.Module):
    """ResNet18 backbone, first conv modified to accept 6 input channels
    (RGB + canny + sobel + emboss) instead of the usual 3."""
 
    def __init__(self, num_bins=NUM_BINS, in_channels=6):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
 
        # Expand first conv layer from 3->6 input channels, keeping pretrained
        # RGB weights and initializing the 3 new edge channels as a copy of the
        # (per-output-channel) mean of the RGB weights -- a standard, cheap way
        # to extend a pretrained conv without discarding the RGB prior.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                              stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = mean_weight.repeat(1, in_channels - 3, 1, 1)
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
        cossin = F.normalize(cossin, dim=-1)  # project onto unit circle, standard MultiBin practice
        return conf, cossin

#train
 
def train_yaw_head(train_loader, val_loader):
    model = YawHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
 
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for crops, bin_ids, cossin_targets in train_loader:
            crops, bin_ids, cossin_targets = crops.to(DEVICE), bin_ids.to(DEVICE), cossin_targets.to(DEVICE)
            conf, cossin_pred = model(crops)
 
            cls_loss = ce_loss(conf, bin_ids)
            # regression loss only on the TARGET bin's cos/sin prediction
            pred_for_target_bin = cossin_pred[torch.arange(len(bin_ids)), bin_ids]
            reg_loss = F.smooth_l1_loss(pred_for_target_bin, cossin_targets)
            loss = cls_loss + reg_loss
 
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
 
        # Validation: bin classification accuracy + mean angular error (degrees)
        model.eval()
        correct, total, angle_errs = 0, 0, []
        with torch.no_grad():
            for crops, bin_ids, cossin_targets in val_loader:
                crops, bin_ids, cossin_targets = crops.to(DEVICE), bin_ids.to(DEVICE), cossin_targets.to(DEVICE)
                conf, cossin_pred = model(crops)
                pred_bins = conf.argmax(dim=1)
                correct += (pred_bins == bin_ids).sum().item()
                total += len(bin_ids)
 
                for i in range(len(bin_ids)):
                    pred_alpha = decode_bin_prediction(conf[i].cpu(), cossin_pred[i].cpu())
                    true_res = cossin_targets[i].cpu()
                    true_alpha_res = float(torch.atan2(true_res[1], true_res[0]))
                    bin_centers = np.linspace(-np.pi, np.pi, NUM_BINS, endpoint=False)
                    true_alpha = float(np.angle(np.exp(1j * (bin_centers[bin_ids[i].item()] + true_alpha_res))))
                    err = abs(np.angle(np.exp(1j * (pred_alpha - true_alpha))))
                    angle_errs.append(np.degrees(err))
 
        acc = correct / total
        mean_err_deg = float(np.mean(angle_errs))
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
 
        print(f"epoch {epoch+1}/{EPOCHS}  train_loss={total_loss/len(train_loader):.4f}  "
              f"val_bin_acc={acc:.4f}  val_mean_angle_err={mean_err_deg:.1f}deg  (best_acc={best_acc:.4f})")
 
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "yaw_head.pt"))
    print(f"Saved best yaw head (val_bin_acc={best_acc:.4f}) to {CKPT_DIR}/yaw_head.pt")
    return model
 
 
def main():
    train_loader, val_loader = get_dataloaders()
    train_yaw_head(train_loader, val_loader)
 
 
if __name__ == "__main__":
    main()
