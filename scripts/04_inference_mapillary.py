#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [33, 29, 30, 31, 32, 34]).
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

MAPILLARY_DIR = "/kaggle/input/datasets/coconotchanel/mapillary-subset-usage"
OUT_DIR = "/kaggle/working/inference_compare"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_DIR = "/kaggle/working/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MODEL = "mlp"

GRID_RANGE_FWD = 40.0
GRID_RANGE_LAT = 20.0
CELL_SIZE = 0.2
GRID_H = int(GRID_RANGE_FWD / CELL_SIZE)
GRID_W = int(2 * GRID_RANGE_LAT / CELL_SIZE)

ASSUMED_CAMERA_HEIGHT = 1.5
GROUND_BAND_ROWS = (0.75, 0.95)
YOLO_CONF_THRESH = 0.35
YOLO_CLASSES = [2, 3, 5, 7]
EDGE_WEIGHTS = (0.4, 0.4, 0.2)
FIXED_WEIGHTS = (0.5, 0.2, 0.3)

SELECTED_FILENAMES_PATH = "/kaggle/working/selected_40_filenames.txt"

#MiDaS depth + ground-plane calibration 
 
_midas_model, _midas_transform = None, None
 
def get_midas():
    global _midas_model, _midas_transform
    if _midas_model is None:
        _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _midas_transform = transforms.small_transform
    return _midas_model, _midas_transform
 
 
@torch.no_grad()
def estimate_relative_depth(rgb_bgr):
    """Returns MiDaS relative inverse depth (higher value = closer), same
    resolution as input rgb_bgr."""
    model, transform = get_midas()
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(DEVICE)
    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=rgb_bgr.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()
    return prediction.cpu().numpy()  # (H, W), higher = closer (inverse-depth-like)
 
 
def default_intrinsics(img_w, img_h):
    """No calibration file for arbitrary Mapillary images -- assume a
    typical phone/dashcam FOV. This is a heuristic, same spirit as the
    paper's other stated assumptions (flag it as such if this goes in
    the qualitative-results writeup)."""
    fx = fy = 0.85 * img_w
    cx, cy = img_w / 2.0, img_h / 2.0
    return fx, fy, cx, cy
 
 
def ground_plane_calibrate(rel_depth, fx, fy, cx, cy, camera_height=ASSUMED_CAMERA_HEIGHT,
                            band=GROUND_BAND_ROWS):
    """Section 3.2: assume the bottom band of the frame is flat ground at a
    fixed camera height, solve the scale that makes MiDaS's relative depth
    match that assumption, then convert to metric depth everywhere.
 
    Ground-plane ray intersection (forward-facing, no pitch):
        Z_ground(v) = camera_height * fy / (v - cy),  for v > cy
    Metric depth from MiDaS output (treated as ~ 1/depth up to scale s):
        Z_metric(u,v) = s / rel_depth(u,v)
    Solve s by matching Z_metric to Z_ground over the assumed-ground band.
    """
    H, W = rel_depth.shape
    row_lo, row_hi = int(band[0] * H), int(band[1] * H)
    rows = np.arange(row_lo, row_hi)
    valid_rows = rows[rows > cy]  # ray-ground intersection undefined at/above horizon
    if len(valid_rows) == 0:
        scale = 1.0
    else:
        z_ground = camera_height * fy / (valid_rows - cy)               # (n_rows,)
        band_rel_depth = rel_depth[valid_rows, :].mean(axis=1)          # avg across row
        band_rel_depth = np.clip(band_rel_depth, 1e-4, None)
        scale = float(np.median(z_ground * band_rel_depth))
 
    metric_depth = scale / np.clip(rel_depth, 1e-4, None)
    metric_depth = np.clip(metric_depth, 0.5, 80.0)  # discard implausible extremes near horizon
    return metric_depth.astype(np.float32)

#Reused feature functions (same as KITTI script)
def backproject_to_bev(depth_m, fx, fy, cx, cy):
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth_m > 0
    Z = depth_m[valid]
    X = (us[valid] - cx) * Z / fx
    row = (Z / CELL_SIZE).astype(np.int32)
    col = ((X + GRID_RANGE_LAT) / CELL_SIZE).astype(np.int32)
    in_range = (row >= 0) & (row < GRID_H) & (col >= 0) & (col < GRID_W)
    row, col, Z = row[in_range], col[in_range], Z[in_range]
 
    count_grid = np.zeros((GRID_H, GRID_W), dtype=np.int32)
    depth_sum_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    depth_sq_sum_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    np.add.at(count_grid, (row, col), 1)
    np.add.at(depth_sum_grid, (row, col), Z)
    np.add.at(depth_sq_sum_grid, (row, col), Z ** 2)
    return count_grid, depth_sum_grid, depth_sq_sum_grid
 
 
def make_density_map(count_grid):
    if count_grid.max() == 0:
        return np.zeros_like(count_grid, dtype=np.float32)
    return (count_grid / count_grid.max()).astype(np.float32)
 
 
def make_depth_mean_var(count_grid, depth_sum_grid, depth_sq_sum_grid):
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count_grid > 0, depth_sum_grid / np.maximum(count_grid, 1), 0.0)
        var = np.where(count_grid > 0, depth_sq_sum_grid / np.maximum(count_grid, 1) - mean ** 2, 0.0)
    return mean.astype(np.float32), np.clip(var, 0.0, None).astype(np.float32)
 
 
def make_edge_confidence_map(rgb_img, weights=EDGE_WEIGHTS):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_norm = sobel_mag / (sobel_mag.max() + 1e-6)
    emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    emboss = cv2.filter2D(gray.astype(np.float32), -1, emboss_kernel)
    emboss_norm = np.abs(emboss) / (np.abs(emboss).max() + 1e-6)
    w_c, w_s, w_e = weights
    return w_c * canny + w_s * sobel_norm + w_e * emboss_norm
 
 
def project_image_map_to_bev(img_space_map, depth_m, fx, fy, cx, cy):
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth_m > 0
    Z = depth_m[valid]
    X = (us[valid] - cx) * Z / fx
    vals = img_space_map[valid]
    row = (Z / CELL_SIZE).astype(np.int32)
    col = ((X + GRID_RANGE_LAT) / CELL_SIZE).astype(np.int32)
    in_range = (row >= 0) & (row < GRID_H) & (col >= 0) & (col < GRID_W)
    row, col, vals = row[in_range], col[in_range], vals[in_range]
    sum_grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    count_grid = np.zeros((GRID_H, GRID_W), dtype=np.int32)
    np.add.at(sum_grid, (row, col), vals)
    np.add.at(count_grid, (row, col), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_grid = np.where(count_grid > 0, sum_grid / np.maximum(count_grid, 1), 0.0)
    return mean_grid.astype(np.float32)
 
 
_yolo_model = None
 
def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model
 
 
def make_yolo_conf_map(rgb_img, depth_m, fx, fy, cx, cy):
    model = get_yolo_model()
    results = model(rgb_img, verbose=False)[0]
    conf_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in YOLO_CLASSES or conf < YOLO_CONF_THRESH:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        u_bc, v_bc = int((x1 + x2) / 2), int(y2)
        v_bc = np.clip(v_bc, 0, depth_m.shape[0] - 1)
        u_bc = np.clip(u_bc, 0, depth_m.shape[1] - 1)
        Z = depth_m[v_bc, u_bc]
        if Z <= 0:
            continue
        X = (u_bc - cx) * Z / fx
        row, col = int(Z / CELL_SIZE), int((X + GRID_RANGE_LAT) / CELL_SIZE)
        if 0 <= row < GRID_H and 0 <= col < GRID_W:
            r0, r1 = max(0, row - 1), min(GRID_H, row + 2)
            c0, c1 = max(0, col - 1), min(GRID_W, col + 2)
            conf_grid[r0:r1, c0:c1] = np.maximum(conf_grid[r0:r1, c0:c1], conf)
    return conf_grid

#model definitions

class FusionMLP(nn.Module):
    def __init__(self, in_ch=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
    def forward(self, x):
        return self.net(x)
 
 
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, base=32):
        super().__init__()
        import torch.nn.functional as F
        self._F = F
        def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.d1, self.d2, self.d3, self.d4 = down(in_ch, base), down(base, base*2), down(base*2, base*4), down(base*4, base*8)
        self.u1, self.u2, self.u3 = up(base*8, base*4), up(base*8, base*2), up(base*4, base)
        self.u4 = nn.ConvTranspose2d(base*2, out_ch, 4, 2, 1)
 
    def forward(self, x):
        F = self._F
        B, C, H, W = x.shape
        pad_h, pad_w = (16 - H % 16) % 16, (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        e1 = self.d1(x); e2 = self.d2(e1); e3 = self.d3(e2); e4 = self.d4(e3)
        u1 = self.u1(e4)
        u2 = self.u2(torch.cat([u1, e3], 1))
        u3 = self.u3(torch.cat([u2, e2], 1))
        out = self.u4(torch.cat([u3, e1], 1))
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out
 
 
def load_trained_fusion_model():
    if USE_MODEL == "mlp":
        model = FusionMLP().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "fusion_mlp.pt"), map_location=DEVICE))
    else:
        model = UNetGenerator().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "unet_generator.pt"), map_location=DEVICE))
    model.eval()
    return model

# Visualization 
 
def grid_to_heatmap_png(grid_0to1, out_path):
    heat = (np.clip(grid_0to1, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    cv2.imwrite(out_path, heat_color)


def process_one_mapillary_image(rgb_path, fusion_model):
    rgb = cv2.imread(rgb_path)
    H, W = rgb.shape[:2]
    fx, fy, cx, cy = default_intrinsics(W, H)
 
    rel_depth = estimate_relative_depth(rgb)
    metric_depth = ground_plane_calibrate(rel_depth, fx, fy, cx, cy)
 
    count_grid, depth_sum_grid, depth_sq_sum_grid = backproject_to_bev(metric_depth, fx, fy, cx, cy)
    density = make_density_map(count_grid)
    depth_mean, depth_var = make_depth_mean_var(count_grid, depth_sum_grid, depth_sq_sum_grid)
 
    edge_conf_img = make_edge_confidence_map(rgb)
    edge_conf_bev = project_image_map_to_bev(edge_conf_img, metric_depth, fx, fy, cx, cy)
 
    yolo_conf_bev = make_yolo_conf_map(rgb, metric_depth, fx, fy, cx, cy)
 
    input_stack = np.stack([density, edge_conf_bev, yolo_conf_bev, depth_mean, depth_var], axis=0).astype(np.float32)
 
    # OLD: fixed hand-tuned weights (paper Section 3.5)
    w_d, w_e, w_y = FIXED_WEIGHTS
    old_grid = w_d * density + w_e * edge_conf_bev + w_y * yolo_conf_bev
 
    # NEW: trained fusion model
    with torch.no_grad():
        x_t = torch.from_numpy(input_stack).unsqueeze(0).to(DEVICE)
        new_grid = torch.sigmoid(fusion_model(x_t)).squeeze().cpu().numpy()
 
    return old_grid, new_grid
 
 
SELECTED_FILENAMES_PATH = "/kaggle/working/selected_40_filenames.txt"
 
 
def main():
    if os.path.exists(SELECTED_FILENAMES_PATH):
        with open(SELECTED_FILENAMES_PATH, "r") as f:
            selected_names = set(line.strip() for line in f if line.strip())
        all_paths = sorted(glob.glob(os.path.join(MAPILLARY_DIR, "*.jpg")) +
                            glob.glob(os.path.join(MAPILLARY_DIR, "*.png")))
        image_paths = [p for p in all_paths if os.path.basename(p) in selected_names]
        print(f"Filtering to the {len(selected_names)} selected filenames -> "
              f"found {len(image_paths)} matching files in {MAPILLARY_DIR}")
        if len(image_paths) != len(selected_names):
            missing = selected_names - set(os.path.basename(p) for p in image_paths)
            print(f"  WARNING: {len(missing)} selected filenames not found in MAPILLARY_DIR: {missing}")
    else:
        image_paths = sorted(glob.glob(os.path.join(MAPILLARY_DIR, "*.jpg")) +
                              glob.glob(os.path.join(MAPILLARY_DIR, "*.png")))
        print(f"No {SELECTED_FILENAMES_PATH} found -- processing all "
              f"{len(image_paths)} images in {MAPILLARY_DIR} instead.")
 
    if len(image_paths) == 0:
        print("0 found -- fix MAPILLARY_DIR at the top of this script first.")
        return
 
    fusion_model = load_trained_fusion_model()
    print(f"Loaded trained fusion model: {USE_MODEL}")
 
    for i, rgb_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        old_grid, new_grid = process_one_mapillary_image(rgb_path, fusion_model)
 
        np.save(os.path.join(OUT_DIR, f"{name}_old.npy"), old_grid)
        np.save(os.path.join(OUT_DIR, f"{name}_new.npy"), new_grid)
        grid_to_heatmap_png(old_grid, os.path.join(OUT_DIR, f"{name}_old.png"))
        grid_to_heatmap_png(new_grid, os.path.join(OUT_DIR, f"{name}_new.png"))
 
        if (i + 1) % 10 == 0:
            print(f"processed {i+1}/{len(image_paths)}")
 
    print(f"Done. Old/new BEV grids (.npy + heatmap .png) saved to {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()
