#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [54, 55, 56, 57, 58, 59, 60, 61, 62]).
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

# Config 
 
MAPILLARY_DIR = "/kaggle/input/datasets/coconotchanel/mapillary-subset-usage"
SELECTED_FILENAMES_PATH = "/kaggle/working/selected_40_filenames.txt"
OUT_DIR = "/kaggle/working/pose_integrated"
os.makedirs(OUT_DIR, exist_ok=True)
 
CKPT_DIR = "/kaggle/working/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
GRID_RANGE_FWD = 40.0
GRID_RANGE_LAT = 20.0
CELL_SIZE = 0.2
GRID_H = int(GRID_RANGE_FWD / CELL_SIZE)
GRID_W = int(2 * GRID_RANGE_LAT / CELL_SIZE)
 
ASSUMED_CAMERA_HEIGHT = 1.5
GROUND_BAND_ROWS = (0.75, 0.95)
YOLO_CONF_THRESH = 0.35
YOLO_CLASSES = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
EDGE_WEIGHTS = (0.4, 0.4, 0.2)
 
# Canonical vehicle footprint (length, width) in meters -- same assumption as
# the 4-DoF pose design doc. Single template for all classes, stated simplification.
ASSUMED_LENGTH = 4.5
ASSUMED_WIDTH = 1.8
 
CROP_SIZE = 224
NUM_BINS = 2

# MiDaS depth (reused pattern) 
 
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
    model, transform = get_midas()
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(DEVICE)
    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=rgb_bgr.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()
    return prediction.cpu().numpy()
 
 
def default_intrinsics(img_w, img_h):
    fx = fy = 0.85 * img_w
    cx, cy = img_w / 2.0, img_h / 2.0
    return fx, fy, cx, cy
 
 
def ground_plane_calibrate(rel_depth, fx, fy, cx, cy, camera_height=ASSUMED_CAMERA_HEIGHT,
                            band=GROUND_BAND_ROWS):
    H, W = rel_depth.shape
    row_lo, row_hi = int(band[0] * H), int(band[1] * H)
    rows = np.arange(row_lo, row_hi)
    valid_rows = rows[rows > cy]
    if len(valid_rows) == 0:
        scale = 1.0
    else:
        z_ground = camera_height * fy / (valid_rows - cy)
        band_rel_depth = rel_depth[valid_rows, :].mean(axis=1)
        band_rel_depth = np.clip(band_rel_depth, 1e-4, None)
        scale = float(np.median(z_ground * band_rel_depth))
    metric_depth = scale / np.clip(rel_depth, 1e-4, None)
    return np.clip(metric_depth, 0.5, 80.0).astype(np.float32)

# FusionMLP (reused, must match training script) 
 
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
 
 
def load_fusion_mlp():
    model = FusionMLP().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "fusion_mlp.pt"), map_location=DEVICE))
    model.eval()
    return model


# BEV feature functions (reused) 
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
 
 
def make_yolo_conf_map_and_boxes(rgb_img, depth_m, fx, fy, cx, cy):
    """Same as before, but now ALSO returns the raw box list (for the pose head),
    not just the confidence map used by FusionMLP."""
    model = get_yolo_model()
    results = model(rgb_img, verbose=False)[0]
    conf_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in YOLO_CLASSES or conf < YOLO_CONF_THRESH:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        u_bc, v_bc = int((x1 + x2) / 2), int(y2)
        v_bc_clip = np.clip(v_bc, 0, depth_m.shape[0] - 1)
        u_bc_clip = np.clip(u_bc, 0, depth_m.shape[1] - 1)
        Z = depth_m[v_bc_clip, u_bc_clip]
        boxes.append({"bbox": (float(x1), float(y1), float(x2), float(y2)),
                       "conf": conf, "cls_id": cls_id,
                       "ground_contact_px": (u_bc, v_bc), "Z": float(Z)})
        if Z > 0:
            X = (u_bc - cx) * Z / fx
            row, col = int(Z / CELL_SIZE), int((X + GRID_RANGE_LAT) / CELL_SIZE)
            if 0 <= row < GRID_H and 0 <= col < GRID_W:
                r0, r1 = max(0, row - 1), min(GRID_H, row + 2)
                c0, c1 = max(0, col - 1), min(GRID_W, col + 2)
                conf_grid[r0:r1, c0:c1] = np.maximum(conf_grid[r0:r1, c0:c1], conf)
    return conf_grid, boxes

# Yaw head (reused from train_yaw_head.py) 
 
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
 
 
def crop_and_stack(rgb_full, bbox):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h_img, w_img = rgb_full.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    crop = rgb_full[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    canny, sobel, emboss = compute_edge_channels(crop)
    rgb_norm = crop.astype(np.float32) / 255.0
    stacked = np.dstack([rgb_norm, canny, sobel, emboss])
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)
 
 
class YawHead(nn.Module):
    def __init__(self, num_bins=NUM_BINS, in_channels=6):
        super().__init__()
        backbone = models.resnet18(weights=None)  # weights loaded from checkpoint, not ImageNet, at inference
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
 
 
def load_yaw_head():
    model = YawHead().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "yaw_head.pt"), map_location=DEVICE))
    model.eval()
    return model
 
 
def decode_bin_prediction(bin_logits, cossin, num_bins=NUM_BINS):
    bin_centers = np.linspace(-np.pi, np.pi, num_bins, endpoint=False)
    bin_id = int(torch.argmax(bin_logits).item())
    cos_r, sin_r = cossin[bin_id]
    residual = float(torch.atan2(sin_r, cos_r))
    alpha = bin_centers[bin_id] + residual
    return float(np.angle(np.exp(1j * alpha)))
 
 
@torch.no_grad()
def predict_alpha(yaw_model, rgb_full, bbox):
    crop_tensor = crop_and_stack(rgb_full, bbox)
    if crop_tensor is None:
        return None
    x = torch.from_numpy(crop_tensor).unsqueeze(0).to(DEVICE)
    conf, cossin = yaw_model(x)
    return decode_bin_prediction(conf[0].cpu(), cossin[0].cpu())

 
# Pose assembly: yaw + depth -> oriented footprint
 
def compute_oriented_footprint(X, Z, global_yaw, length=ASSUMED_LENGTH, width=ASSUMED_WIDTH):
    """Returns 4 (X,Z) footprint corners in camera-frame ground-plane coordinates,
    given the estimated center position and global yaw. Translation here comes
    from the depth back-projection (X,Z), NOT a separate least-squares fit --
    see module docstring."""
    half_l, half_w = length / 2.0, width / 2.0
    corners_obj = np.array([
        [ half_l,  half_w], [ half_l, -half_w],
        [-half_l, -half_w], [-half_l,  half_w],
    ])
    cos_y, sin_y = np.cos(global_yaw), np.sin(global_yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    corners_cam = corners_obj @ R.T + np.array([X, Z])
    return corners_cam  # (4,2) in (X,Z) meters
 
 
def footprint_to_grid_coords(corners_cam):
    grid_pts = []
    for X, Z in corners_cam:
        row = int(Z / CELL_SIZE)
        col = int((X + GRID_RANGE_LAT) / CELL_SIZE)
        grid_pts.append((col, row))
    return np.array(grid_pts, dtype=np.int32)

# Visualization 
def render_overlay(occupancy_grid, oriented_boxes_grid_coords, out_path):
    heat = (np.clip(occupancy_grid, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    for pts in oriented_boxes_grid_coords:
        valid_pts = pts.copy()
        valid_pts[:, 0] = np.clip(valid_pts[:, 0], 0, GRID_W - 1)
        valid_pts[:, 1] = np.clip(valid_pts[:, 1], 0, GRID_H - 1)
        cv2.polylines(heat_color, [valid_pts], isClosed=True, color=(255, 255, 255), thickness=1)
    cv2.imwrite(out_path, heat_color)

# ----------------------------- Main per-image pipeline -------------------------
 
def process_one_image(rgb_path, fusion_model, yaw_model):
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
    yolo_conf_bev, boxes = make_yolo_conf_map_and_boxes(rgb, metric_depth, fx, fy, cx, cy)
 
    input_stack = np.stack([density, edge_conf_bev, yolo_conf_bev, depth_mean, depth_var], axis=0).astype(np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(input_stack).unsqueeze(0).to(DEVICE)
        occupancy_grid = torch.sigmoid(fusion_model(x_t)).squeeze().cpu().numpy()
 
    pose_records = []
    footprints_grid = []
    for box in boxes:
        if box["Z"] <= 0:
            continue
        alpha = predict_alpha(yaw_model, rgb, box["bbox"])
        if alpha is None:
            continue
 
        u_bc, v_bc = box["ground_contact_px"]
        Z = box["Z"]
        X = (u_bc - cx) * Z / fx
        theta_ray = float(np.arctan2(X, Z))
        global_yaw = float(np.angle(np.exp(1j * (alpha + theta_ray))))
 
        footprint_cam = compute_oriented_footprint(X, Z, global_yaw)
        footprint_grid = footprint_to_grid_coords(footprint_cam)
        footprints_grid.append(footprint_grid)
 
        pose_records.append({
            "bbox": box["bbox"], "conf": box["conf"], "cls_id": box["cls_id"],
            "X": X, "Z": Z, "alpha_deg": np.degrees(alpha),
            "theta_ray_deg": np.degrees(theta_ray), "global_yaw_deg": np.degrees(global_yaw),
        })
 
    return occupancy_grid, footprints_grid, pose_records
 
 
def main():
    if os.path.exists(SELECTED_FILENAMES_PATH):
        with open(SELECTED_FILENAMES_PATH, "r") as f:
            selected_names = set(line.strip() for line in f if line.strip())
        all_paths = sorted(glob.glob(os.path.join(MAPILLARY_DIR, "*.jpg")) +
                            glob.glob(os.path.join(MAPILLARY_DIR, "*.png")))
        image_paths = [p for p in all_paths if os.path.basename(p) in selected_names]
    else:
        image_paths = sorted(glob.glob(os.path.join(MAPILLARY_DIR, "*.jpg")) +
                              glob.glob(os.path.join(MAPILLARY_DIR, "*.png")))
    print(f"Processing {len(image_paths)} images.")
 
    fusion_model = load_fusion_mlp()
    yaw_model = load_yaw_head()
    print("Loaded FusionMLP and YawHead checkpoints.")
 
    all_records = {}
    for i, rgb_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        occupancy_grid, footprints_grid, pose_records = process_one_image(rgb_path, fusion_model, yaw_model)
 
        render_overlay(occupancy_grid, footprints_grid, os.path.join(OUT_DIR, f"{name}_pose_overlay.png"))
        all_records[name] = pose_records
 
        if (i + 1) % 10 == 0:
            print(f"processed {i+1}/{len(image_paths)}")
 
    with open(os.path.join(OUT_DIR, "all_pose_records.json"), "w") as f:
        json.dump(all_records, f, indent=2)
 
    print(f"Done. Overlays + pose records saved to {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()
