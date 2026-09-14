#!/usr/bin/env python3
# Extracted verbatim from notebooks/trainingpipeline-v1.ipynb (cells [12, 13, 14, 15, 16, 17, 18, 19]).
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

#config 
DATA_DIR = "/kaggle/working/bev_pseudolabels"
CKPT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
VAL_FRACTION = 0.15   # held-out split, drawn from the KITTI pairs (separate from your Table 1 test-100)
BATCH_SIZE = 8
MLP_EPOCHS = 60
GAN_EPOCHS = 80
LR = 1e-3
GAN_LR = 2e-4
LAMBDA_ADV = 0.1       # weight on the adversarial term, kept small deliberately (see design doc)
FIXED_WEIGHTS = (0.5, 0.2, 0.3)   # density, edge_conf, yolo_conf -- your paper's original Section 3.5
 
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

#dataset

class BEVPseudoLabelDataset(Dataset):
    """Loads the .pt files saved by kitti_bev_pseudolabels.py.
    Each item: input (5,H,W) float32, label (H,W) float32 in {0,1}."""
 
    def __init__(self, data_dir=DATA_DIR):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(
                f"No .pt files found in {data_dir} -- run kitti_bev_pseudolabels.py first."
            )
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        return d["input"], d["label"]
 
 
def get_dataloaders():
    full_ds = BEVPseudoLabelDataset()
    n_val = max(1, int(len(full_ds) * VAL_FRACTION))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Dataset: {len(full_ds)} total | {n_train} train | {n_val} held-out val")
    return train_loader, val_loader

# Model 1: FusionMLP 
 
class FusionMLP(nn.Module):
    """Per-cell MLP, applied identically to every grid cell via 1x1 convs
    (equivalent to a per-cell MLP, but avoids reshaping HxW into a flat batch)."""
 
    def __init__(self, in_ch=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
 
    def forward(self, x):  # x: (B,5,H,W) -> (B,1,H,W) logits
        return self.net(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, base=32):
        super().__init__()
        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))
        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.d1 = down(in_ch, base)
        self.d2 = down(base, base * 2)
        self.d3 = down(base * 2, base * 4)
        self.d4 = down(base * 4, base * 8)
        self.u1 = up(base * 8, base * 4)
        self.u2 = up(base * 8, base * 2)
        self.u3 = up(base * 4, base)
        self.u4 = nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        u1 = self.u1(e4)
        u2 = self.u2(torch.cat([u1, e3], 1))
        u3 = self.u3(torch.cat([u2, e2], 1))
        out = self.u4(torch.cat([u3, e1], 1))
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        def block(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            block(in_ch, base, norm=False),
            block(base, base * 2),
            block(base * 2, base * 4),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))

# Metrics 
 
@torch.no_grad()
def iou_f1(pred_prob, label, thresh=0.5, eps=1e-6):
    pred = (pred_prob > thresh).float()
    tp = (pred * label).sum()
    fp = (pred * (1 - label)).sum()
    fn = ((1 - pred) * label).sum()
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return iou.item(), f1.item()
 
 
@torch.no_grad()
def best_threshold_iou_f1(all_preds, all_labels, thresholds=None):
    """Sweeps thresholds and returns the best (by F1) IoU/F1/threshold. Fairer than a
    fixed 0.5 cutoff when comparing a calibrated (trained) model against an
    uncalibrated hand-tuned score -- see discussion in the response this patch answers."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    preds_cat = torch.cat([p.flatten() for p in all_preds])
    labels_cat = torch.cat([l.flatten() for l in all_labels])
    best = {"f1": -1, "iou": 0, "thresh": 0.5}
    for t in thresholds:
        i, f = iou_f1(preds_cat, labels_cat, thresh=float(t))
        if f > best["f1"]:
            best = {"f1": f, "iou": i, "thresh": float(t)}
    return best["iou"], best["f1"], best["thresh"]
 
 
@torch.no_grad()
def evaluate_fixed_weights(loader, weights=FIXED_WEIGHTS):
    """Baseline: your paper's original hand-tuned fusion, no learning at all.
    Returns (iou@0.5, f1@0.5, iou@best, f1@best, best_thresh)."""
    w_density, w_edge, w_yolo = weights
    all_preds, all_labels = [], []
    for x, y in loader:
        density, edge_conf, yolo_conf = x[:, 0], x[:, 1], x[:, 2]
        pred = w_density * density + w_edge * edge_conf + w_yolo * yolo_conf
        all_preds.append(pred)
        all_labels.append(y)
    iou_50, f1_50 = iou_f1(torch.cat([p.flatten() for p in all_preds]),
                            torch.cat([l.flatten() for l in all_labels]))
    iou_best, f1_best, best_t = best_threshold_iou_f1(all_preds, all_labels)
    return iou_50, f1_50, iou_best, f1_best, best_t
 
 
@torch.no_grad()
def evaluate_model(model, loader):
    """Returns (iou@0.5, f1@0.5, iou@best, f1@best, best_thresh)."""
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu()
        all_preds.append(prob)
        all_labels.append(y.cpu())
    iou_50, f1_50 = iou_f1(torch.cat([p.flatten() for p in all_preds]),
                            torch.cat([l.flatten() for l in all_labels]))
    iou_best, f1_best, best_t = best_threshold_iou_f1(all_preds, all_labels)
    return iou_50, f1_50, iou_best, f1_best, best_t
 
 
@torch.no_grad()
def evaluate_model_iou_only(model, loader):
    """Cheap version (@0.5 only) for use inside the per-epoch training loop print --
    the full best_threshold_iou_f1 sweep is reserved for the final comparison."""
    model.eval()
    ious, f1s = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        prob = torch.sigmoid(model(x)).squeeze(1)
        for b in range(prob.shape[0]):
            i, f = iou_f1(prob[b], y[b])
            ious.append(i); f1s.append(f)
    return float(np.mean(ious)), float(np.mean(f1s))

# Training: FusionMLP 
 
def pos_weight_for_batch(y, eps=1e-6):
    n_pos = y.sum()
    n_neg = y.numel() - n_pos
    return (n_neg / (n_pos + eps)).clamp(max=50.0)  # cap to avoid instability on near-empty grids
 
 
def train_fusion_mlp(train_loader, val_loader):
    print("\n=== Training FusionMLP ===")
    model = FusionMLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_iou, best_state = -1.0, None
 
    for epoch in range(MLP_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).squeeze(1)
            pw = pos_weight_for_batch(y)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
 
        val_iou, val_f1 = evaluate_model_iou_only(model, val_loader)
        if val_iou > best_iou:
            best_iou = val_iou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
 
        if (epoch + 1) % 10 == 0 or epoch == MLP_EPOCHS - 1:
            print(f"  epoch {epoch+1}/{MLP_EPOCHS}  train_loss={epoch_loss/len(train_loader):.4f}  "
                  f"val_IoU={val_iou:.4f}  val_F1={val_f1:.4f}  (best so far: {best_iou:.4f})")
 
    model.load_state_dict(best_state)  # restore best checkpoint, not final epoch
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "fusion_mlp.pt"))
    print(f"  Loaded best checkpoint (val_IoU={best_iou:.4f}) for final evaluation.")
    return model

#Training: U-Net + GAN
 
def train_gan(train_loader, val_loader, lambda_adv=LAMBDA_ADV):
    print("\n=== Training U-Net + PatchGAN ===")
    G = UNetGenerator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)
    opt_g = torch.optim.Adam(G.parameters(), lr=GAN_LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=GAN_LR, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    best_iou, best_state = -1.0, None
 
    for epoch in range(GAN_EPOCHS):
        G.train(); D.train()
        g_loss_total, d_loss_total = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_img = y.unsqueeze(1)  # (B,1,H,W) to match discriminator's expected shape
 
            # ---- Discriminator step ----
            with torch.no_grad():
                fake_logits = G(x)
                fake_prob = torch.sigmoid(fake_logits)
            d_real = D(x, y_img)
            d_fake = D(x, fake_prob)
            d_loss = 0.5 * (
                bce(d_real, torch.ones_like(d_real))
                + bce(d_fake, torch.zeros_like(d_fake))
            )
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
 
            # ---- Generator step ----
            fake_logits = G(x)
            fake_prob = torch.sigmoid(fake_logits)
            pw = pos_weight_for_batch(y)
            recon_loss = F.binary_cross_entropy_with_logits(fake_logits.squeeze(1), y, pos_weight=pw)
            d_fake_for_g = D(x, fake_prob)
            adv_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_loss = recon_loss + lambda_adv * adv_loss
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
 
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
 
        val_iou, val_f1 = evaluate_model_iou_only(G, val_loader)
        if val_iou > best_iou:
            best_iou = val_iou
            best_state = {k: v.clone() for k, v in G.state_dict().items()}
 
        if (epoch + 1) % 10 == 0 or epoch == GAN_EPOCHS - 1:
            print(f"  epoch {epoch+1}/{GAN_EPOCHS}  G_loss={g_loss_total/len(train_loader):.4f}  "
                  f"D_loss={d_loss_total/len(train_loader):.4f}  val_IoU={val_iou:.4f}  val_F1={val_f1:.4f}  "
                  f"(best so far: {best_iou:.4f})")
 
    G.load_state_dict(best_state)  # restore best checkpoint, not final (overfitting-prone) epoch
    torch.save(G.state_dict(), os.path.join(CKPT_DIR, "unet_generator.pt"))
    torch.save(D.state_dict(), os.path.join(CKPT_DIR, "patch_discriminator.pt"))
    print(f"  Loaded best checkpoint (val_IoU={best_iou:.4f}) for final evaluation.")
    return G, D

# Main: run all three + compare 
 
def main():
    train_loader, val_loader = get_dataloaders()
 
    print("\n=== Baseline: hand-tuned fixed weights (paper Section 3.5, 0.5/0.2/0.3) ===")
    fixed_iou50, fixed_f150, fixed_iou_best, fixed_f1_best, fixed_t = evaluate_fixed_weights(val_loader)
    print(f"  Fixed weights  ->  IoU@0.5={fixed_iou50:.4f}  F1@0.5={fixed_f150:.4f}  |  "
          f"IoU@best(t={fixed_t:.2f})={fixed_iou_best:.4f}  F1@best={fixed_f1_best:.4f}")
 
    mlp_model = train_fusion_mlp(train_loader, val_loader)
    mlp_iou50, mlp_f150, mlp_iou_best, mlp_f1_best, mlp_t = evaluate_model(mlp_model, val_loader)
 
    gan_G, gan_D = train_gan(train_loader, val_loader)
    gan_iou50, gan_f150, gan_iou_best, gan_f1_best, gan_t = evaluate_model(gan_G, val_loader)
 
    print("\n" + "=" * 78)
    print("FINAL COMPARISON (same held-out split, same pseudo-GT)")
    print("=" * 78)
    print(f"{'Method':<28}{'IoU@0.5':>10}{'F1@0.5':>10}{'IoU@best':>11}{'F1@best':>10}{'thresh':>8}")
    print(f"{'Fixed weights (0.5/0.2/0.3)':<28}{fixed_iou50:>10.4f}{fixed_f150:>10.4f}"
          f"{fixed_iou_best:>11.4f}{fixed_f1_best:>10.4f}{fixed_t:>8.2f}")
    print(f"{'FusionMLP (learned)':<28}{mlp_iou50:>10.4f}{mlp_f150:>10.4f}"
          f"{mlp_iou_best:>11.4f}{mlp_f1_best:>10.4f}{mlp_t:>8.2f}")
    print(f"{'U-Net+GAN (learned)':<28}{gan_iou50:>10.4f}{gan_f150:>10.4f}"
          f"{gan_iou_best:>11.4f}{gan_f1_best:>10.4f}{gan_t:>8.2f}")
    print("=" * 78)
    print("Checkpoints saved to:", CKPT_DIR)
    print("\nNote: @0.5 column is included for completeness, but the fixed-weight score's\n"
          "own per-image density normalization means it's not calibrated to a 0.5 cutoff\n"
          "the way the trained models are (see discussion). @best is the fairer comparison\n"
          "to report as the headline ablation number.")
 
    with open(os.path.join(CKPT_DIR, "results_summary.txt"), "w") as f:
        f.write(f"Fixed weights: IoU@0.5={fixed_iou50:.4f} F1@0.5={fixed_f150:.4f} | "
                f"IoU@best(t={fixed_t:.2f})={fixed_iou_best:.4f} F1@best={fixed_f1_best:.4f}\n")
        f.write(f"FusionMLP: IoU@0.5={mlp_iou50:.4f} F1@0.5={mlp_f150:.4f} | "
                f"IoU@best(t={mlp_t:.2f})={mlp_iou_best:.4f} F1@best={mlp_f1_best:.4f}\n")
        f.write(f"U-Net+GAN: IoU@0.5={gan_iou50:.4f} F1@0.5={gan_f150:.4f} | "
                f"IoU@best(t={gan_t:.2f})={gan_iou_best:.4f} F1@best={gan_f1_best:.4f}\n")
 
 
if __name__ == "__main__":
    main()
