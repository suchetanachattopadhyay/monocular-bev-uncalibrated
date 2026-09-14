# Monocular BEV Occupancy and 4-DoF Vehicle Pose from Uncalibrated Street Imagery

Code and results for *Calibrated Occupancy Fusion and 4-DoF Vehicle Pose for Monocular
Bird's-Eye-View Mapping of Uncalibrated Street Imagery* (ICVGIP <!-- YEAR -->, first author).

**The problem.** Standard BEV perception assumes a calibrated multi-camera rig, LiDAR
supervision, and temporal context. Crowd-sourced street photos, dashcam stills and phone
captures have none of those. This pipeline takes **one uncalibrated RGB image** — no
intrinsics, no extrinsics, no depth ground truth, no previous frame — and produces a
200×200 BEV occupancy grid (0.2 m cells, 40 m forward, ±20 m lateral) with oriented
vehicle footprints.

**Headline numbers.** Replacing a hand-tuned weighted sum over the BEV feature channels
with a learned per-cell fusion raises held-out IoU on KITTI LiDAR pseudo-labels from
0.251 to 0.822, and on 40 in-the-wild images raises recovery of geometrically supported
cells from 14.0% to 75.2% while cutting isolated-cell speckle from 23.3% to 2.3%.

**What that does and doesn't mean.** The pseudo-label is a threshold on projected point
count, and point density is also input channel 0 — so the learned model is largely
recovering a *calibrated decision threshold* that the hand-tuned sum destroyed, not
discovering new geometric structure. The paper argues this explicitly (§4.4) rather than
presenting the 3.3× as a capability gain. The in-the-wild statistics are behavioural
(support recovery, speckle, component count), not accuracy — there is no ground truth on
that domain and none is claimed.

---

## Pipeline

```
single RGB image
  ├─ MiDaS-small relative depth ──┐
  ├─ edge bank (Canny/Sobel/emboss)─┼─→ 5-channel BEV stack (200×200) ─→ FusionMLP ─→ occupancy grid
  └─ YOLOv8n vehicle boxes ───────┘                                                      │
        └─ per-box crop + edge channels ─→ MultiBin yaw head ─→ α                        │
                    metric depth at box contact point ─→ (X, Z), θ_ray ─→ θ = α + θ_ray ─┘
```

Metric scale comes from a single assumption: the image band between 0.75H and 0.95H is
flat ground at camera height h = 1.5 m. Intrinsics are assumed as fx = fy = 0.85·W with a
central principal point. Both assumptions are localised to one equation and are the
dominant error source (see Limitations).

## Results

| Fusion method | params | IoU@0.5 | IoU@best | threshold |
|---|---|---|---|---|
| Hand-tuned (0.5 density / 0.2 edge / 0.3 detector) | 0 | 0.0003 | 0.251 | 0.05 |
| U-Net + PatchGAN | ~10⁶ | 0.722 | 0.722 | 0.50 |
| **FusionMLP** (1×1 convs, per-cell) | **421** | 0.694 | **0.822** | 0.85 |

The 421-parameter per-cell model beats the spatial U-Net at each model's own best-F1
threshold. Raw numbers: [`results/results_summary.txt`](results/results_summary.txt).

Yaw head: MultiBin over a 6-channel ResNet-18 (RGB + Canny/Sobel/emboss), trained on
27,965 filtered KITTI Car/Van instances (15% held out). On the held-out split it reaches
**97.85% bin accuracy, 3.77° mean absolute angular error, and 0.9931 mean orientation
similarity**. Applied to the 40-image target set it produces 168 vehicle poses. Its
predicted observation angles collapse towards the perpendicular band (78.6% in
45°≤|α|≤135°), which is diagnosed in the paper as a consequence of the `N_BINS = 2`
simplification, not a property of the scenes.

**Negative result worth the space:** a Qwen2-VL-2B judge used as a stand-in for human
raters scored near ceiling on every axis (4.65 / 4.68 / 5.00), agreed with a human rater
at chance (κ_lin = 0.042, 0.022), and was *perfectly* reproducible across reruns at
temperature 0.7 (κ_lin = 1.000). Self-consistency was not validity. The quantitative
claims were moved onto geometric statistics instead.

## Repository layout

```
scripts/                       pipeline stages, extracted verbatim from the notebook
  01_kitti_bev_pseudolabels.py   KITTI LiDAR → 5-channel stack + occupancy labels (1000 pairs)
  02_train_fusion_models.py      trains + compares all three fusion strategies
  03_select_diverse_40.py        CLIP ViT-B/32 + k-means(40) target-set selection
  04_inference_mapillary.py      fusion inference on the in-the-wild set
  05_train_yaw_head.py           MultiBin orientation head on KITTI 3D labels
  06_pose_integrated_pipeline.py full pipeline: occupancy + oriented footprints
  07_show_pose_overlays.py       inline overlay viewer
  08_extract_crops_for_labeling.py  exports 168 vehicle crops + label template
  09_zero_shot_vs_few_shot.py    yaw-head evaluation against hand labels
notebooks/trainingpipeline-v1.ipynb   the original Kaggle notebook the scripts came from
docs/                          fileLog.docx — documentation of a separate, earlier
                               internship codebase; see docs/README.md
results/                       occupancy grids, pose overlays, pose records, metrics
checkpoints/                   fusion_mlp.pt, unet_generator.pt, patch_discriminator.pt, yaw_head.pt
paper/                         the paper PDF
```

## Reproducing

Developed on Kaggle (Tesla T4). The scripts retain their original `/kaggle/input/...`
paths — edit the config block at the top of each before running elsewhere.

```bash
pip install -r requirements.txt
python scripts/01_kitti_bev_pseudolabels.py     # writes bev_pseudolabels/*.pt
python scripts/02_train_fusion_models.py        # writes checkpoints/ + results_summary.txt
python scripts/03_select_diverse_40.py          # writes selected_40_filenames.txt
python scripts/06_pose_integrated_pipeline.py   # writes pose_integrated/
```

Seeds are fixed at 42 for splits, k-means, and initialisation. Training uses the
best-validation-IoU checkpoint, not the final epoch.

Datasets are **not** redistributed here. KITTI depth-prediction (`val_selection_cropped`)
and KITTI 3D object detection come from the official benchmark; the target imagery is a
subset of Mapillary Vistas. Obtain both from their own sources under their own licences.

## Limitations

Stated in full in §5 of the paper; the short version:

- Equation (2) fits a depth **scale but not the shift** that mixed-dataset relative depth
  also needs. The radial striping in far-field maps is that missing term's signature.
- Assumed intrinsics are a consumer-optics average, not the camera that took any given
  photo. Focal-length error scales lateral position linearly.
- The KITTI split is drawn from continuous drives, so a random 15% split puts near-duplicate
  frames on both sides. Table 2 should be read as an upper bound.
- The pseudo-label is a function of an input channel — this bounds what the fusion
  comparison can establish at all.
- Pose is coarse by construction: 2 MultiBin bins, one 4.5 × 1.8 m footprint template for
  every vehicle class including trucks and buses, and translation from ground-contact
  back-projection rather than a 2D–3D box-edge solve.
- The target set is 40 images, selected by a fixed protocol but still 40, and unlabelled.
- The `facing_label` column in the exported label template is still blank — the zero-shot
  vs few-shot yaw comparison in `09_` is written but **has not been run**. It is not
  claimed as a result anywhere.

## Next steps

Proper affine (scale-and-shift) depth alignment; an overlapping-bin MultiBin head with
N ≥ 4; completion of the human facing-direction annotation over the 168 crops; and an
occupancy supervision signal independent of the density channel.

## Licence

Code: see `LICENSE`. Model checkpoints derive from KITTI-trained supervision and inherit
KITTI's non-commercial terms.
