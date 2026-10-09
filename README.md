# Monocular BEV Occupancy and 4-DoF Vehicle Pose from Uncalibrated Street Imagery

Code and results for *Calibrated Occupancy Fusion and 4-DoF Vehicle Pose for Monocular
Bird's-Eye-View Mapping of Uncalibrated Street Imagery*, Suchetana Chattopadhyay and
Souparna Chatterjee, to appear at **ICVGIP 2026** (17th Indian Conference on Computer
Vision, Graphics and Image Processing, Kolkata).

📄 **Paper:** [`paper/Chattopadhyay_Chatterjee_ICVGIP2026.pdf`](paper/Chattopadhyay_Chatterjee_ICVGIP2026.pdf)
(camera-ready; LaTeX source in [`paper/latex/`](paper/latex/))

**The problem.** Standard BEV perception assumes a calibrated multi-camera rig, LiDAR
supervision, and temporal context. Crowd-sourced street photos, dashcam stills and phone
captures have none of those. This pipeline takes **one uncalibrated RGB image** — no
intrinsics, no extrinsics, no depth ground truth, no previous frame — and produces a
200×200 BEV occupancy grid (0.2 m cells, 40 m forward, ±20 m lateral) with oriented
vehicle footprints.

**Headline numbers.** All fusion methods are scored against the same KITTI LiDAR
pseudo-labels under two input regimes:

- **Uncalibrated front-end** (estimated depth + assumed intrinsics — the real setting):
  the U-Net reaches **0.419 IoU** (0.461 with Depth Anything V2 depth) against 0.101
  (0.127) for the hand-tuned weighted sum, with no ground-truth calibration anywhere.
- **Oracle inputs** (ground-truth depth + intrinsics, a reference only): learned fusion
  raises IoU from 0.251 to 0.805 ± 0.018 (3 seeds) — but an isotonic recalibration of the
  density channel alone reaches 0.812, so that gain is **calibration**, not new evidence.
- **In the wild** (40 Mapillary images, no ground truth): learned fusion recovers 75.2% of
  geometrically supported cells versus 14.0%, and cuts isolated-cell speckle from 23.3%
  to 2.3%.

**What that does and doesn't mean.** On oracle inputs the pseudo-label is a threshold on
projected point count, which is also input channel 0, and the experiment confirms the
learned gain is pure recalibration (§4.3). On front-end inputs the learned models beat
every density-only reference, but ablations locate the gain precisely: for the per-cell
MLP it is a *range-dependent* recalibration carried by the depth-mean channel; for the
U-Net it is partly calibration to the training camera and partly spatial context (§4.5).
The in-the-wild statistics are behavioural (support recovery, speckle, component count),
not accuracy — there is no ground truth on that domain and none is claimed.

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

**Front-end inputs** (estimated depth, assumed intrinsics f = 0.85W, h = 1.5 m) — IoU@best / AP:

| Depth source | Hand-tuned | Density (isotonic) | Oracle threshold | FusionMLP | U-Net + PatchGAN |
|---|---|---|---|---|---|
| MiDaS-small + ground-plane scale (ours) | 0.101 / 0.171 | 0.197 / 0.213 | 0.208 | 0.245 / 0.299 | **0.419 / 0.610** |
| Depth Anything V2 † | 0.127 / 0.243 | 0.281 / 0.338 | 0.295 | 0.381 / 0.502 | **0.461 / 0.666** |

† Outdoor metric model fine-tuned on Virtual KITTI 2, so it has a domain advantage on KITTI.
*Density (isotonic)* is the best global recalibration of density alone; *oracle threshold*
picks each test image's best density threshold using its label, so no density-only method
can beat it. U-Net values are 3-seed means (±0.002 for MiDaS-small).

**Oracle inputs** (ground-truth depth and intrinsics; reference only):

| Fusion method | params | IoU@0.5 | IoU@best |
|---|---|---|---|
| Hand-tuned (0.5 density / 0.2 edge / 0.3 detector) | 0 | 0.0003 | 0.251 |
| Density (isotonic) | — | — | 0.812 |
| U-Net + PatchGAN | ~10⁶ | 0.722 | 0.724 |
| FusionMLP (1×1 convs, per-cell) | 421 | 0.694 | 0.805 ± 0.018 (3 seeds) |

FusionMLP does not differ from isotonic density in AP (Δ = −0.004, 95% CI [−0.008, 0.001]).

**Where the drop from oracle to front-end comes from** (IoU@best):

| Depth | Intrinsics | Density (isotonic) | FusionMLP | U-Net |
|---|---|---|---|---|
| ground truth | ground truth | 0.812 | 0.805 | 0.724 |
| ground truth | assumed | 0.34 | 0.379 | 0.581 |
| MiDaS-small | ground truth | 0.29 | 0.33 | — |
| MiDaS-small | assumed | 0.197 | 0.245 | 0.419 |

The assumed intrinsics cost as much as the cheap depth network. Further ablations (§4.5):
removing the depth-mean channel drops front-end FusionMLP to exactly density-only (0.197),
and any other channel changes IoU by ≤ 0.0003; U-Nets trained and tested on the same
assumed focal length absorb the mismatch, and removing the adversarial loss leaves the
U-Net unchanged (0.421 vs 0.417). All conclusions also hold on a drive-held-out 726/274
split, with threshold-free AP, 95% bootstrap CIs and 3 seeds.

Original single-seed oracle numbers: [`results/results_summary.txt`](results/results_summary.txt).

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
paper/                         camera-ready PDF (ICVGIP 2026) + LaTeX source in paper/latex/
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
- KITTI frames come from continuous drives, so the random 850/150 split puts near-duplicate
  frames on both sides; a drive-held-out 726/274 split is also reported, and conclusions
  hold on it.
- The pseudo-label is a function of an input channel. On oracle inputs this makes the fusion
  gain pure calibration; on front-end inputs the per-cell gain is still a recalibration.
- The U-Net's front-end gain is partly calibration to the KITTI camera and would not be
  expected to carry over to a camera with different optics without retraining.
- No calibrated BEV method (MonoLayout, PON, OFT) is compared — they all require known
  intrinsics, which this setting excludes.
- Pose is coarse by construction: 2 MultiBin bins, one 4.5 × 1.8 m footprint template for
  every vehicle class including trucks and buses, and translation from ground-contact
  back-projection rather than a 2D–3D box-edge solve.
- The target set is 40 images, selected by a fixed protocol but still 40, and unlabelled.
- The `facing_label` column in the exported label template is still blank — the zero-shot
  vs few-shot yaw comparison in `09_` is written but **has not been run**. It is not
  claimed as a result anywhere.

## Next steps

Proper affine (scale-and-shift) depth alignment; an overlapping-bin MultiBin head with
N ≥ 4; completion of the human facing-direction annotation over the 168 crops; a
calibrated reference such as MonoLayout; and an occupancy supervision signal independent
of the density channel.

## Citation

```bibtex
@inproceedings{chattopadhyay2026monobev,
  author    = {Suchetana Chattopadhyay and Souparna Chatterjee},
  title     = {Calibrated Occupancy Fusion and 4-DoF Vehicle Pose for Monocular
               Bird's-Eye-View Mapping of Uncalibrated Street Imagery},
  booktitle = {17th Indian Conference on Computer Vision, Graphics and Image
               Processing (ICVGIP '26)},
  year      = {2026}
}
```

## Licence

Code: see `LICENSE`. Model checkpoints derive from KITTI-trained supervision and inherit
KITTI's non-commercial terms.
