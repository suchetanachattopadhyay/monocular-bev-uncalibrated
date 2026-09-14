# Assembling and pushing this repo

The scaffold ships without the large binaries. Drop them in from `paper_results_lean.zip`,
then push.

## 1. Add the big files locally

```bash
unzip paper_results_lean.zip -d /tmp/lean
cp /tmp/lean/checkpoints/*.pt        checkpoints/
cp -r /tmp/lean/pose_integrated      results/
cp -r /tmp/lean/inference_compare    results/
# crops_for_labeling/ -- see the Mapillary licence note in the chat before including
```

## 2. Git LFS for the checkpoints

`yaw_head.pt` is 44 MB. Under the 100 MB hard limit, over the 50 MB warning, and it
bloats every clone. `.gitattributes` already routes `*.pt` to LFS:

```bash
git lfs install
git add .gitattributes
```

Alternative, if you'd rather not require LFS of anyone cloning: leave `checkpoints/`
out of the tree entirely, attach the four `.pt` files to a GitHub Release, and link
the Release from the README. Cleaner for a reviewer who just wants to read code.

## 3. Initial commit

```bash
git init -b main
git add .
git commit -m "Monocular BEV occupancy fusion and 4-DoF vehicle pose (ICVGIP)"
git remote add origin git@github.com:suchetanachattopadhyay/<repo-name>.git
git push -u origin main
```

## 4. Repo settings worth two minutes

- **About** blurb: one sentence — "Single uncalibrated image -> BEV occupancy grid with
  oriented vehicle footprints. ICVGIP paper code."
- **Topics**: `computer-vision`, `birds-eye-view`, `monocular-depth`, `occupancy-grid`,
  `pose-estimation`, `pytorch`
- Pin the repo on your profile so the link lands on something a manager sees in context.
- Add `paper/` PDF and link it from the About section.
