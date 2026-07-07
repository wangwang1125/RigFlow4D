# RigFlow4D

**RigFlow4D: Camera-Optional Skeleton-Peer 4D Human Motion Capture for Arbitrary Rigs**

This folder is the new implementation workspace for RigFlow4D. Its initial code layout follows the current MoCapAnything V2 repository so that we can reuse familiar training, inference, preprocessing, model, data-loader, and utility boundaries while replacing the core research direction.

## Scope

RigFlow4D targets human motion capture from image or video inputs with:

- DINOv3-first visual tokens.
- Single-view, multi-view, image, and video inputs under one schema.
- Optional camera intrinsics/extrinsics rather than hard camera requirements.
- Skeleton-peer decoding directly on the target rig.
- Rig-native 4D motion fields.
- Kinematic VAE latent space and flow matching refinement.
- Differentiable Pose2Rot and FK supervision.

## Environment

The default environment is kept small for the new RigFlow4D path:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # tests only
```

For GPU training, prefer a cloud image with PyTorch already installed, or install the CUDA-matched PyTorch wheel recommended by the provider before installing `requirements.txt`. Legacy MoCapAnything/TripoSG reference scripts use heavier optional packages in `requirements-legacy.txt`.

## Copied Project Skeleton

The initial structure mirrors the current project:

```text
RigFlow4D/
  assets/
  configs/
  data/
  inference/
  models/
  preprocess/
  train/
  utils/
  docs/
  datasets/   # local/cloud data root, ignored except datasets/README.md
```

The original implementation should be treated as an architectural reference, not as the final algorithm. In particular, the old animal/multi-species assumptions, DINOv2 feature cache, and deterministic pose regression path will be replaced by the RigFlow4D design.

## Dataset Layout

Datasets should live inside the project directory, matching the MoCapAnything convention of project-relative paths:

```text
RigFlow4D/
  datasets/
    AMASS_archives/      # downloaded .tar.bz2 files
    AMASS/               # extracted official AMASS subsets: ACCAD/, CMU/, BMLmovi/, ...
    AMASS_RigFlow4D/     # normalized cache: manifest.json + converted .npz files
```

Do not add extra `raw/amass/normalized/amass` nesting. The AMASS subset folders under `datasets/AMASS/` are the official dataset structure, not additional RigFlow4D layers.

## Stage 1 Training

After converting AMASS into `datasets/AMASS_RigFlow4D/manifest.json`, train the motion-only Temporal-Graph Kinematic VAE:

```bash
python train/rigflow4d_stage1_vae.py --device cuda
```

The defaults are the recommended TG-VAE recipe: `datasets/AMASS_RigFlow4D`, `manifest.json`, `checkpoints/rigflow4d_stage1_tgvae`, 64-frame windows, 256-d latent, 4 layers, 8 heads, and the motion-consistency losses enabled. Stage 1 now trains the body pose in root-relative coordinates while predicting root translation separately, so the model cannot satisfy the loss by preserving only the first pose plus global displacement. For non-standard dataset locations, only pass the paths:

```bash
python train/rigflow4d_stage1_vae.py \
  --data-root /path/to/AMASS_RigFlow4D \
  --manifest manifest.json \
  --device cuda
```

The script writes `vae_latest.pt`, `vae_best.pt`, and `metrics.jsonl`. The main body-pose metric is `*_recon_position` in root-relative coordinates; `*_root_position`, `*_root_velocity`, and `*_absolute_position` are logged separately for diagnosis. Checkpoints trained before this root-relative Stage 1 change should be retrained from scratch. Resume interrupted training with:

```bash
python train/rigflow4d_stage1_vae.py \
  --resume-from checkpoints/rigflow4d_stage1_tgvae/vae_latest.pt \
  --device cuda
```

Visualize Stage 1 reconstruction:

```bash
python inference/visualize_stage1_vae.py --device cuda
```

By default this reads `checkpoints/rigflow4d_stage1_tgvae/vae_best.pt` and writes side-by-side `Input / GT` vs `Encoder + Decoder reconstruction` GIFs under `outputs/visualize_stage1_vae/`. Each GIF frame contains front, side, and top views in one image, with a short motion trail so movement is easier to inspect. The visualizer also defaults to `--selection motion`, which scores candidate windows by joint velocity instead of always picking the first static-looking clips. The `.npz` and `metrics.json` outputs include root translation and root-relative reconstruction errors, which helps separate "root motion is right" from "body motion is right". Use `--selection first`, `--sample-index`, `--trail-frames 0`, or larger `--width/--height` when you want a specific debug view.

## Planning Document

The current research and implementation plan is copied to:

```text
docs/RigFlow4D_plan.md
```

That document is the source of truth for the new model architecture, dataset strategy, training stages, losses, ablations, and paper framing.
