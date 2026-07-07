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

After converting AMASS into `datasets/AMASS_RigFlow4D/manifest.json`, train the first motion-only kinematic VAE:

```bash
python train/rigflow4d_stage1_vae.py \
  --data-root datasets/AMASS_RigFlow4D \
  --manifest datasets/AMASS_RigFlow4D/manifest.json \
  --output-dir checkpoints/rigflow4d_stage1_vae \
  --window-size 32 \
  --stride 16 \
  --batch-size 64 \
  --max-steps 100000 \
  --lr 1e-4 \
  --latent-dim 128 \
  --hidden-dim 256 \
  --device cuda \
  --num-workers 4
```

The script writes `vae_latest.pt`, `vae_best.pt`, and `metrics.jsonl`. Resume interrupted training with:

```bash
python train/rigflow4d_stage1_vae.py \
  --data-root datasets/AMASS_RigFlow4D \
  --manifest datasets/AMASS_RigFlow4D/manifest.json \
  --output-dir checkpoints/rigflow4d_stage1_vae \
  --resume-from checkpoints/rigflow4d_stage1_vae/vae_latest.pt \
  --device cuda
```

## Planning Document

The current research and implementation plan is copied to:

```text
docs/RigFlow4D_plan.md
```

That document is the source of truth for the new model architecture, dataset strategy, training stages, losses, ablations, and paper framing.
