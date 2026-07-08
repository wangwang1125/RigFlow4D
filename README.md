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

## Non-SMPL And Custom Skeleton Inputs

RigFlow4D does not require the training skeleton to be SMPL. The normalized cache stores the real rig metadata for each sample: `parents`, `rest_offsets`, `joint_names`, `chain_ids`, and `chain_coordinates`. The Stage 1 TG-VAE batches different joint counts with padding and masks, and conditions the network on topology tokens plus parent-child graph mixing.

There are two supported data paths:

```bash
# 1) Already parsed rig-native motion npz files.
# Each file should contain positions, parents, rest_offsets, joint_names,
# root_translation, and either local_rotations_6d or local_axis_angle.
python -m preprocess.converters.motion_npz \
  --input-dir datasets/YourRigNPZ \
  --output-dir datasets/YourRig_RigFlow4D \
  --dataset-name yourrig
```

```bash
# 2) Raw pose-parameter npz files with an explicit skeleton template.
# Use this for SMPL-X, SMPL-H, Mixamo, UE/MetaHuman, or custom rigs when the
# source file has per-joint axis-angle poses but the topology is not SMPL24.
python -m preprocess.converters.raw_motion_capture \
  --input-dir datasets/YourRawMotions \
  --output-dir datasets/YourRig_RigFlow4D \
  --dataset-name yourrig \
  --source-format generic \
  --skeleton-template assets/skeleton_templates/your_rig_template.json \
  --skip-invalid
```

`--source-format amass` keeps the current dependency-free bootstrap behavior, but it no longer blindly truncates every sequence to 24 joints. If an AMASS file stores the common 156-D SMPL-H pose vector, the converter uses the built-in 52-joint SMPL-H topology, including both hands. If the file only stores 24 body joints, it falls back to SMPL24. `--source-format aistpp` keeps the 24-joint AIST++ path. For `generic` or `smplx`, a skeleton template is required so the converter does not silently truncate back to a hidden standard skeleton. See `assets/skeleton_templates/README.md` for the template schema.

For high-fidelity SMPL-X/SMPL-H supervision, prefer exporting joints, rotations, and topology from the official body model into the rig-native normalized `.npz` contract first. The lightweight template path is intentionally dependency-free, so it can run on cloud machines without installing body-model packages or storing restricted model files in the repository.

## Stage 1 Training

After converting AMASS into `datasets/AMASS_RigFlow4D/manifest.json`, train the motion-only Temporal-Graph Kinematic VAE:

```bash
python train/rigflow4d_stage1_vae.py --device cuda
```

The defaults are the recommended topology-conditioned TG-VAE recipe: `datasets/AMASS_RigFlow4D`, `manifest.json`, `checkpoints/rigflow4d_stage1_tgvae`, 64-frame windows, 256-d latent, 4 layers, 8 heads, topology tokens, parent-child graph mixing, and the motion-consistency losses enabled. Stage 1 trains the body pose in root-relative coordinates while predicting root translation separately, so the model cannot satisfy the loss by preserving only the first pose plus global displacement. For non-standard dataset locations, only pass the paths:

```bash
python train/rigflow4d_stage1_vae.py \
  --data-root /path/to/AMASS_RigFlow4D \
  --manifest manifest.json \
  --device cuda
```

The script writes `vae_latest.pt`, `vae_best.pt`, and `metrics.jsonl`. The main body-pose metric is `*_recon_position` in root-relative coordinates; `*_root_position`, `*_root_velocity`, and `*_absolute_position` are logged separately for diagnosis. The model now conditions on `parents`, `rest_offsets`, and chain coordinates from each rig, while joint-index embedding is disabled by default; use `--use-joint-index-embedding`, `--no-topology-conditioning`, or `--no-graph-mixer` only for ablations. Checkpoints trained before this topology-conditioned Stage 1 change should be retrained from scratch. Resume interrupted training with:

```bash
python train/rigflow4d_stage1_vae.py \
  --resume-from checkpoints/rigflow4d_stage1_tgvae/vae_latest.pt \
  --device cuda
```

Visualize Stage 1 reconstruction:

```bash
python inference/visualize_stage1_vae.py --device cuda
```

By default this reads `checkpoints/rigflow4d_stage1_tgvae/vae_best.pt` and writes side-by-side `Input / GT` vs `Encoder + Decoder reconstruction` GIFs under `outputs/visualize_stage1_vae/`. It uses the same `val_fraction` and `seed` stored in the checkpoint and defaults to `--split val`, so the automatic previews come from held-out windows rather than the optimizer's training windows. Use `--split train` only when you explicitly want to inspect overfitting, or `--split all` for broad dataset browsing.

Each GIF frame contains front, side, and top views in one image, with a short motion trail so movement is easier to inspect. The visualizer also defaults to `--selection motion`, which scores candidate windows by joint velocity instead of always picking the first static-looking clips. The `.npz` and `metrics.json` outputs include root translation, root-relative reconstruction errors, `joint_count`, and `joint_names_head`. If the preview still shows 24 points, check `outputs/visualize_stage1_vae/metrics.json`: `joint_count=24` means the normalized dataset or checkpoint was produced from 24-joint data; delete `datasets/AMASS_RigFlow4D`, reconvert AMASS with the current converter, and retrain. Use `--selection first`, `--sample-index`, `--trail-frames 0`, or larger `--width/--height` when you want a specific debug view.

## Stage 2 Latent Flow

After Stage 1 has produced `checkpoints/rigflow4d_stage1_tgvae/vae_best.pt`, train the first formal latent flow refiner:

```bash
python train/rigflow4d_latent_refiner.py --device cuda
```

This loads the Stage 1 VAE checkpoint, freezes the VAE by default, encodes each motion window into the rig-native latent target `z1`, and trains a conditional flow matcher from noise `z0` to that latent. The current condition path uses a pose seed derived from root-relative positions and 6D rotations, plus optional cached visual tokens when the normalized dataset contains them. The script writes `flow_latest.pt`, `flow_best.pt`, and `metrics.jsonl` under `checkpoints/rigflow4d_stage2_latent_flow/`.

For custom paths:

```bash
python train/rigflow4d_latent_refiner.py \
  --data-root /path/to/AMASS_RigFlow4D \
  --manifest manifest.json \
  --stage1-checkpoint /path/to/vae_best.pt \
  --device cuda
```

## Planning Document

The current research and implementation plan is copied to:

```text
docs/RigFlow4D_plan.md
```

That document is the source of truth for the new model architecture, dataset strategy, training stages, losses, ablations, and paper framing.
