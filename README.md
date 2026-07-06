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

## Planning Document

The current research and implementation plan is copied to:

```text
docs/RigFlow4D_plan.md
```

That document is the source of truth for the new model architecture, dataset strategy, training stages, losses, ablations, and paper framing.
