# Stage 2.1 Visual-Conditioned Latent Flow Design

## Goal

Build the first real visual Stage 2 path for RigFlow4D:

```text
image or video frames
-> DINOv3 dense visual tokens
-> visual-conditioned latent flow
-> Stage 1 topology-guided VAE decoder
-> skeleton motion
```

This stage turns the current motion-only latent-flow trainer into a dataset-backed
visual training path. It should still keep camera parameters optional, so the same
code can train on monocular video, multiview video, calibrated multiview datasets,
or synthetic data where cameras are known.

## Recommended Dataset Order

1. BEDLAM or BEDLAM2 for the first visual pretraining loop.
   - Official BEDLAM: https://bedlam.is.tue.mpg.de/
   - Official BEDLAM2: https://bedlam2.is.tuebingen.mpg.de/
   - Reason: RGB/video observations are paired with SMPL-X style labels, which
     matches RigFlow4D's current SMPL-H/SMPL-X direction better than sparse 3D
     keypoint datasets.
2. AIST++ as a supplemental long-motion dataset.
   - Official page: https://google.github.io/aichoreographer/
   - Reason: strong long-sequence and high-motion coverage, useful for temporal
     stability and rhythm-heavy evaluation, but not the best first source for
     hand/detail supervision.
3. MVHumanNet++ or HuMMan for the multiview 4D main stage.
   - MVHumanNet++: https://kevinlee09.github.io/research/MVHumanNet%2B%2B/
   - MVHumanNet GitHub: https://github.com/kevinlee09/MVHumanNet
   - HuMMan paper: https://arxiv.org/abs/2204.13686
   - HuMMan project page: https://caizhongang.github.io/projects/HuMMan/
   - Reason: they are closer to the final multiview 4D target, but the first
     implementation should avoid spending most effort on heavy dataset cleanup.

## Cloud Dataset Layout

Use project-local datasets so the cloud training folder is self-contained:

```text
RigFlow4D/
  datasets/
    AMASS_RigFlow4D/
    BEDLAM/
      raw/
    BEDLAM_RigFlow4D/
    AISTPP/
      raw/
    AISTPP_RigFlow4D/
    MVHumanNet/
      raw/
    HuMMan/
      raw/
```

The normalized visual-motion dataset should follow the existing manifest pattern:

```text
datasets/BEDLAM_RigFlow4D/
  manifest.json
  samples/
    sequence_000001.npz
    sequence_000002.npz
```

Each normalized sample should be loadable by `data.adapters.normalized_npz`.

## Data Contract

The Stage 2.1 normalized sample extends the existing motion sample with optional
visual fields:

```text
positions              [T, J, 3]
local_rotations_6d     [T, J, 6]
root_translation       [T, 3]
parents                [J]
rest_offsets           [J, 3]
chain_ids              [J]
chain_coordinates      [J]
joint_names            [J]
visual_tokens          optional [V, T, P, D]
visual_backbone_name   optional scalar string
visual_feature_dim     optional scalar int
visual_patch_grid      optional [2]
visual_has_cls         optional scalar int
visual_num_registers   optional scalar int
camera_intrinsics      optional [V, 3, 3]
camera_extrinsics      optional [V, 4, 4]
camera_valid_mask      optional [V] or [V, T]
```

Camera values must be accepted when present but cannot be required by the training
entrypoint. Missing cameras should be represented by the existing missing-token
path in `RigFlowConditionEncoder`.

## DINOv3 Weight Strategy

Use HuggingFace lazy loading at runtime, not committed checkpoints:

```text
AutoImageProcessor.from_pretrained(model_id, cache_dir=...)
AutoModel.from_pretrained(model_id, cache_dir=...)
```

The CLI must accept a `--hf-model-id` and `--cache-dir`. The first cloud run can
download weights into HuggingFace cache; later runs should use the cached files.
No DINOv3 weight file should be stored in the repository.

The implementation should keep the current deterministic backbone for tests, and
use the HuggingFace backbone only when real encoding is requested.

## Architecture

### Component 1: Frame Reader

Add a small visual input reader that can load:

- a directory of frame images,
- a `.npz` containing `frames [V,T,H,W,C]`,
- later, video files through an optional dependency.

The first implementation should prioritize image-frame directories and `.npz`
frames because they are easy to test and cloud-friendly.

### Component 2: DINOv3 Token Cache Writer

Extend `preprocess.visual.token_cache` so real visual preprocessing can do:

```text
frames source -> DINOv3 -> visual_cache.npz
```

The cache writer should record token layout metadata and validate token count and
feature dimension. It should not import `transformers` at module import time.

### Component 3: BEDLAM Visual-Motion Converter

Add a BEDLAM-oriented converter that pairs:

```text
RGB frame sequence + SMPL-X/SMPL-H motion labels
```

into normalized RigFlow4D `.npz` samples. The converter should produce a manifest
compatible with `NormalizedNpzAdapter`.

The first pass may support a generic paired-manifest input to avoid overfitting to
one BEDLAM release archive layout:

```json
{
  "samples": [
    {
      "frames": "datasets/BEDLAM/raw/sequence_001/images",
      "motion": "datasets/BEDLAM/raw/sequence_001/motion.npz",
      "out": "samples/sequence_001.npz"
    }
  ]
}
```

BEDLAM-specific archive parsing can be added after the local downloaded layout is
available.

### Component 4: Stage 2 Visual Training Defaults

Update the Stage 2 training entry so a real visual run can be short:

```text
python train/rigflow4d_latent_refiner.py \
  --data-root datasets/BEDLAM_RigFlow4D \
  --manifest manifest.json \
  --stage1-checkpoint checkpoints/rigflow4d_stage1_tgvae/vae_best.pt \
  --device cuda
```

The trainer should infer `visual_dim` from the first sample when visual tokens are
present, so the user does not need to pass `--visual-dim` manually.

## Training Flow

1. Train Stage 1 VAE on AMASS/SMPL-H or SMPL-X motion.
2. Convert BEDLAM raw labels to normalized RigFlow4D motion samples.
3. Extract DINOv3 dense tokens for the corresponding frames.
4. Inject or write visual fields into the normalized samples.
5. Train Stage 2 latent flow with frozen Stage 1 VAE.
6. Evaluate with held-out BEDLAM samples, then test cross-dataset transfer on
   AIST++ or other video datasets.

## Error Handling

- If visual frame count and motion frame count differ, fail with a file-specific
  message.
- If DINOv3 token dimension does not match metadata, fail before writing cache.
- If `transformers` is missing, only the HuggingFace encoding command should fail.
  Importing the package and running unit tests should still work.
- If camera metadata is absent, training should continue with camera missing-token
  embeddings.
- If a raw dataset sample has unsupported motion labels, skip only when an explicit
  `--skip-invalid` flag is supplied.

## Testing

Add tests for:

- frame directory reader producing deterministic `[V,T,H,W,C]` arrays,
- HuggingFace backbone remaining lazy at construction time,
- visual cache writer preserving metadata,
- paired visual-motion converter producing adapter-loadable samples,
- Stage 2 trainer inferring `visual_dim` from a normalized sample,
- Stage 2 smoke training using `visual_tokens` from the converted dataset.

Use deterministic test frames and tiny synthetic motion samples. Do not require
network access or real DINOv3 weights in tests.

## Out Of Scope For This Stage

- Full MVHumanNet++ and HuMMan archive-specific converters.
- DINOv3 LoRA or end-to-end fine-tuning.
- Online video decoding with heavy dependencies.
- Full camera-aware epipolar or geometric view fusion.
- Rotation/twist final head beyond the current Stage 1 decoder path.

## Acceptance Criteria

- A tiny synthetic visual-motion dataset can be converted to normalized RigFlow4D
  format and loaded by `NormalizedNpzAdapter`.
- The DINOv3/HuggingFace writer can be invoked without repository-stored weights.
- Stage 2 training runs from a visual-token dataset without requiring `--visual-dim`.
- Motion-only Stage 2 behavior remains intact.
- All new tests pass locally without network access.
