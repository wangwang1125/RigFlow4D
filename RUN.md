# RigFlow4D Run Notes

RigFlow4D currently starts from the MoCapAnything V2 code layout, but the algorithmic path will be replaced according to `docs/RigFlow4D_plan.md`.

## Near-term Implementation Order

1. Normalize dataset schema for AMASS, AIST++, BEDLAM, MVHumanNet++, HuMMan, 3DPW, EMDB, and RICH.
2. Replace DINOv2 image caches with DINOv3-first visual token caches.
3. Implement camera-optional view-time relation encoding.
4. Implement target rig query bank and Skeleton-Peer decoder.
5. Implement rig-native 4D motion field outputs.
6. Implement kinematic VAE and latent flow refinement.
7. Add differentiable Pose2Rot, FK losses, and evaluation scripts.

## Current Status

The copied modules are structural references. Before training, each copied stage should be audited and renamed or replaced so that old animal/multi-species assumptions do not silently leak into the RigFlow4D human pipeline.

## Important Paths

```text
docs/RigFlow4D_plan.md      Research and implementation plan
configs/                    Training and inference configs
data/                       Dataset loaders and schema adapters
models/                     Model components
preprocess/                 Data and visual-token preprocessing
train/                      Training entrypoints
inference/                  Inference entrypoints
utils/                      Shared geometry, rotation, FK, and visualization utilities
datasets/                   Project-local dataset root, ignored by git except datasets/README.md
```

## Project-Local Dataset Layout

Follow the same project-relative style used by MoCapAnything. On both local machines and cloud servers, put training data under the cloned project directory:

```text
RigFlow4D/
  datasets/
    AMASS_archives/
      ACCAD.tar.bz2
      BMLmovi.tar.bz2
      BMLrub.tar.bz2
      CMU.tar.bz2
      ...
    AMASS/
      ACCAD/
      BMLmovi/
      BMLrub/
      CMU/
      ...
    AMASS_RigFlow4D/
      manifest.json
      *.npz
```

`datasets/AMASS/` should keep the official AMASS subset layout after extraction. Do not create extra levels such as `raw/amass/normalized/amass`; the only RigFlow4D-specific output directory is `datasets/AMASS_RigFlow4D/`.

Cloud setup from the project root:

```bash
cd ~/RigFlow4D
mkdir -p datasets/AMASS_archives datasets/AMASS datasets/AMASS_RigFlow4D
cp /path/to/downloaded/*.tar.bz2 datasets/AMASS_archives/
for f in datasets/AMASS_archives/*.tar.bz2; do
  tar -xjf "$f" -C datasets/AMASS
done
```

If the archives are currently staged locally under `G:\dataset`, copy them into the project-local archive folder before uploading or syncing:

```powershell
cd E:\vscode\project\MocapAnything\RigFlow4D
New-Item -ItemType Directory -Force -Path datasets\AMASS_archives,datasets\AMASS,datasets\AMASS_RigFlow4D
Copy-Item G:\dataset\*.tar.bz2 datasets\AMASS_archives\
```

## Dataset Adapter Contract

All dataset adapters should normalize raw samples into `data.schema.RigFlowSample` before caching or batching. Each adapter must call `sample.validate()` so shape mismatches, camera-mode mistakes, and rig-definition inconsistencies are caught before training.

The first implementation target is the Stage 0.5 contract:

```text
raw dataset sample
  -> dataset-specific adapter
  -> RigFlowSample
  -> sample.validate()
  -> cache / batch / train
```

The first concrete adapter is `normalized_npz`, registered by `data.create_default_registry()`. It reads a JSON manifest:

```json
{
  "samples": [
    {
      "sample_id": "sample_0001",
      "path": "sample_0001.npz"
    }
  ]
}
```

Each `.npz` file must contain the required `RigFlowSample` fields:

```text
dataset_name
input_type
source_label_type
camera_mode
parents
rest_offsets
joint_names
chain_ids
chain_coordinates
positions
local_rotations_6d
root_translation
```

Optional keys include `visual_tokens`, `visual_backbone_name`, `visual_feature_dim`, `visual_patch_grid`, `camera_intrinsics`, `camera_extrinsics`, `camera_valid_mask`, `view_ids`, `frame_times`, `observed_time_mask`, and `contact_labels`.

## Motion-Only Converter

The first converter is `preprocess.converters.motion_npz.convert_motion_npz_directory`. It is meant for parsed motion data, not raw AMASS/AIST++ archives. Raw dataset parsers should first decode their source format into a simple motion `.npz` with:

```text
parents
rest_offsets
joint_names
chain_ids
chain_coordinates
positions
root_translation              optional, defaults to zeros
local_rotations_6d            preferred
local_axis_angle              accepted and converted to 6D when local_rotations_6d is absent
contact_labels                optional
```

The converter writes normalized `.npz` files plus `manifest.json`, which can be loaded by the `normalized_npz` adapter.

## Raw AMASS / AIST++ Motion Conversion

`preprocess.converters.raw_motion_capture.convert_raw_motion_capture_directory` is the first raw motion parser. It accepts AMASS-style `poses/trans` and AIST++-style `smpl_poses/smpl_trans` `.npz` files:

```powershell
python -m preprocess.converters.raw_motion_capture `
  --input-dir datasets/AMASS `
  --output-dir datasets/AMASS_RigFlow4D `
  --dataset-name amass `
  --source-format amass
```

This parser does not require SMPL or SMPL-X model assets. It uses the first 24 body pose joints and a default SMPL-like rest skeleton to generate approximate FK `positions`, then reuses the normalized motion converter. These positions are good enough to bootstrap data loading and smoke training; precise AMASS/SMPL-X joints should be added later through an asset-backed parser.

The input directory is scanned recursively, so AMASS archives can be extracted with their original nested structure, for example `datasets/AMASS/ACCAD/s007/*.npz`. Output sample names preserve the relative path with `__` separators to avoid collisions.

## Motion Window Batching

Training code should wrap sequence adapters with `data.MotionWindowDataset` before building motion priors or VAE/flow batches:

```text
NormalizedNpzAdapter
  -> MotionWindowDataset(window_size, stride)
  -> collate_motion_windows
  -> model batch
```

`MotionWindowDataset` slices each validated sequence into fixed temporal windows and emits a `time_mask`. `collate_motion_windows` pads variable joint counts to the largest rig in the batch and emits `joint_mask`, so heterogeneous rigs can share one batch without losing their native joint definitions.

## Kinematic VAE

The first latent-motion model is `models.rigflow4d.KinematicVAE`. It consumes batches emitted by `collate_motion_windows`:

```text
positions [B, T, J, 3]
local_rotations_6d [B, T, J, 6]
time_mask [B, T]
joint_mask [B, J]
```

It returns reconstructed positions/rotations plus `mu`, `logvar`, and sampled `z`. This is the initial `z_kin` contract for the later conditional flow matching refinement module.

## Latent Flow Matching

The first refinement module is `models.rigflow4d.LatentFlowMatcher`. It learns a conditional velocity field in VAE latent space:

```text
z0, z1
  -> sample_latent_flow_pair
  -> z_t, t, target_velocity
  -> LatentFlowMatcher(z_t, t, condition)
  -> predicted latent velocity
```

The current implementation keeps `condition [B, Dcond]` generic. Later stages can pack DINOv3 visual tokens, camera-optional view relations, rig query summaries, deterministic pose seeds, and contact/physics hints into this vector without changing the flow-matching API.

## Condition Encoding

`models.rigflow4d.RigFlowConditionEncoder` is the first bridge from observation/model context into the latent flow condition vector. It accepts:

```text
visual_tokens   [B, V, T, P, Dv]       DINOv3-style dense view-time tokens
camera_features [B, V, Dc] or [B,V,T,Dc] optional calibrated/noisy/learned camera hints
rig_features    [B, J, Dr]             target rig joint features
pose_seed       [B, T, J, Dp]          deterministic seed pose or coarse skeleton features
```

Each modality is mask-pooled and projected to a shared hidden dimension, then fused into `condition [B, Dcond]`. Missing modalities use learned missing tokens, so camera parameters remain optional rather than a hard input requirement.

## Latent Refiner Training Path

`models.rigflow4d.RigFlowLatentRefiner` is the first integrated training module. It connects the current motion prior and conditional flow pieces:

```text
motion batch
  -> KinematicVAE
  -> z1 = mu

DINOv3 / camera hints / rig features / pose seed
  -> RigFlowConditionEncoder
  -> condition

z0 noise, z1 target, condition
  -> LatentFlowMatcher
  -> flow matching loss
```

By default, `z1` is detached from the VAE encoder during flow training. This keeps early training stable: first learn a motion latent space, then learn conditional flow into that space, and only later enable end-to-end coupling if needed.

## Smoke Training Command

Once a normalized `.npz` dataset and manifest exist, run a short CPU smoke train:

```powershell
python train/rigflow4d_latent_refiner.py `
  --data-root datasets/AMASS_RigFlow4D `
  --manifest datasets/AMASS_RigFlow4D/manifest.json `
  --window-size 8 `
  --stride 8 `
  --batch-size 2 `
  --max-steps 2 `
  --visual-dim 1
```

This command validates the current core path:

```text
NormalizedNpzAdapter
  -> MotionWindowDataset
  -> RigFlowLatentRefiner
  -> optimizer step
```

If normalized samples contain `visual_tokens [V,T,P,D]`, set `--visual-dim D` so DINOv3-style dense tokens are passed into the condition encoder. It is intentionally a smoke entry, not the final research training recipe.

## DINOv3-Style Visual Token Cache

`preprocess.visual.token_cache` defines the visual cache format used by normalized samples:

```text
visual_tokens          [V, T, P, D]
visual_backbone_name   string
visual_feature_dim     D
visual_patch_grid      [grid_h, grid_w]
visual_has_cls         0/1
visual_num_registers   int
```

Inject an existing cache into a normalized motion sample:

```powershell
python -m preprocess.visual.token_cache inject `
  --normalized-npz path/to/sample.npz `
  --visual-cache path/to/visual_cache.npz `
  --out path/to/sample_with_visual.npz
```

For real DINOv3 extraction, use the HuggingFace-backed writer. It lazy-loads the model only when encoding starts; on the first run, `from_pretrained(...)` downloads weights into the HuggingFace cache or `--cache-dir`.

```powershell
python -m preprocess.visual.token_cache write-hf `
  --frames-npz path/to/frames.npz `
  --frames-key frames `
  --out path/to/visual_cache.npz `
  --hf-model-id your-dinov3-hf-model-id `
  --feature-dim 1024 `
  --patch-grid 16 16 `
  --cache-dir path/to/hf_cache
```

`frames.npz` should contain `frames [V,T,H,W,C]`. Unit tests use `DeterministicVisualBackbone` so tests do not download weights or require network access.
