# Skeleton Template Format

RigFlow4D skeleton templates describe the target rig topology used by
`preprocess.converters.raw_motion_capture` when raw pose files are not SMPL24.
They can be stored as JSON or NPZ files.

AMASS SMPL-H pose files with 156-D `poses` do not need an external template:
`--source-format amass` and `--source-format smplh` use RigFlow4D's built-in
52-joint SMPL-H body+hands topology. SMPL-X, Mixamo, UE/MetaHuman, and custom
rigs should provide a template explicitly.

Required fields:

- `parents`: integer array `[J]`. Joint `0` is the root and must have parent `-1`.
  Every other joint must reference an earlier parent index.
- `rest_offsets`: float array `[J, 3]`. Local rest-pose offset from each parent
  to the joint, in the same units as the training positions.
- `joint_names`: string array `[J]`. Names are used for debugging, visualization,
  and future rig adapters.

Optional fields:

- `chain_ids`: integer array `[J]`. If omitted, RigFlow4D derives a simple chain
  id from the first child below the root.
- `chain_coordinates`: float array `[J]`. If omitted, RigFlow4D derives normalized
  parent-chain depth in `[0, 1]`.

Minimal JSON example:

```json
{
  "parents": [-1, 0, 1, 1, 3],
  "rest_offsets": [
    [0.0, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [-0.1, 0.1, 0.0],
    [0.1, 0.1, 0.0],
    [0.2, 0.0, 0.0]
  ],
  "joint_names": ["root", "spine", "left_tip", "right_shoulder", "right_tip"]
}
```

Use the template when converting raw axis-angle pose files:

```bash
python -m preprocess.converters.raw_motion_capture \
  --input-dir datasets/YourRawMotions \
  --output-dir datasets/YourRig_RigFlow4D \
  --dataset-name yourrig \
  --source-format generic \
  --skeleton-template assets/skeleton_templates/your_rig_template.json
```
