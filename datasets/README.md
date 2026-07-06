# RigFlow4D Dataset Layout

Keep datasets inside the project directory so cloud machines can train from a single cloned workspace.

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

`AMASS/` is the raw extracted AMASS root. Its child folders are official AMASS subsets. They should not be flattened.

`AMASS_RigFlow4D/` is the normalized cache written by RigFlow4D:

```bash
python -m preprocess.converters.raw_motion_capture \
  --input-dir datasets/AMASS \
  --output-dir datasets/AMASS_RigFlow4D \
  --dataset-name amass \
  --source-format amass
```

The converter scans recursively, so paths such as `datasets/AMASS/ACCAD/s007/*.npz` are expected.
