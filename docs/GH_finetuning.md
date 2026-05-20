# GH Fine-Tuning Data Prep

This branch prepares the project to fine-tune OrienterNet on a GH dataset using
the existing Mapillary data module.

## Expected Layout

The Google Drive zip parts have been consolidated into:

```text
datasets/GH/
  intrinsics.json
  metadata_full.csv
  poses.txt
  splits_GH.json
  gh/
    images/
      DJI_20260514131230_0001_V.JPG
      DJI_20260514131252_0002_V.JPG
    dump.json
    tiles.pkl
```

The current packed dataset contains 760 images: 646 train and 114 validation.

## Pack Images And Poses

Use either a poses text file:

```text
<filename> <lat> <lon> <alt> <roll> <pitch> <yaw>
```

```bash
python3 scripts/pack_mapillary_dump.py \
  --poses datasets/GH/poses.txt \
  --image-root datasets/GH/gh/images \
  --dataset-root datasets/GH \
  --scene gh \
  --preserve-names \
  --image-ext .JPG \
  --fx 2666.6666666666665 \
  --fy 2666.6666666666665 \
  --cx 2000.0 \
  --cy 1500.0
```

Or a metadata CSV with required columns `image_path,latitude,longitude,yaw` and
optional columns `altitude,roll,pitch`:

```bash
python3 scripts/pack_mapillary_dump.py \
  --metadata datasets/gist_abc/metadata.csv \
  --image-root datasets/gist_abc/images \
  --dataset-root datasets/GH \
  --scene gh
```

If real camera intrinsics are available, pass `--fx --fy --cx --cy`. Otherwise
the script uses a simple focal estimate suitable for a first smoke test.

## Build Map Tiles

The downloaded zips already included `datasets/GH/gh/tiles.pkl`. Rebuild it only
if poses or OSM coverage changes:

```bash
python3 scripts/build_tiles.py \
  --poses datasets/GH/poses.txt \
  --out_dir datasets/GH/gh \
  --ppm 2
```

The tile output must be `datasets/GH/gh/tiles.pkl`, matching
`maploc/conf/data/GH.yaml`.

## Fine-Tune

```bash
python -m maploc.train \
  experiment.name=GH_finetune \
  data=GH \
  experiment.gpus=1 \
  training.finetune_from_checkpoint=/path/to/checkpoint.ckpt
```
