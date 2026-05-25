# GIST-CV-Team-Project
The original work can be found from the below repository.

https://github.com/facebookresearch/OrienterNet

## GIST ABC fine-tuning

The custom drone dataset is expected under:

```text
data_one/
  metadata_full.csv
  intrinsics.json
  tiles.pkl
  splits_balanced.json
  dataset/
    *.JPG
```

Prepare split and helper metadata files:

```bash
python3 scripts/prepare_gist_abc.py
```

Create a balanced train/val/test split for fine-tuning:

```bash
python3 scripts/make_balanced_splits.py --out data_one/splits_balanced.json
```

Install the dependencies needed for this custom dataset:

```bash
pip install -r requirements/gist_abc.txt
```

If `data_one/tiles.pkl` is missing or stale, rebuild map tiles after installing the
project dependencies:

```bash
python3 scripts/prepare_gist_abc.py --build-tiles
```

Fine-tune with the custom data config:

```bash
python3 -m maploc.train \
  data=gist_abc \
  experiment.name=gist_abc_finetune \
  training.finetune_from_checkpoint=OrienterNet_MGL
```
