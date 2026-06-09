# GIST-CV-Team-Project

This project is based on **OrienterNet**.

Original repository:
https://github.com/facebookresearch/OrienterNet

## 1. Project Goal

This project fine-tunes OrienterNet for image-based localization on a GIST campus-scale dataset. Given a query image, the model predicts:

* camera location on a 2D map
* viewing direction, yaw

The main target scenario is localization around visually similar GIST campus buildings. The model estimates the camera pose ((x, y, yaw)) by matching an image-derived Bird’s-Eye View representation with a 2D public map.

This can also be used for facility repair guidance. For example, when a user reports a repair spot with a photo, the predicted pose anchors the reported location on the map. A worker can then take another photo nearby, localize their current pose, and receive a map-based route to the estimated repair spot.

## 2. Repository Layout

Keep the following repository-relative paths:

```text
maploc/                         # model, data module, training code
scripts/                        # evaluation, demo, and KD utility scripts
requirements/                   # dependency lists
configs/                         # composed evaluation and training configs
MODEL_TEST_COMMANDS.txt          # extra command examples
SUBMISSION_README.md             # submission file manifest
```

Dataset and checkpoint files are not included in the code submission package. Place them separately under the following paths.

```text
datasets/
  dataset/
    datasets_full/
      A/
      B/
      C/
      metadata_full.csv
      splits_balanced.json
      tiles.pkl
      intrinsics.json

checkpoints/
  teacher/
    checkpoint-epoch=12.ckpt
  softkd/
    checkpoint-epoch=76.ckpt
  zeroshot/
    orienternet_mgl.ckpt
```

All checkpoints can be downloaded from the folling link:

https://drive.google.com/drive/folders/15GSNhh-Hpazm3ZIQmghaPYAP8Mm9UlBG?usp=sharing


For Soft KD retraining, also provide teacher score volumes:

```text
outputs/teacher_scores/orienternet_gist/
```

Expected teacher score volume layout:

```text
outputs/teacher_scores/orienternet_gist/train/*.pt
outputs/teacher_scores/orienternet_gist/val/*.pt
```

## 3. Environment Setup

Recommended Python version:

```text
Python 3.9
```

This code was tested with Python 3.9.23 in the `cv_proj` conda environment.

Use the project conda environment if it already exists:

```bash
conda activate cv_proj
```

If dependencies need to be installed:

```bash
python -m pip install --upgrade pip
pip install -r requirements/full.txt
pip install -e .
```

`pip install -e .` installs this repository in editable mode so that the `maploc` package and command-line modules can be imported from the source tree.

All commands below should be run from the repository root.

## 4. Fine-Tune Full OrienterNet Teacher

This command fine-tunes the full OrienterNet model on the GIST dataset.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m maploc.train \
  data=gist_abc \
  data.split=splits_balanced.json \
  data.rectify_pitch=false \
  data.max_init_error=16 \
  experiment.name=example_finetune \
  experiment.gpus=1 \
  training.lr=1e-5 \
  training.finetune_from_checkpoint=/path/to/checkpoint.ckpt \
  +training.trainer.accumulate_grad_batches=4 \
  +training.trainer.max_epochs=40 \
  training.trainer.max_steps=-1
```

The fine-tuned full OrienterNet checkpoint is used as the **Teacher model** for later knowledge distillation.

## 5. Evaluate Full Fine-Tuned OrienterNet Teacher

After training, evaluate a checkpoint on the validation or test split.

Example:

```bash
python scripts/evaluate_gist_abc_checkpoint.py \
  --experiment experiments/example_finetune \
  --checkpoint experiments/example_finetune/checkpoint-epoch=12.ckpt \
  --split test \
  --device cuda \
  --out experiments/example_finetune/eval_test.csv
```

Alternative evaluation command using the composed config:

```bash
python scripts/evaluate.py \
  --experiment checkpoints/teacher \
  --checkpoint checkpoints/teacher/checkpoint-epoch=12.ckpt \
  --config configs/full_orienternet_gist_eval.yaml \
  --split test \
  --device cuda \
  --out outputs/full_orienternet_test_eval.csv
```

The output CSV contains one localized pose per image:

```text
pred_latitude
pred_longitude
pred_yaw_max_deg
xy_max_error_m
yaw_max_error_deg
```

To summarize the generated CSV:

```bash
python scripts/report.py \
  --predictions outputs/full_orienternet_test_eval.csv \
  --out outputs/full_orienternet_test_metrics.json \
  --recall-thresholds 2,5,10,16,20
```

## 6. Navigation Demo

The navigation demo does not run localization directly from a checkpoint. It uses an evaluation CSV that was already generated from a checkpoint.

The report image and worker image must be selected from the evaluated split. For example, if `--split test` was used during evaluation, both selected images must be present in the test-set CSV.

Example command:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. .venv/bin/python scripts/demo_repair_guidance.py \
  --csv experiments/example_finetune/eval_test.csv \
  --data-dir data_one \
  --image-dirname dataset \
  --report-name DJI_20260514133810_0492_V \
  --worker-name DJI_20260514131623_0069_V \
  --out-dir outputs/repair_guidance_demo/example_report_0492_worker_0069
```

Outputs:

```text
outputs/repair_guidance_demo/example_report_0492_worker_0069/repair_guidance_demo.png
outputs/repair_guidance_demo/example_report_0492_worker_0069/guidance_summary.txt
```

In this demo:

* `report-name` is the user-reported repair photo.
* `worker-name` is the worker's current photo.
* the estimated repair anchor is placed in front of the report photo using the predicted report pose and yaw.
* the route avoids OSM building and water cells and prefers path, road, and parking cells.

## 7. Knowledge Distillation Overview

This project also includes a lightweight OrienterNet student trained with knowledge distillation.

The purpose of knowledge distillation is to transfer the localization behavior of the fine-tuned full OrienterNet Teacher into a lighter Student model. The Student is designed to reduce inference cost while preserving localization performance as much as possible.

The general distillation pipeline is:

```text
Fine-tuned Full OrienterNet Teacher
        ↓
Teacher prediction or teacher score volume generation
        ↓
Lightweight Student training
        ↓
Student evaluation on the same GIST test split
```

Two student directions are supported:

1. **LightEncoder Student**
   A lighter OrienterNet variant using a smaller image encoder such as ResNet18/FPN.

2. **Soft KD Student**
   A lightweight student trained using teacher score volumes from the full fine-tuned OrienterNet Teacher.

## 8. Evaluate LightEncoder + KD Student

This evaluates the submitted lightweight KD checkpoint on the GIST test split.

```bash
python scripts/evaluate.py \
  --experiment checkpoints/softkd \
  --checkpoint checkpoints/softkd/checkpoint-epoch=76.ckpt \
  --config configs/lightencoder_kd_eval.yaml \
  --split test \
  --device cuda \
  --out outputs/lightencoder_kd_test_eval.csv
```

Notes:

* Teacher score volumes are not needed for inference/evaluation.
* `scripts/evaluate.py` disables teacher-score loading by default.
* The output CSV contains per-image prediction and error values.

To summarize the generated CSV:

```bash
python scripts/report.py \
  --predictions outputs/lightencoder_kd_test_eval.csv \
  --out outputs/lightencoder_kd_test_metrics.json \
  --recall-thresholds 2,5,10,16,20
```

## 9. Train LightEncoder Baseline

This trains OrienterNet with a ResNet18/FPN image encoder on the GIST dataset.

```bash
python -m maploc.train \
  --config-name orienternet_resnet18_fpn_gist \
  experiment.name=orienternet_resnet18_fpn_gist \
  experiment.gpus=4 \
  data.loading.train.batch_size=8 \
  data.loading.val.batch_size=8 \
  data.loading.train.num_workers=8 \
  data.loading.val.num_workers=8 \
  +training.trainer.precision=16-mixed \
  training.trainer.val_check_interval=1.0
```

## 10. Generate Teacher Score Volumes

Soft KD training requires teacher score volumes for the train and val splits. Generate them from the fine-tuned full OrienterNet Teacher:

```bash
python scripts/dump_teacher_volumes.py \
  --config-name orienternet_resnet18_fpn_gist \
  --checkpoint checkpoints/teacher/checkpoint-epoch=12.ckpt \
  --out-dir outputs/teacher_scores/orienternet_gist \
  --splits train val \
  --num-workers 4
```

The generated files should be placed at:

```text
outputs/teacher_scores/orienternet_gist/train/*.pt
outputs/teacher_scores/orienternet_gist/val/*.pt
```

## 11. Train LightEncoder + Soft KD

After teacher score volumes are available, train the KD student:

```bash
python -m maploc.train \
  --config-name orienternet_resnet18_fpn_gist_softkd \
  experiment.name=orienternet_resnet18_fpn_gist_softkd \
  experiment.gpus=4 \
  data.teacher_scores_dir=../outputs/teacher_scores/orienternet_gist \
  data.require_teacher_scores=true \
  data.loading.train.batch_size=8 \
  data.loading.val.batch_size=8 \
  data.loading.train.num_workers=8 \
  data.loading.val.num_workers=8 \
  +training.trainer.precision=16-mixed \
  training.trainer.val_check_interval=1.0 \
  training.trainer.max_epochs=80
```

## 12. Sequential Localization

Sequential localization can be used as an optional extension when multiple consecutive frames are available. Instead of estimating each image independently, sequential localization can smooth pose predictions across nearby frames and reduce frame-level instability.

This is optional and not required for the main evaluation.

## 13. Important Notes

* Use the same GIST train/val/test split for all comparisons.
* Use the same image root, metadata, tiles, intrinsics, crop size, and `pixel_per_meter` setting.
* Do not include checkpoints or datasets in the code submission package.
* For inference, only the student checkpoint is required.
* Teacher score volumes are only required for Soft KD retraining.
* The navigation demo requires an already generated evaluation CSV.
* For fair comparison, evaluate the full Teacher and lightweight Student on the same test split.

## 14. Declaration of AI Use

We personally designed the implementation structure, core idea, and methodology, while Codex and Claude were used as coding assistants for script generation.
