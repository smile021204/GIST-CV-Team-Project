# GIST-CV-Team-Project

**Adapting OrienterNet to Academic Campuses via Fine-tuning and Knowledge Distillation**
AI5302 Computer Vision — Final Project, 2026 Spring · Team 21
Gyuheon Kim · Hyeonji Shin · Minsoo Kang

This project builds on **OrienterNet** (Sarlin et al., CVPR 2023).
Original repository: https://github.com/facebookresearch/OrienterNet

---

## 1. Project Goal

We adapt OrienterNet to GIST campus-scale visual localization through two contributions:

1. **Fine-tuning** the pretrained OrienterNet on a custom drone-captured GIST dataset.
2. **Knowledge Distillation** of the fine-tuned teacher into a lightweight ResNet-18/FPN student (Soft KD).

Given a query image and rough GPS prior, the model predicts:

- Camera position `(x, y)` on a 2D OpenStreetMap
- Viewing direction `yaw`

The main scenario is localization around visually similar GIST campus buildings.
This can also be used for **facility repair guidance**: a user reports a repair spot with a photo, a worker takes a current photo nearby, and the system localizes both poses and produces an OSM-routed path to the repair anchor.

---

## 2. Pipeline Overview

```
        ┌──────────────────────────────────┐
        │  Pretrained OrienterNet (MGL)    │
        │  →  "zeroshot" checkpoint        │ ──► §5.1 eval CSV ──┐
        └──────────────┬───────────────────┘                     │
                       │ fine-tune on GIST  (§6.1)               │
                       ▼                                         │
        ┌──────────────────────────────────┐                     │
        │  Fine-tuned OrienterNet          │                     │
        │  →  "teacher" checkpoint         │ ──► §5.2 eval CSV ──┤
        └──────────────┬───────────────────┘                     │
                       │ distillation  (§6.3 → §6.4)             │
                       ▼                                         │
        ┌──────────────────────────────────┐                     │
        │  LightEncoder + Soft KD Student  │                     │
        │  →  "softkd" checkpoint          │ ──► §5.3 eval CSV ──┤
        └──────────────────────────────────┘                     │
                                                                 ▼
                                  ┌──────────────────────────────────┐
                                  │  Navigation Demo (§7)            │
                                  │  • report + worker image pair    │
                                  │  • OSM-based optimal route       │
                                  └──────────────────────────────────┘
```

For reproduction, use §5 (Quick Start). For re-training from scratch, use §6.

The Navigation Demo (§7) is a downstream use of the evaluation CSV.

Sequential mode (§8) is an optional inference mode for the zero-shot and fine-tuned models.

---

## 3. Repository Layout & External Downloads

This code repository:

```
maploc/                  # model, data module, training code
scripts/                 # eval, demo, KD utilities
configs/                 # eval / training configs
requirements/            # dependency lists
MODEL_TEST_COMMANDS.txt  # extra command examples
SUBMISSION_README.md     # submission file manifest
README.md                # this file
```

Datasets and checkpoints are **not included in the code zip**. Download from Google Drive:

| Item | Drive folder | Local target path |
|---|---|---|
| GIST drone dataset | [📁 dataset](https://drive.google.com/drive/folders/14hUNPoCRi6Ds2jaUAEEued9OcLpkBUVD) | `datasets/` (see tree below) |
| Checkpoints (zeroshot, teacher, softkd) | [📁 checkpoints](https://drive.google.com/drive/folders/15GSNhh-Hpazm3ZIQmghaPYAP8Mm9UlBG?usp=sharing) | `checkpoints/` |

The file `datasets/datasets_full.tar.gz` is a compressed archive of the image folder `datasets_full/`.
After downloading it, extract the archive inside the `datasets/` directory:

```bash
tar -xzf datasets/datasets_full.tar.gz -C datasets/
```

After downloading, the layout should look like:

```
datasets/                       # = config `data_dir`
  ├── metadata_full.csv         # config `metadata`  → datasets/metadata_full.csv
  ├── splits_balanced.json      # config `split`     → datasets/splits_balanced.json
  ├── tiles.pkl                 # config `tiles`     → datasets/tiles.pkl
  ├── intrinsics.json           # config `intrinsics`→ datasets/intrinsics.json
  └── datasets_full/            # config `image_root` (images only, searched recursively)
        └── A/, B/, C/          # per-building drone sequences (flat images/ also works)

checkpoints/
  ├── zeroshot/orienternet_mgl.ckpt        # original MGL pretrained (no fine-tuning)
  ├── teacher/checkpoint-epoch=12.ckpt     # fine-tuned full OrienterNet (= KD teacher)
  └── softkd/checkpoint-epoch=76.ckpt      # LightEncoder + Soft KD student
```

> **Note**: the `teacher/` checkpoint **is** the fine-tuned model. It serves both as the §5.2 fine-tuned result and as the KD teacher in §6.

For **Soft KD re-training only** (§6.4), teacher score volumes are also required:

```
outputs/teacher_scores/orienternet_gist/
  ├── train/*.pt
  └── val/*.pt
```

These can be regenerated from the teacher checkpoint via §6.3.

---

## 4. Environment Setup

Tested with Python 3.9.23. Recommended setup with conda:

```bash
conda create -n cv_proj python=3.10 -y
conda activate cv_proj
pip install torch==2.1.0 torchvision==0.16.0
pip install -e .
pip install -r requirements/full.txt
```

All commands below assume:

- Conda environment `cv_proj` is active
- Current working directory is the repository root

---

## 5. Quick Start — Reproduce Reported Numbers

The fastest path: evaluate the three submitted checkpoints on the GIST test split. **No training required.**

### 5.1 Zero-shot Baseline

```bash
python scripts/evaluate.py \
  --experiment checkpoints/zeroshot \
  --checkpoint checkpoints/zeroshot/orienternet_mgl.ckpt \
  --config configs/full_orienternet_gist_eval.yaml \
  --split test --device cuda \
  --out outputs/zeroshot_test_eval.csv
```

### 5.2 Fine-tuned Full OrienterNet (Teacher)

```bash
python scripts/evaluate.py \
  --experiment checkpoints/teacher \
  --checkpoint checkpoints/teacher/checkpoint-epoch=12.ckpt \
  --config configs/full_orienternet_gist_eval.yaml \
  --split test --device cuda \
  --out outputs/teacher_test_eval.csv
```

### 5.3 LightEncoder + Soft KD (Student)

```bash
python scripts/evaluate.py \
  --experiment checkpoints/softkd \
  --checkpoint checkpoints/softkd/checkpoint-epoch=76.ckpt \
  --config configs/lightencoder_kd_eval.yaml \
  --split test --device cuda \
  --out outputs/softkd_test_eval.csv
```

### 5.4 Summarize Metrics

```bash
for name in zeroshot teacher softkd; do
  python scripts/report.py \
    --predictions outputs/${name}_test_eval.csv \
    --out outputs/${name}_test_metrics.json \
    --recall-thresholds 2,5,10,16,20
done
```

Each `*_test_eval.csv` contains per-image fields:
`pred_latitude`, `pred_longitude`, `pred_yaw_max_deg`, `xy_max_error_m`, `yaw_max_error_deg`.
The aggregated `*_test_metrics.json` matches the table reported in our presentation slides.

---

## 6. Full Training Pipeline (Optional)

To re-train from scratch instead of using the provided checkpoints, follow the steps below. Each step requires the previous step's output.

| Step | Stage | Input | Output | Time |
|---|---|---|---|---|
| 6.1 | Fine-tune Teacher | `zeroshot/orienternet_mgl.ckpt` + GIST train+val | `teacher/checkpoint-epoch=XX.ckpt` | ~3 hr |
| 6.2 | Train LightEncoder baseline *(optional)* | GIST train+val | LightEncoder ckpt | ~3 hr |
| 6.3 | Generate teacher score volumes | `teacher/checkpoint-epoch=12.ckpt` + GIST | `outputs/teacher_scores/...` | ~30 min |
| 6.4 | Train LightEncoder + Soft KD | Teacher scores + GIST train+val | `softkd/checkpoint-epoch=XX.ckpt` | ~3 hr |

*Approximate times measured on 1× 32 GB GPU, batch size 8.*

### 6.1 Fine-tune Full OrienterNet (Teacher)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m maploc.train \
  data=gist_abc \
  data.split=splits_balanced.json \
  data.rectify_pitch=false \
  data.max_init_error=16 \
  experiment.name=example_finetune \
  experiment.gpus=1 \
  training.lr=1e-5 \
  training.finetune_from_checkpoint=checkpoints/zeroshot/orienternet_mgl.ckpt \
  +training.trainer.accumulate_grad_batches=4 \
  +training.trainer.max_epochs=40 \
  training.trainer.max_steps=-1
```

The resulting checkpoint serves **both** as the fine-tuned result and as the KD teacher.

### 6.2 Train LightEncoder Baseline *(optional)*

Trains a ResNet-18/FPN OrienterNet directly with NLL only — useful as a non-distilled reference for the student architecture.

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

### 6.3 Generate Teacher Score Volumes (required for §6.4)

```bash
python scripts/dump_teacher_volumes.py \
  --config-name orienternet_resnet18_fpn_gist \
  --checkpoint checkpoints/teacher/checkpoint-epoch=12.ckpt \
  --out-dir outputs/teacher_scores/orienternet_gist \
  --splits train val \
  --num-workers 4
```

### 6.4 Train LightEncoder + Soft KD (Student)

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

After training, evaluate the produced checkpoint via §5.3.

---

## 7. Navigation Demo

The demo consumes an evaluation CSV (from §5) and renders a routed OSM map.
**It does not run localization itself** — the report and worker images must already exist in the evaluation CSV.

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/demo_repair_guidance.py \
  --csv outputs/teacher_test_eval.csv \
  --data-dir datasets \
  --image-dirname datasets_full \
  --report-name DJI_20260514133810_0492_V \
  --worker-name DJI_20260514131623_0069_V \
  --out-dir outputs/repair_guidance_demo/example_report_0492_worker_0069
```

- `--report-name`: filename of the user-reported repair photo
- `--worker-name`: filename of the worker's current photo
- Both filenames must appear in the evaluation CSV (i.e., in the same split that was evaluated)

Outputs:

- `repair_guidance_demo.png` — annotated OSM with route
- `guidance_summary.txt` — text summary

The route avoids OSM building and water cells and prefers path / road / parking cells.

---

## 8. Sequential Localization (Optional)

Sequential localization aggregates pose predictions across consecutive frames for smoother trajectories. It is implemented on a separate branch and can be run optionally with the commands below.

```bash
git checkout seq/HJ
```

Run sequential visualization on the **zero-shot** model (MGL):

```bash
python -m scripts.visualize_sequential \
  --experiment OrienterNet_MGL \
  --split test --stride 1 \
  --crop_size_meters 256 \
  --fixed_search \
  --output outputs/viz_seq/mgl_fixed_256.pdf
```

Run sequential visualization on the **fine-tuned** model:

```bash
python -m scripts.visualize_sequential \
  --experiment datasets/gist_abc/checkpoints/finetune_v1.ckpt \
  --split test --stride 1 \
  --crop_size_meters 256 \
  --fixed_search \
  --output outputs/viz_seq/finetune_fixed_256.pdf
```

Sequential mode is **not used to produce the main quantitative results** in our report — it serves as an optional visualization for smoother pose trajectories.

---

## 9. Implementation Notes

- Use the same `splits_balanced.json` for all comparisons.
- Keep `pixel_per_meter`, crop size, intrinsics, and `tiles.pkl` consistent across teacher / student / zero-shot evaluations.
- For inference, only the student checkpoint is required.
- Teacher score volumes are only required for re-training Soft KD (§6.4).
- The navigation demo (§7) requires an already-generated evaluation CSV.
- For fair comparison, evaluate the teacher and student on the **same test split with the same config**.

---

## 10. Declaration of AI Use

We personally designed the implementation structure, core idea, and methodology, while Codex and Claude were used as coding assistants for script generation.
