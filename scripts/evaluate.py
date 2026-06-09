#!/usr/bin/env python3
"""Evaluate a GIST ABC checkpoint on val/test without starting training."""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maploc.data import modules as data_modules
from maploc.models.metrics import angle_error, location_error
from maploc.module import GenericModule


def to_float(x):
    return float(x.detach().cpu().item())


def get_first(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return value[0]
    return value


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiments/gist_abc_no_rectify_lr1e-5")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--load-teacher-scores",
        action="store_true",
        help="Load teacher score volumes when evaluating KD loss. Disabled by default.",
    )
    args = parser.parse_args()

    exp = Path(args.experiment)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else exp / "last.ckpt"
    cfg_path = Path(args.config) if args.config else exp / "config.yaml"
    if args.out is None:
        suffix = f"{args.split}_{'all' if args.num is None else args.num}"
        args.out = exp / f"eval_{suffix}.csv"

    cfg = OmegaConf.load(cfg_path)
    if not args.load_teacher_scores and "teacher_scores_dir" in cfg.data:
        cfg.data.teacher_scores_dir = None
        cfg.data.require_teacher_scores = False
    cfg.data.loading[args.split].num_workers = 0
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    data = data_modules[cfg.data.name](cfg.data)
    data.prepare_data()
    data.setup("test")
    loader = data.dataloader(args.split, shuffle=False, num_workers=0)

    model = GenericModule.load_from_checkpoint(
        ckpt_path, map_location=device, strict=True, find_best=False, cfg=cfg
    )
    model.to(device)
    model.eval()

    ppm = cfg.model.pixel_per_meter
    rows = []
    iterator = islice(loader, args.num) if args.num is not None else loader
    total = args.num if args.num is not None else len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(iterator, total=total)):
            batch = model.transfer_batch_to_device(batch, device, batch_idx)
            pred = model(batch)
            losses = model.model.loss(pred, batch)

            xy_max = location_error(pred["uv_max"], batch["uv"], ppm)
            xy_exp = location_error(pred["uv_expectation"], batch["uv"], ppm)
            yaw_max = angle_error(pred["yaw_max"], batch["roll_pitch_yaw"][..., -1])
            yaw_exp = angle_error(
                pred["yaw_expectation"], batch["roll_pitch_yaw"][..., -1]
            )

            scene = get_first(batch["scene"])
            canvas = get_first(batch["canvas"])
            uv_pred = pred["uv_max"][0].detach().cpu().numpy()
            pred_xy = canvas.to_xy(uv_pred)
            pred_latlon = data.tile_managers[scene].projection.unproject(pred_xy)

            rows.append(
                {
                    "index": int(get_first(batch["index"]).detach().cpu().item()),
                    "name": get_first(batch["name"]),
                    "scene": scene,
                    "sequence": get_first(batch["sequence"]),
                    "loss_total": to_float(losses["total"][0]),
                    "xy_max_error_m": to_float(xy_max[0]),
                    "xy_expectation_error_m": to_float(xy_exp[0]),
                    "yaw_max_error_deg": to_float(yaw_max[0]),
                    "yaw_expectation_error_deg": to_float(yaw_exp[0]),
                    "gt_yaw_deg": to_float(batch["roll_pitch_yaw"][0, -1]),
                    "pred_yaw_max_deg": to_float(pred["yaw_max"][0]),
                    "pred_latitude": float(pred_latlon[0]),
                    "pred_longitude": float(pred_latlon[1]),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    metrics = {
        "num_samples": len(rows),
        "loss_total": summarize([r["loss_total"] for r in rows]),
        "xy_max_error_m": summarize([r["xy_max_error_m"] for r in rows]),
        "xy_expectation_error_m": summarize(
            [r["xy_expectation_error_m"] for r in rows]
        ),
        "yaw_max_error_deg": summarize([r["yaw_max_error_deg"] for r in rows]),
        "yaw_expectation_error_deg": summarize(
            [r["yaw_expectation_error_deg"] for r in rows]
        ),
        "xy_recall_2m": float(
            np.mean([r["xy_max_error_m"] <= 2.0 for r in rows])
        )
        if rows
        else 0.0,
        "xy_recall_5m": float(
            np.mean([r["xy_max_error_m"] <= 5.0 for r in rows])
        )
        if rows
        else 0.0,
        "yaw_recall_2deg": float(
            np.mean([r["yaw_max_error_deg"] <= 2.0 for r in rows])
        )
        if rows
        else 0.0,
        "yaw_recall_5deg": float(
            np.mean([r["yaw_max_error_deg"] <= 5.0 for r in rows])
        )
        if rows
        else 0.0,
        "csv": str(args.out),
    }

    print("Evaluation summary")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  config:     {cfg_path}")
    print(f"  split:      {args.split}")
    print(f"  device:     {device}")
    print(f"  samples:    {metrics['num_samples']}")
    for key in [
        "loss_total",
        "xy_max_error_m",
        "xy_expectation_error_m",
        "yaw_max_error_deg",
        "yaw_expectation_error_deg",
    ]:
        s = metrics[key]
        if s:
            print(
                f"  {key}: mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"min={s['min']:.4f}, max={s['max']:.4f}"
            )
    print(f"  xy_recall_2m:    {metrics['xy_recall_2m']:.4f}")
    print(f"  xy_recall_5m:    {metrics['xy_recall_5m']:.4f}")
    print(f"  yaw_recall_2deg: {metrics['yaw_recall_2deg']:.4f}")
    print(f"  yaw_recall_5deg: {metrics['yaw_recall_5deg']:.4f}")
    print(f"  wrote: {args.out}")


if __name__ == "__main__":
    main()
