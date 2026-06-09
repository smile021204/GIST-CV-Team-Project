"""Evaluate a checkpoint on the custom DJI test set.

Single-image:
    python -m scripts.evaluate_sequential \
        --experiment OrienterNet_MGL \
        --split test --output_dir outputs/eval/mgl_single

Sequential (whole-track refinement):
    python -m scripts.evaluate_sequential \
        --experiment OrienterNet_MGL \
        --split test --sequential --output_dir outputs/eval/mgl_seq

For a fine-tuned checkpoint, pass the .ckpt path as --experiment:
    python -m scripts.evaluate_sequential \
        --experiment datasets/gist_abc/checkpoints/finetune_v1.ckpt \
        --split test --sequential --output_dir outputs/eval/finetune_seq

The script mirrors maploc.evaluation.kitti.run.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from maploc import logger
from maploc.data import CustomDjiDataModule
from maploc.evaluation.run import evaluate


default_cfg_single = OmegaConf.create({})
default_cfg_sequential = OmegaConf.create(
    {
        "data": {
            "mask_radius": CustomDjiDataModule.default_cfg["max_init_error"],
            "prior_range_rotation": (
                CustomDjiDataModule.default_cfg["max_init_error_rotation"] + 1
            ),
            "max_init_error": 0,
            "max_init_error_rotation": 0,
        },
        "chunking": {
            "max_length": 100,
        },
    }
)


def run(
    split: str,
    experiment: str,
    cfg: Optional[DictConfig] = None,
    sequential: bool = False,
    thresholds: Tuple[int, ...] = (1, 3, 5),
    **kwargs,
):
    cfg = cfg or {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = default_cfg_sequential if sequential else default_cfg_single
    cfg = OmegaConf.merge(default, cfg)
    dataset = CustomDjiDataModule(cfg.get("data", {}))

    metrics = evaluate(
        experiment,
        cfg,
        dataset,
        split=split,
        sequential=sequential,
        viz_kwargs=dict(show_dir_error=True, show_masked_prob=False),
        **kwargs,
    )

    keys = ["xy_max_error", "yaw_max_error"]
    if sequential:
        keys += ["xy_seq_error", "yaw_seq_error"]
    for k in keys:
        if k in metrics:
            rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
            logger.info("Recall %s: %s at %s m/°", k, rec, list(thresholds))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val", "train"]
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num", type=int)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    run(
        args.split,
        args.experiment,
        cfg,
        args.sequential,
        output_dir=args.output_dir,
        num=args.num,
    )
