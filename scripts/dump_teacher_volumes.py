#!/usr/bin/env python3
"""Dump OrienterNet teacher score volumes for soft distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maploc.data.gist.dataset import GISTDataModule  # noqa: E402
from maploc.module import GenericModule  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config-name", default="orienternet_resnet18_fpn_gist_4gpu")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=None,
        help="Optionally override only teacher model.num_rotations for dump shape.",
    )
    parser.add_argument(
        "--override-model-from-config",
        action="store_true",
        help=(
            "Use the model section from --config-name. By default the checkpoint "
            "model config is preserved and only data config is overridden."
        ),
    )
    parser.add_argument(
        "dotlist",
        nargs="*",
        help="Hydra overrides for the dump data/grid config.",
    )
    return parser.parse_args()


def checkpoint_cfg_override(
    cfg, override_model_from_config: bool, num_rotations: int | None
):
    payload = {"data": cfg.data}
    if override_model_from_config:
        payload["model"] = cfg.model
    elif num_rotations is not None:
        payload["model"] = {"num_rotations": num_rotations}
    return OmegaConf.create(payload)


def main() -> None:
    args = parse_args()
    with initialize(version_base=None, config_path="../maploc/conf"):
        cfg = compose(config_name=args.config_name, overrides=args.dotlist)
    OmegaConf.resolve(cfg)

    cfg.data.loading.train.batch_size = 1
    cfg.data.loading.val.batch_size = 1
    cfg.data.loading.test.batch_size = 1
    cfg.data.loading.train.num_workers = args.num_workers
    cfg.data.loading.val.num_workers = args.num_workers
    cfg.data.loading.test.num_workers = args.num_workers

    model = GenericModule.load_from_checkpoint(
        args.checkpoint,
        cfg=checkpoint_cfg_override(
            cfg, args.override_model_from_config, args.num_rotations
        ),
        strict=True,
        find_best=False,
    ).eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = GISTDataModule(cfg.data)
    dataset.prepare_data()
    dataset.setup()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "score_dtype": "float16",
    }
    torch.save(metadata, args.out_dir / "metadata.pt")

    for split in args.splits:
        loader = dataset.dataloader(split, shuffle=False, num_workers=args.num_workers)
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        iterator = tqdm(loader, desc=f"Dump {split}", total=args.limit or len(loader))
        for idx, batch in enumerate(iterator):
            if args.limit is not None and idx >= args.limit:
                break
            filename = Path(batch["name"][0]).name
            out_path = split_dir / f"{filename}.pt"
            if args.resume and out_path.exists():
                continue
            batch = model.transfer_batch_to_device(batch, device, idx)
            with torch.inference_mode():
                pred = model(batch)
            scores = pred["scores"].detach().cpu().to(torch.float16).squeeze(0)
            dump = {
                "filename": filename,
                "scores": scores,
                "shape": tuple(scores.shape),
                "uv": batch["uv"].detach().cpu().squeeze(0),
                "yaw": batch["roll_pitch_yaw"][..., -1].detach().cpu().squeeze(0),
            }
            torch.save(dump, out_path)


if __name__ == "__main__":
    main()
