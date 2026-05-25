# Copyright (c) Meta Platforms, Inc. and affiliates.

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf

from ... import logger, repo_dir
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..torch import collate, worker_init_fn


class GistABCDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "gist_abc",
        "data_dir": repo_dir / "data_one",
        "image_dirname": "dataset",
        "metadata_filename": "metadata_full.csv",
        "intrinsics_filename": "intrinsics.json",
        "tiles_filename": "tiles.pkl",
        "scene": "gist_abc",
        "split": "splits.json",
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.image_root = self.root / self.cfg.image_dirname

    def prepare_data(self):
        required = [
            self.root / self.cfg.metadata_filename,
            self.root / self.cfg.intrinsics_filename,
            self.root / self.cfg.tiles_filename,
            self.image_root,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing GIST ABC training assets:\n  "
                + "\n  ".join(missing)
                + "\nRun: python scripts/prepare_gist_abc.py --build-tiles"
            )

    def setup(self, stage: Optional[str] = None):
        scene = self.cfg.scene
        self.tile_managers = {
            scene: TileManager.load(self.root / self.cfg.tiles_filename)
        }
        self.image_dirs = {scene: self.image_root}

        rows = self._load_rows()
        cameras = self._load_cameras(rows)
        self.splits = self._make_splits(rows)
        rows_by_key = {
            (r["scene"], r["sequence"], r["name"]): r
            for r in rows
        }
        self.data = {
            split: self._pack_data([rows_by_key[tuple(key)] for key in keys], cameras)
            for split, keys in self.splits.items()
        }

    def _load_rows(self):
        image_by_name = {}
        for path in self.image_root.rglob("*"):
            if path.suffix in self.image_extensions:
                image_by_name.setdefault(path.name, path.relative_to(self.image_root))

        projection = self.tile_managers[self.cfg.scene].projection
        rows = []
        metadata_path = self.root / self.cfg.metadata_filename
        with metadata_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                filename = row["filename"]
                rel_image = image_by_name.get(filename)
                if rel_image is None:
                    continue

                lat = float(row["lat"])
                lon = float(row["lon"])
                alt = float(row.get("alt_abs") or row.get("alt_rel") or 0.0)
                xyz = projection.project(np.array([lat, lon, alt]), return_z=True)
                rel_parts = rel_image.parts
                building_id = rel_parts[0] if len(rel_parts) > 1 else "single"
                rows.append(
                    {
                        "scene": self.cfg.scene,
                        "sequence": building_id,
                        "name": str(rel_image),
                        "filename": filename,
                        "building_id": building_id,
                        "gps_position": np.array([lat, lon, alt], dtype=np.float32),
                        "t_c2w": xyz.astype(np.float32),
                        "roll_pitch_yaw": np.array(
                            [
                                float(row.get("roll") or 0.0),
                                float(row.get("pitch") or 0.0),
                                float(row.get("yaw") or 0.0),
                            ],
                            dtype=np.float32,
                        ),
                    }
                )

        if not rows:
            raise ValueError(f"No usable images found under {self.image_root}.")

        logger.info(
            "Loaded %d GIST ABC images from %s using %s.",
            len(rows),
            self.image_root,
            metadata_path,
        )
        return rows

    def _load_cameras(self, rows):
        with (self.root / self.cfg.intrinsics_filename).open("r", encoding="utf-8") as f:
            intrinsics = json.load(f)
        camera = {
            "model": intrinsics.get("model", "PINHOLE"),
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
            "params": np.array(
                [
                    float(intrinsics["fx"]),
                    float(intrinsics["fy"]),
                    float(intrinsics["cx"]),
                    float(intrinsics["cy"]),
                ],
                dtype=np.float32,
            ),
        }
        cameras = {self.cfg.scene: defaultdict(dict)}
        for row in rows:
            cameras[self.cfg.scene][row["sequence"]]["cam0"] = camera
        cameras[self.cfg.scene] = dict(cameras[self.cfg.scene])
        return cameras

    def _pack_data(self, rows, cameras):
        return {
            "camera_id": np.array(["cam0"] * len(rows)),
            "t_c2w": torch.from_numpy(
                np.stack([r["t_c2w"] for r in rows]).astype(np.float32)
            ),
            "roll_pitch_yaw": torch.from_numpy(
                np.stack([r["roll_pitch_yaw"] for r in rows]).astype(np.float32)
            ),
            "gps_position": torch.from_numpy(
                np.stack([r["gps_position"] for r in rows]).astype(np.float32)
            ),
            "cameras": cameras,
            "building_id": np.array([r["building_id"] for r in rows]),
            "filename": np.array([r["filename"] for r in rows]),
        }

    def _make_splits(self, rows):
        keys = [(r["scene"], r["sequence"], r["name"]) for r in rows]
        split_path = self.root / self.cfg.split if self.cfg.split is not None else None
        if split_path is not None and split_path.exists():
            with split_path.open("r", encoding="utf-8") as f:
                split_json = json.load(f)
            by_name = {name: key for key in keys for name in (key[2], Path(key[2]).name)}
            return {
                split: np.array(
                    [by_name[name] for name in names if name in by_name],
                    dtype=object,
                )
                for split, names in split_json.items()
            }

        rng = np.random.RandomState(self.cfg.seed)
        grouped = defaultdict(list)
        for key in keys:
            grouped[key[1]].append(key)

        splits = {"train": [], "val": [], "test": []}
        for group_keys in grouped.values():
            group_keys = list(group_keys)
            rng.shuffle(group_keys)
            n = len(group_keys)
            n_test = int(round(n * self.cfg.test_fraction))
            n_val = int(round(n * self.cfg.val_fraction))
            splits["test"].extend(group_keys[:n_test])
            splits["val"].extend(group_keys[n_test : n_test + n_val])
            splits["train"].extend(group_keys[n_test + n_val :])

        return {k: np.array(v, dtype=object) for k, v in splits.items()}

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext="",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        return torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)
