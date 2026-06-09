from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf

from ... import DATASETS_PATH
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..torch import collate, worker_init_fn


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class GISTDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "gist",
        "data_dir": DATASETS_PATH,
        "image_root": "datasets_full",
        "metadata": "metadata_full.csv",
        "split": "splits_balanced.json",
        "tiles": "tiles.pkl",
        "intrinsics": "intrinsics.json",
        "target_csv": None,
        "require_target": False,
        "teacher_scores_dir": None,
        "require_teacher_scores": False,
        "scene": "gist_abc",
        "sequence": "single",
        "target_lat_col": "target_lat",
        "target_lon_col": "target_lon",
        "target_yaw_col": "target_yaw",
        "gps_lat_col": "gps_lat",
        "gps_lon_col": "gps_lon",
        "loading": {
            "train": {"batch_size": 2, "num_workers": 4},
            "val": {"batch_size": 2, "num_workers": 4},
            "test": {"batch_size": 1, "num_workers": 0},
        },
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.image_root = self._resolve_path(self.cfg.image_root)
        self.metadata_path = self._resolve_path(self.cfg.metadata)
        self.split_path = self._resolve_path(self.cfg.split)
        self.tiles_path = self._resolve_path(self.cfg.tiles)
        self.intrinsics_path = self._resolve_path(self.cfg.intrinsics)
        if self.cfg.teacher_scores_dir is not None:
            self.cfg.teacher_scores_dir = str(
                self._resolve_path(self.cfg.teacher_scores_dir)
            )

    def _resolve_path(self, path_like) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path
        return self.root / path

    def prepare_data(self):
        for path in [
            self.image_root,
            self.metadata_path,
            self.split_path,
            self.tiles_path,
            self.intrinsics_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(path)
        if self.cfg.teacher_scores_dir is not None:
            teacher_scores_dir = Path(self.cfg.teacher_scores_dir)
            if not teacher_scores_dir.exists():
                raise FileNotFoundError(teacher_scores_dir)

    def setup(self, stage: Optional[str] = None):
        self.tile_managers = {
            self.cfg.scene: TileManager.load(self.tiles_path),
        }
        self._check_tiles()

        self.image_index = self._build_image_index()
        self.metadata = self._read_metadata()
        self.targets = self._read_targets()
        self.camera = self._read_camera()

        split_entries = json.loads(self.split_path.read_text(encoding="utf-8"))
        self.splits = {}
        for split_name, filenames in split_entries.items():
            names = []
            for filename in filenames:
                filename = Path(filename).name
                if filename not in self.metadata:
                    continue
                if filename not in self.image_index:
                    continue
                if self.cfg.require_target and filename not in self.targets:
                    continue
                image_path = str(self.image_index[filename])
                names.append((self.cfg.scene, self.cfg.sequence, image_path))
            self.splits[split_name] = names

        self._pack_data()
        self.image_dirs = {self.cfg.scene: self.image_root}

    def _check_tiles(self):
        tile_manager = self.tile_managers[self.cfg.scene]
        groups = tile_manager.groups
        if self.cfg.num_classes:
            if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                raise ValueError(
                    "Inconsistent map groups: "
                    f"{groups.keys()} vs {self.cfg.num_classes.keys()}"
                )
            for key in groups:
                if len(groups[key]) != self.cfg.num_classes[key]:
                    raise ValueError(
                        f"Inconsistent class count for {key}: "
                        f"{len(groups[key])} vs {self.cfg.num_classes[key]}"
                    )
        if tile_manager.ppm != self.cfg.pixel_per_meter:
            raise ValueError(
                "The tile manager and config use different map resolutions: "
                f"{tile_manager.ppm} vs {self.cfg.pixel_per_meter}"
            )

    def _build_image_index(self) -> dict[str, Path]:
        index = {}
        for path in self.image_root.rglob("*"):
            if not path.is_file() or path.suffix not in IMAGE_EXTS:
                continue
            if "temp" in path.relative_to(self.image_root).parts:
                continue
            rel_path = path.relative_to(self.image_root)
            if path.name in index:
                raise RuntimeError(f"Duplicate image filename: {path.name}")
            index[path.name] = rel_path
        return index

    def _read_metadata(self) -> dict[str, dict[str, Any]]:
        df = pd.read_csv(self.metadata_path)
        required = {"filename", "lat", "lon", "yaw"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")
        df["filename"] = df["filename"].astype(str).apply(lambda x: Path(x).name)
        return {row["filename"]: row for row in df.to_dict("records")}

    def _read_targets(self) -> dict[str, dict[str, Any]]:
        if self.cfg.target_csv is None:
            return {}

        paths = self.cfg.target_csv
        if isinstance(paths, (str, Path)):
            paths = [paths]

        frames = []
        for path in paths:
            frames.append(pd.read_csv(self._resolve_path(path)))
        df = pd.concat(frames, ignore_index=True)

        if "filename" not in df.columns:
            if "image_path" not in df.columns:
                raise ValueError("target_csv must contain filename or image_path.")
            df["filename"] = df["image_path"].astype(str).apply(lambda x: Path(x).name)
        else:
            df["filename"] = df["filename"].astype(str).apply(lambda x: Path(x).name)

        required = {
            self.cfg.target_lat_col,
            self.cfg.target_lon_col,
            self.cfg.target_yaw_col,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"target_csv is missing columns: {sorted(missing)}")

        if df["filename"].duplicated().any():
            duplicated = df[df["filename"].duplicated(keep=False)]
            raise RuntimeError(
                f"Duplicate target rows: {duplicated['filename'].head(10).tolist()}"
            )

        return {row["filename"]: row for row in df.to_dict("records")}

    def _read_camera(self) -> dict[str, Any]:
        intrinsics = json.loads(self.intrinsics_path.read_text(encoding="utf-8"))
        return {
            "model": "PINHOLE",
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
            "params": [
                float(intrinsics["fx"]),
                float(intrinsics["fy"]),
                float(intrinsics["cx"]),
                float(intrinsics["cy"]),
            ],
        }

    def _pack_data(self):
        cameras = {
            self.cfg.scene: {
                self.cfg.sequence: {
                    "camera0": self.camera,
                }
            }
        }
        tile_manager = self.tile_managers[self.cfg.scene]
        self.data = {}

        for split_name, names in self.splits.items():
            camera_id = []
            t_c2w = []
            gps_position = []
            roll_pitch_yaw = []

            for _, _, image_path in names:
                filename = Path(image_path).name
                meta = self.metadata[filename]
                target = self.targets.get(filename)

                gps_lat = self._row_float(
                    target, self.cfg.gps_lat_col, fallback=meta["lat"]
                )
                gps_lon = self._row_float(
                    target, self.cfg.gps_lon_col, fallback=meta["lon"]
                )
                lat = self._row_float(
                    target, self.cfg.target_lat_col, fallback=meta["lat"]
                )
                lon = self._row_float(
                    target, self.cfg.target_lon_col, fallback=meta["lon"]
                )
                yaw = self._row_float(
                    target, self.cfg.target_yaw_col, fallback=meta["yaw"]
                )

                xy = tile_manager.projection.project([lat, lon])
                t_c2w.append([xy[0], xy[1], float(meta.get("alt_rel", 0.0))])
                gps_position.append([gps_lat, gps_lon, float(meta.get("alt_rel", 0.0))])
                roll_pitch_yaw.append(
                    [
                        float(meta.get("roll", 0.0)),
                        float(meta.get("pitch", 0.0)),
                        yaw,
                    ]
                )
                camera_id.append("camera0")

            self.data[split_name] = {
                "camera_id": camera_id,
                "t_c2w": torch.tensor(t_c2w, dtype=torch.float32),
                "gps_position": torch.tensor(gps_position, dtype=torch.float64),
                "roll_pitch_yaw": torch.tensor(roll_pitch_yaw, dtype=torch.float32),
                "cameras": cameras,
                "points": {self.cfg.scene: {self.cfg.sequence: {}}},
            }

    @staticmethod
    def _row_float(row, key, fallback):
        if row is not None and key in row and not pd.isna(row[key]):
            return float(row[key])
        return float(fallback)

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
