"""LightningDataModule for the custom DJI Mavic 3T dataset.

Mirrors KittiDataModule:
  - flat image dir (images/*.JPG)
  - poses.txt with lat/lon/alt + roll/pitch/yaw
  - intrinsics.json (single PINHOLE camera shared by every frame)
  - splits.json with train/val/test lists of bare filenames

Local metric frame is derived from the GPS positions via Projection.from_points
(same recipe used by tools/build_tiles.py). Yaw convention matches KITTI's
get_frame_data: `roll_pitch_yaw = [-roll, -pitch, yaw]`.
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf

from ... import DATASETS_PATH, logger
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn
from .utils import load_intrinsics, load_splits, parse_poses_file


class CustomDjiDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "custom_dji",
        "data_dir": DATASETS_PATH / "gist_abc",
        "image_dirname": "images",
        "poses_filename": "poses.txt",
        "intrinsics_filename": "intrinsics.json",
        "splits_filename": "splits_balanced.json",
        "tiles_filename": "tiles.pkl",
        "image_ext": "",  # filenames in poses.txt already carry the extension
        "loading": {
            "train": {"batch_size": 1, "num_workers": 2},
            "val": {"batch_size": 1, "num_workers": 2},
            "test": {"batch_size": 1, "num_workers": 0},
        },
        # match the fine-tune training config exactly
        "crop_size_meters": 64,
        "max_init_error": 16,
        "max_init_error_rotation": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "resize_image": 512,
        "pad_to_square": True,
        "rectify_pitch": False,
        "target_focal_length": None,
    }
    dummy_scene_name = "gist_abc"
    dummy_seq_name = "all"
    dummy_camera_id = 0

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

        self.splits = {}
        self.data = {}
        self.image_paths = {}
        self._poses = None
        self._camera = None

    def prepare_data(self):
        for fname in [
            self.cfg.poses_filename,
            self.cfg.intrinsics_filename,
            self.cfg.splits_filename,
            self.cfg.tiles_filename,
        ]:
            p = self.root / fname
            if not p.exists():
                raise FileNotFoundError(p)
        if not (self.root / self.cfg.image_dirname).exists():
            raise FileNotFoundError(self.root / self.cfg.image_dirname)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]

        self._poses = parse_poses_file(self.root / self.cfg.poses_filename)
        self._camera = load_intrinsics(self.root / self.cfg.intrinsics_filename)
        splits_all = load_splits(self.root / self.cfg.splits_filename)
        for s in stages:
            if s not in splits_all:
                raise KeyError(f"split '{s}' not in {self.cfg.splits_filename}")
            names = [n for n in splits_all[s] if n in self._poses]
            missing = [n for n in splits_all[s] if n not in self._poses]
            if missing:
                logger.warning(
                    "%d %s entries missing from poses.txt (e.g. %s)",
                    len(missing), s, missing[:3],
                )
            self.splits[s] = names

        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        self.pack_data(stages)

    def pack_data(self, stages):
        for stage in stages:
            t_c2w = []
            roll_pitch_yaw = []
            indices = []
            names = []
            for i, fname in enumerate(self.splits[stage]):
                p = self._poses[fname]
                xy = self.tile_manager.projection.project(p["latlon"])
                t_c2w.append(np.array([xy[0], xy[1], p["alt"]], dtype=np.float32))
                roll_pitch_yaw.append(
                    np.array([-p["roll"], -p["pitch"], p["yaw"]], dtype=np.float32)
                )
                indices.append(i)
                names.append((self.dummy_scene_name, self.dummy_seq_name, fname))

            self.data[stage] = {
                "t_c2w": torch.from_numpy(np.stack(t_c2w)),
                "roll_pitch_yaw": torch.from_numpy(np.stack(roll_pitch_yaw)),
                "index": torch.tensor(indices, dtype=torch.long),
                "camera_id": np.full(len(names), self.dummy_camera_id),
                "cameras": {
                    self.dummy_scene_name: {
                        self.dummy_seq_name: {self.dummy_camera_id: self._camera}
                    }
                },
            }
            self.image_paths[stage] = np.array(names)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root / self.cfg.image_dirname},
            {self.dummy_scene_name: self.tile_manager},
            image_ext=self.cfg.image_ext,
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
        nw = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=nw,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=nw > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.image_paths[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        chunk2indices = {}
        for seq, ids in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], ids, names=self.image_paths[stage], **kwargs
            )
            for i, sub in enumerate(chunks):
                chunk2indices[(seq, i)] = sub
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        return self.dataset(stage), chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
        nw = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=nw,
            shuffle=False,
            pin_memory=True,
            persistent_workers=nw > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, seq_keys, chunk2idx
