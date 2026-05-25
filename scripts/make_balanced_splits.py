#!/usr/bin/env python3
"""Create balanced train/val/test splits for the GIST ABC dataset.

The original split was time ordered, which can put whole pose ranges in val/test.
This script stratifies by coarse XY position, yaw, and pitch bins so each split
sees a similar mix of locations and camera poses.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maploc.osm.tiling import TileManager


def load_rows(data_dir: Path, image_dirname: str):
    image_root = data_dir / image_dirname
    rows = []
    with (data_dir / "metadata_full.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (image_root / row["filename"]).exists():
                continue
            try:
                row = dict(row)
                row["lat"] = float(row["lat"])
                row["lon"] = float(row["lon"])
                row["yaw"] = float(row["yaw"])
                row["pitch"] = float(row["pitch"])
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def quantile_bin(values: np.ndarray, num_bins: int):
    if num_bins <= 1:
        return np.zeros(len(values), dtype=int)
    edges = np.quantile(values, np.linspace(0, 1, num_bins + 1)[1:-1])
    edges = np.unique(edges)
    return np.searchsorted(edges, values, side="right")


def circular_yaw_bin(yaw_deg: np.ndarray, num_bins: int):
    yaw = (yaw_deg + 180.0) % 360.0
    return np.floor(yaw / (360.0 / num_bins)).astype(int).clip(0, num_bins - 1)


def build_strata(rows, tm, xy_bins, yaw_bins, pitch_bins):
    latlon = np.array([[r["lat"], r["lon"]] for r in rows])
    xy = tm.projection.project(latlon)
    x_bin = quantile_bin(xy[:, 0], xy_bins)
    y_bin = quantile_bin(xy[:, 1], xy_bins)
    yaw_bin = circular_yaw_bin(np.array([r["yaw"] for r in rows]), yaw_bins)
    pitch_bin = quantile_bin(np.array([r["pitch"] for r in rows]), pitch_bins)

    strata = defaultdict(list)
    for i, row in enumerate(rows):
        key = (int(x_bin[i]), int(y_bin[i]), int(yaw_bin[i]), int(pitch_bin[i]))
        strata[key].append(row["filename"])
    return strata


def allocate_group(names, rng, train, val):
    names = list(names)
    rng.shuffle(names)
    n = len(names)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))
    elif n == 2:
        n_train, n_val = 1, 0
    elif n == 1:
        n_train, n_val = 1, 0
    return {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }


def rebalance_splits(splits, target_counts, rng, name_to_stratum):
    while any(len(splits[split]) != target_counts[split] for split in splits):
        sources = [
            split for split in splits if len(splits[split]) > target_counts[split]
        ]
        targets = [
            split for split in splits if len(splits[split]) < target_counts[split]
        ]
        if not sources or not targets:
            break

        source = max(sources, key=lambda split: len(splits[split]) - target_counts[split])
        target = max(targets, key=lambda split: target_counts[split] - len(splits[split]))

        source_strata_counts = defaultdict(int)
        for name in splits[source]:
            source_strata_counts[name_to_stratum[name]] += 1

        candidates = [
            i
            for i, name in enumerate(splits[source])
            if source_strata_counts[name_to_stratum[name]] > 1
        ]
        if not candidates:
            candidates = list(range(len(splits[source])))

        idx = int(candidates[rng.randint(len(candidates))])
        splits[target].append(splits[source].pop(idx))
    return splits


def summarize(rows_by_name, splits):
    print({k: len(v) for k, v in splits.items()})
    for split, names in splits.items():
        yaw = np.array([rows_by_name[n]["yaw"] for n in names])
        pitch = np.array([rows_by_name[n]["pitch"] for n in names])
        print(
            f"{split:>5}: yaw mean/std={yaw.mean():7.2f}/{yaw.std():6.2f}, "
            f"pitch mean/std={pitch.mean():6.2f}/{pitch.std():5.2f}, "
            f"pitch>10={np.mean(np.abs(pitch) > 10):6.2%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_one"))
    parser.add_argument("--image-dirname", default="dataset")
    parser.add_argument("--out", type=Path, default=Path("data_one/splits_balanced.json"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xy-bins", type=int, default=4)
    parser.add_argument("--yaw-bins", type=int, default=8)
    parser.add_argument("--pitch-bins", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.data_dir, args.image_dirname)
    rows_by_name = {r["filename"]: r for r in rows}
    tm = TileManager.load(args.data_dir / "tiles.pkl")
    strata = build_strata(rows, tm, args.xy_bins, args.yaw_bins, args.pitch_bins)
    rng = np.random.RandomState(args.seed)

    splits = {"train": [], "val": [], "test": []}
    for names in strata.values():
        part = allocate_group(names, rng, args.train, args.val)
        for split in splits:
            splits[split].extend(part[split])

    n = len(rows)
    target_counts = {
        "train": int(round(n * args.train)),
        "val": int(round(n * args.val)),
        "test": n - int(round(n * args.train)) - int(round(n * args.val)),
    }
    name_to_stratum = {
        name: key for key, names in strata.items() for name in names
    }
    splits = rebalance_splits(splits, target_counts, rng, name_to_stratum)
    for names in splits.values():
        names.sort()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    summarize(rows_by_name, splits)


if __name__ == "__main__":
    main()
