#!/usr/bin/env python3
"""Create simple train/val/test splits from metadata.

Default policy:
  - sort by timestamp if available, otherwise by image_path
  - split per building_id to avoid class imbalance

For a stronger evaluation, manually edit the resulting splits.json by building side
or separate walking laps.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    args = parser.parse_args()

    by_building = defaultdict(list)
    with args.metadata.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_building[row.get("building_id", "UNKNOWN")].append(row)

    splits = {"train": [], "val": [], "test": []}
    for building_id, rows in by_building.items():
        rows = sorted(rows, key=lambda r: (r.get("timestamp", ""), r.get("image_path", "")))
        n = len(rows)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        splits["train"].extend(r["image_path"] for r in rows[:n_train])
        splits["val"].extend(r["image_path"] for r in rows[n_train:n_train + n_val])
        splits["test"].extend(r["image_path"] for r in rows[n_train + n_val:])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
