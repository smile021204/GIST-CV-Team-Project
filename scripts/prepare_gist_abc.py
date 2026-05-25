#!/usr/bin/env python3
"""Prepare the current GIST ABC folder for OrienterNet fine-tuning.

Expected input:
  data_one/metadata_full.csv
  data_one/intrinsics.json
  data_one/dataset/*.JPG

Generated files:
  data_one/metadata.csv
  data_one/poses.txt
  data_one/splits.json
  data_one/tiles.pkl          when --build-tiles is passed
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def scan_images(image_root: Path) -> dict[str, Path]:
    images = {}
    for path in image_root.rglob("*"):
        if path.suffix in IMAGE_EXTENSIONS:
            images.setdefault(path.name, path.relative_to(image_root))
    return images


def load_rows(metadata_path: Path, image_root: Path) -> list[dict[str, str]]:
    images = scan_images(image_root)
    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel_image = images.get(row["filename"])
            if rel_image is None:
                continue
            try:
                float(row["lat"])
                float(row["lon"])
            except ValueError:
                continue
            row = dict(row)
            row["image_path"] = str(rel_image)
            row["latitude"] = row["lat"]
            row["longitude"] = row["lon"]
            row["building_id"] = (
                rel_image.parts[0] if len(rel_image.parts) > 1 else "single"
            )
            row["source_file"] = str(image_root / rel_image)
            rows.append(row)
    return rows


def write_metadata(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "image_path",
        "latitude",
        "longitude",
        "yaw",
        "time",
        "building_id",
        "source_file",
        "filename",
        "width",
        "height",
        "focal_mm",
        "focal_35mm",
        "alt_abs",
        "alt_rel",
        "roll",
        "pitch",
        "flight_roll",
        "flight_pitch",
        "flight_yaw",
        "image_source",
        "drone_model",
        "rtk_flag",
        "rtk_std_lat",
        "rtk_std_lon",
        "rtk_std_hgt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_poses(rows: list[dict[str, str]], path: Path) -> None:
    lines = []
    for row in rows:
        alt = row.get("alt_abs") or row.get("alt_rel") or "0"
        roll = row.get("roll") or "0"
        pitch = row.get("pitch") or "0"
        yaw = row.get("yaw") or "0"
        lines.append(
            " ".join(
                [
                    row["image_path"],
                    row["latitude"],
                    row["longitude"],
                    alt,
                    roll,
                    pitch,
                    yaw,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_splits(
    rows: list[dict[str, str]],
    path: Path,
    train_fraction: float,
    val_fraction: float,
) -> None:
    by_building = defaultdict(list)
    for row in rows:
        by_building[row["building_id"]].append(row)

    splits = {"train": [], "val": [], "test": []}
    for group_rows in by_building.values():
        group_rows = sorted(
            group_rows, key=lambda r: (r.get("time", ""), r.get("image_path", ""))
        )
        n = len(group_rows)
        n_train = int(n * train_fraction)
        n_val = int(n * val_fraction)
        splits["train"].extend(r["image_path"] for r in group_rows[:n_train])
        splits["val"].extend(
            r["image_path"] for r in group_rows[n_train : n_train + n_val]
        )
        splits["test"].extend(r["image_path"] for r in group_rows[n_train + n_val :])

    path.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")


def build_tiles(rows: list[dict[str, str]], out_dir: Path, margin_m: float, ppm: int):
    from maploc.osm.tiling import TileManager
    from maploc.osm.viz import GeoPlotter
    from maploc.utils.geo import BoundaryBox, Projection

    latlon = np.array([[float(r["latitude"]), float(r["longitude"])] for r in rows])
    projection = Projection.from_points(latlon)
    xy = projection.project(latlon)
    bbox = BoundaryBox(xy.min(0), xy.max(0)) + margin_m

    osm_cache = out_dir / "area.osm.json"
    tiles_path = out_dir / "tiles.pkl"
    tile_manager = TileManager.from_bbox(
        projection=projection,
        bbox=bbox,
        ppm=ppm,
        path=osm_cache,
        tile_size=128,
    )
    tile_manager.save(tiles_path)

    plotter = GeoPlotter()
    plotter.points(latlon, "red", name="frames")
    plotter.bbox(projection.unproject(bbox), "blue", "tile bbox")
    plotter.fig.write_html(out_dir / "tiles_preview.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_one"))
    parser.add_argument("--image-dirname", default="dataset")
    parser.add_argument("--metadata", default="metadata_full.csv")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--build-tiles", action="store_true")
    parser.add_argument("--margin-m", type=float, default=128.0)
    parser.add_argument("--ppm", type=int, default=2)
    args = parser.parse_args()

    image_root = args.data_dir / args.image_dirname
    rows = load_rows(args.data_dir / args.metadata, image_root)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.data_dir / args.metadata}")

    write_metadata(rows, args.data_dir / "metadata.csv")
    write_poses(rows, args.data_dir / "poses.txt")
    write_splits(rows, args.data_dir / "splits.json", args.train, args.val)
    if args.build_tiles:
        build_tiles(rows, args.data_dir, args.margin_m, args.ppm)

    counts = defaultdict(int)
    for row in rows:
        counts[row["building_id"]] += 1
    print(f"Prepared {len(rows)} usable images.")
    print(f"Counts by folder: {dict(counts)}")
    print(f"Wrote {args.data_dir / 'metadata.csv'}")
    print(f"Wrote {args.data_dir / 'poses.txt'}")
    print(f"Wrote {args.data_dir / 'splits.json'}")
    if args.build_tiles:
        print(f"Wrote {args.data_dir / 'tiles.pkl'}")
        print(f"Wrote {args.data_dir / 'tiles_preview.html'}")


if __name__ == "__main__":
    main()
