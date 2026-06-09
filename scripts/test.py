#!/usr/bin/env python3
"""Run zero-shot OrienterNet localization for a JSON split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

FIELDNAMES = [
    "split",
    "filename",
    "image_path",
    "building_id",
    "prior_latitude",
    "prior_longitude",
    "gt_latitude",
    "gt_longitude",
    "gt_yaw",
    "pred_latitude",
    "pred_longitude",
    "pred_yaw",
    "error_m",
    "tile_size_meters",
    "num_rotations",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--experiment", default="OrienterNet_MGL")
    parser.add_argument("--tiles", default=None, type=Path)
    parser.add_argument("--tile-size-meters", default=128, type=int)
    parser.add_argument("--num-rotations", default=256, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_split(path: Path, split_name: str) -> list[str]:
    splits = json.loads(path.read_text(encoding="utf-8"))
    if split_name not in splits:
        raise SystemExit(f"Split '{split_name}' not found in {path}.")
    return list(splits[split_name])


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    metadata = {}
    for row in rows:
        filename = row.get("filename") or Path(row.get("image_path", "")).name
        if filename:
            metadata[filename] = row
    return metadata


def build_image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, []).append(path)
    return index


def resolve_image_path(
    split_entry: str, image_root: Path, image_index: dict[str, list[Path]]
) -> Path:
    direct = image_root / split_entry
    if direct.is_file():
        return direct

    candidates = image_index.get(Path(split_entry).name, [])
    if not candidates:
        raise FileNotFoundError(f"Could not find image for split entry: {split_entry}")

    preferred = [p for p in candidates if p.parent.name.lower() != "temp"]
    candidates = preferred or candidates
    if len(candidates) != 1:
        options = ", ".join(str(p) for p in candidates[:5])
        raise RuntimeError(f"Ambiguous image filename '{split_entry}': {options}")
    return candidates[0]


def require_float(row: dict[str, str], keys: tuple[str, ...], filename: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"Missing {keys} for {filename}.")


def optional_float(row: dict[str, str], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def existing_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["filename"] for row in csv.DictReader(f) if row.get("filename")}


def output_row(
    split_name: str,
    filename: str,
    image_path: Path,
    lat: float,
    lon: float,
    yaw: Optional[float],
    tile_size_meters: int,
    num_rotations: int,
) -> dict[str, Any]:
    return {
        "split": split_name,
        "filename": filename,
        "image_path": str(image_path),
        "building_id": image_path.parent.name,
        "prior_latitude": lat,
        "prior_longitude": lon,
        "gt_latitude": lat,
        "gt_longitude": lon,
        "gt_yaw": "" if yaw is None else yaw,
        "pred_latitude": "",
        "pred_longitude": "",
        "pred_yaw": "",
        "error_m": "",
        "tile_size_meters": tile_size_meters,
        "num_rotations": num_rotations,
        "status": "pending",
        "error": "",
    }


def infer_tiles_path(args: argparse.Namespace) -> Optional[Path]:
    if args.tiles is not None:
        return args.tiles

    candidate = args.metadata.parent / "tiles.pkl"
    if candidate.is_file():
        return candidate
    return None


def validate_entries(
    split_entries: list[str],
    metadata: dict[str, dict[str, str]],
    image_root: Path,
    image_index: dict[str, list[Path]],
) -> None:
    counts = Counter()
    errors = []

    for split_entry in split_entries:
        filename = Path(split_entry).name
        if filename not in metadata:
            errors.append(f"No metadata row for {filename}.")
            continue

        try:
            image_path = resolve_image_path(split_entry, image_root, image_index)
            require_float(metadata[filename], ("lat", "latitude"), filename)
            require_float(metadata[filename], ("lon", "longitude"), filename)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
            continue

        counts[image_path.parent.name] += 1

    if errors:
        preview = "\n".join(errors[:10])
        suffix = "" if len(errors) <= 10 else f"\n... and {len(errors) - 10} more"
        raise SystemExit(f"Dry run failed:\n{preview}{suffix}")

    print(
        json.dumps(
            {
                "num_images": len(split_entries),
                "by_building": dict(sorted(counts.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def localize_one(
    demo: Any,
    image_path: Path,
    lat: float,
    lon: float,
    tile_size_meters: int,
    tiler: Optional[Any],
) -> tuple[float, float, float]:
    from maploc.osm.tiling import TileManager
    from maploc.utils.geo import BoundaryBox

    image, camera, gravity, projection, bbox = demo.read_input_image(
        str(image_path),
        prior_latlon=(lat, lon),
        tile_size_meters=tile_size_meters,
    )

    if tiler is None:
        ppm = demo.config.data.pixel_per_meter
        tiler_local = TileManager.from_bbox(projection, bbox + 10, ppm)
        canvas = tiler_local.query(bbox)
        output_projection = projection
    else:
        center = tiler.projection.project([lat, lon])
        bbox = BoundaryBox(center, center) + tile_size_meters
        canvas = tiler.query(bbox)
        output_projection = tiler.projection

    uv, pred_yaw, *_ = demo.localize(image, camera, canvas, gravity=gravity)
    pred_latlon = output_projection.unproject(canvas.to_xy(uv))
    return float(pred_latlon[0]), float(pred_latlon[1]), float(pred_yaw)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards.")

    split_entries = read_split(args.split, args.split_name)
    split_entries = split_entries[args.shard_index :: args.num_shards]
    if args.limit is not None:
        split_entries = split_entries[: args.limit]

    metadata = read_metadata(args.metadata)
    image_index = build_image_index(args.image_root)

    done = existing_done(args.out) if args.resume else set()
    split_entries = [entry for entry in split_entries if Path(entry).name not in done]

    if args.dry_run:
        validate_entries(split_entries, metadata, args.image_root, image_index)
        return

    import torch
    from maploc.demo import Demo
    from maploc.osm.tiling import TileManager

    device = torch.device(args.device) if args.device else None
    demo = Demo(
        experiment_or_path=args.experiment,
        device=device,
        num_rotations=args.num_rotations,
    )

    tiler = None
    tiles_path = infer_tiles_path(args)
    if tiles_path is not None:
        tiler = TileManager.load(tiles_path)
        expected_ppm = demo.config.data.pixel_per_meter
        if tiler.ppm != expected_ppm:
            raise SystemExit(
                f"{tiles_path} has ppm={tiler.ppm}, but model expects "
                f"pixel_per_meter={expected_ppm}."
            )
        print(f"Using cached map tiles: {tiles_path}")
    else:
        print("No tiles.pkl provided or found; OSM tiles will be queried per image.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.out.is_file() else "w"
    write_header = mode == "w" or args.out.stat().st_size == 0

    with args.out.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for split_entry in tqdm(split_entries, desc=args.split_name):
            filename = Path(split_entry).name
            if filename not in metadata:
                raise KeyError(f"No metadata row for {filename}.")

            metadata_row = metadata[filename]
            image_path = resolve_image_path(split_entry, args.image_root, image_index)
            lat = require_float(metadata_row, ("lat", "latitude"), filename)
            lon = require_float(metadata_row, ("lon", "longitude"), filename)
            yaw = optional_float(metadata_row, ("yaw", "heading"))

            row = output_row(
                args.split_name,
                filename,
                image_path,
                lat,
                lon,
                yaw,
                args.tile_size_meters,
                args.num_rotations,
            )

            try:
                pred_lat, pred_lon, pred_yaw = localize_one(
                    demo,
                    image_path,
                    lat,
                    lon,
                    args.tile_size_meters,
                    tiler,
                )
                row.update(
                    {
                        "pred_latitude": pred_lat,
                        "pred_longitude": pred_lon,
                        "pred_yaw": pred_yaw,
                        "error_m": haversine_m(lat, lon, pred_lat, pred_lon),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row.update({"status": "error", "error": repr(exc)})

            writer.writerow(row)
            f.flush()

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
