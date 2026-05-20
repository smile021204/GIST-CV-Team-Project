#!/usr/bin/env python3
"""Pack a GPS/yaw image dataset into the Mapillary-style dump expected by maploc.

Input poses format:
  <filename> <lat> <lon> <alt> <roll> <pitch> <yaw>

Alternatively, pass metadata CSV columns:
  image_path, latitude, longitude, yaw
with optional:
  altitude, roll, pitch

The output scene directory contains:
  images/<zero_padded_index>.jpg
  dump.json

The dataset root also gets a split file compatible with MapillaryDataModule:
  splits_GH.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PoseRow:
    filename: str
    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float


WGS84_A = 6378137.0
WGS84_B = 6356752.314245


def ecef_from_lla(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray
) -> tuple[np.ndarray, ...]:
    a2 = WGS84_A**2
    b2 = WGS84_B**2
    lat = np.radians(lat)
    lon = np.radians(lon)
    scale = 1.0 / np.sqrt(a2 * np.cos(lat) ** 2 + b2 * np.sin(lat) ** 2)
    x = (a2 * scale + alt) * np.cos(lat) * np.cos(lon)
    y = (a2 * scale + alt) * np.cos(lat) * np.sin(lon)
    z = (b2 * scale + alt) * np.sin(lat)
    return x, y, z


def ecef_from_topocentric_transform(lat: float, lon: float, alt: float) -> np.ndarray:
    x, y, z = ecef_from_lla(np.array(lat), np.array(lon), np.array(alt))
    sin_lat = np.sin(np.radians(lat))
    cos_lat = np.cos(np.radians(lat))
    sin_lon = np.sin(np.radians(lon))
    cos_lon = np.cos(np.radians(lon))
    return np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon, x],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon, y],
            [0.0, cos_lat, sin_lat, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def project_latlon(latlon: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    lat = latlon[..., 0]
    lon = latlon[..., 1]
    alt = np.zeros_like(lat)
    x, y, z = ecef_from_lla(lat, lon, alt)
    transform = np.linalg.inv(ecef_from_topocentric_transform(ref_lat, ref_lon, 0.0))
    tx = (
        transform[0, 0] * x
        + transform[0, 1] * y
        + transform[0, 2] * z
        + transform[0, 3]
    )
    ty = (
        transform[1, 0] * x
        + transform[1, 1] * y
        + transform[1, 2] * z
        + transform[1, 3]
    )
    return np.stack([tx, ty], axis=-1)


def read_poses(path: Path) -> list[PoseRow]:
    rows: list[PoseRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0].lower() in {"<filename>", "filename"}:
            continue
        if len(parts) != 7:
            raise ValueError(
                f"{path}:{line_no}: expected 7 columns "
                "'filename lat lon alt roll pitch yaw', got {len(parts)}"
            )
        filename, lat, lon, alt, roll, pitch, yaw = parts
        rows.append(
            PoseRow(
                filename=filename,
                lat=float(lat),
                lon=float(lon),
                alt=float(alt),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
            )
        )
    if not rows:
        raise ValueError(f"No poses found in {path}")
    return rows


def read_metadata(path: Path) -> list[PoseRow]:
    rows: list[PoseRow] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = set(reader.fieldnames or [])
        image_col = "image_path" if "image_path" in fieldnames else "filename"
        lat_col = "latitude" if "latitude" in fieldnames else "lat"
        lon_col = "longitude" if "longitude" in fieldnames else "lon"
        required = {image_col, lat_col, lon_col, "yaw"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required CSV columns: {missing}")
        for line_no, row in enumerate(reader, 2):
            try:
                rows.append(
                    PoseRow(
                        filename=row[image_col],
                        lat=float(row[lat_col]),
                        lon=float(row[lon_col]),
                        alt=float(row.get("altitude") or row.get("alt_abs") or 0.0),
                        roll=float(row.get("roll") or 0.0),
                        pitch=float(row.get("pitch") or 0.0),
                        yaw=float(row["yaw"]),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid numeric value") from exc
    if not rows:
        raise ValueError(f"No metadata rows found in {path}")
    return rows


def rotation_c2w_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Invert maploc.data.utils.decompose_rotmat for roll/pitch/yaw in degrees."""
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    def rotx(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def roty(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def rotz(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # Equivalent to scipy Rotation.from_euler("YXZ", [roll, pitch, yaw]).
    r_w2c = roty(roll) @ rotx(pitch) @ rotz(yaw)
    r_cv2xyz = rotx(np.deg2rad(-90.0))
    return (r_cv2xyz.T @ r_w2c).T


def find_image(image_root: Path, filename: str) -> Path:
    path = image_root / filename
    if path.exists():
        return path
    path = image_root.parent / filename
    if path.exists():
        return path
    matches = list(image_root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find image '{filename}' under {image_root}")
    raise ValueError(f"Image filename is ambiguous under {image_root}: {filename}")


def write_training_jpg(src: Path, dst: Path) -> tuple[int, int]:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        with Image.open(src) as image:
            return image.size
    if src.suffix.lower() in {".jpg", ".jpeg"}:
        with Image.open(src) as image:
            width, height = image.size
        shutil.copyfile(src, dst)
        return width, height

    with Image.open(src) as image:
        image = image.convert("RGB")
        width, height = image.size
        image.save(dst, quality=95)
    return width, height


def camera_params(
    width: int,
    height: int,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    focal_scale: float,
) -> list[float]:
    f_default = focal_scale * max(width, height)
    fx_out = fx if fx is not None else f_default
    fy_out = fy if fy is not None else fx_out
    cx_out = cx if cx is not None else width / 2.0
    cy_out = cy if cy is not None else height / 2.0
    return [float(fx_out), float(fy_out), float(cx_out), float(cy_out)]


def split_ids(
    num_items: int, val_fraction: float, test_fraction: float
) -> dict[str, list[int]]:
    ids = list(range(num_items))
    num_test = int(round(num_items * test_fraction))
    num_val = int(round(num_items * val_fraction))
    num_train = max(0, num_items - num_val - num_test)
    if num_items == 1:
        return {"train": ids, "val": ids, "test": []}
    if num_items > 1:
        if num_val == 0 and val_fraction > 0:
            num_val = 1
            num_train = max(1, num_train - 1)
        if num_train == 0:
            num_train = 1
            if num_test > 0:
                num_test -= 1
            elif num_val > 1:
                num_val -= 1
    return {
        "train": ids[:num_train],
        "val": ids[num_train : num_train + num_val],
        "test": ids[num_train + num_val : num_train + num_val + num_test],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--poses", type=Path)
    source.add_argument("--metadata", type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--dataset-root", default=Path("datasets/GH"), type=Path)
    parser.add_argument("--scene", default="gh")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--split-filename", default="splits_GH.json")
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--preserve-names", action="store_true")
    parser.add_argument("--val-fraction", default=0.15, type=float)
    parser.add_argument("--test-fraction", default=0.0, type=float)
    parser.add_argument("--fx", default=None, type=float)
    parser.add_argument("--fy", default=None, type=float)
    parser.add_argument("--cx", default=None, type=float)
    parser.add_argument("--cy", default=None, type=float)
    parser.add_argument(
        "--focal-scale",
        default=1.2,
        type=float,
        help="Used as focal = focal_scale * max(width, height) when fx/fy are absent.",
    )
    args = parser.parse_args()

    if not (0 <= args.val_fraction < 1) or not (0 <= args.test_fraction < 1):
        raise ValueError("Split fractions must be in [0, 1).")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1.")

    rows = read_poses(args.poses) if args.poses else read_metadata(args.metadata)
    latlons = np.array([[row.lat, row.lon] for row in rows], dtype=np.float64)
    ref_lat, ref_lon = ((latlons.min(0) + latlons.max(0)) / 2).tolist()

    scene_dir = args.dataset_root / args.scene
    image_dir = scene_dir / "images"
    sequence = args.sequence or args.scene
    camera_id = "cam0"

    dump: dict[str, dict[str, dict]] = {sequence: {"views": {}, "cameras": {}}}
    camera_signature_to_id: dict[tuple[int, int, tuple[float, ...]], str] = {}
    view_names: list[str] = []

    for index, row in enumerate(rows):
        src = find_image(args.image_root, row.filename)
        view_name = Path(row.filename).stem if args.preserve_names else f"{index:06d}"
        view_names.append(view_name)
        width, height = write_training_jpg(src, image_dir / f"{view_name}{args.image_ext}")
        params = camera_params(
            width, height, args.fx, args.fy, args.cx, args.cy, args.focal_scale
        )
        signature = (width, height, tuple(round(p, 6) for p in params))
        if signature not in camera_signature_to_id:
            cam_id = (
                camera_id
                if not camera_signature_to_id
                else f"cam{len(camera_signature_to_id)}"
            )
            camera_signature_to_id[signature] = cam_id
            dump[sequence]["cameras"][cam_id] = {
                "id": cam_id,
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "params": params,
            }
        cam_id = camera_signature_to_id[signature]
        xy = project_latlon(
            np.array([row.lat, row.lon], dtype=np.float64), ref_lat, ref_lon
        )
        dump[sequence]["views"][view_name] = {
            "camera_id": cam_id,
            "latlong": [row.lat, row.lon],
            "t_c2w": [float(xy[0]), float(xy[1]), row.alt],
            "R_c2w": rotation_c2w_from_rpy(row.roll, row.pitch, row.yaw).tolist(),
            "roll_pitch_yaw": [row.roll, row.pitch, row.yaw],
            "capture_time": index,
            "gps_position": [row.lat, row.lon, row.alt],
            "chunk_id": 0,
            "index": index,
        }

    scene_dir.mkdir(parents=True, exist_ok=True)
    dump_path = scene_dir / "dump.json"
    dump_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")

    splits = split_ids(len(rows), args.val_fraction, args.test_fraction)
    split_values = {
        split: [view_names[i] for i in ids] if args.preserve_names else ids
        for split, ids in splits.items()
    }
    split_out = {
        split: {args.scene: values}
        for split, values in split_values.items()
        if split in {"train", "val"} or values
    }
    split_path = args.dataset_root / args.split_filename
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_out, indent=2), encoding="utf-8")

    print(f"Wrote {dump_path}")
    print(f"Wrote {split_path}")
    print(f"Wrote {len(rows)} images to {image_dir}")
    print({split: len(ids) for split, ids in splits.items()})


if __name__ == "__main__":
    main()
