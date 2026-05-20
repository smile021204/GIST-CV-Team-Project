#!/usr/bin/env python3
"""Extract EXIF GPS metadata from GIST ABC images.

Expected input layout:
  datasets/gist_abc/images/A/*.jpg
  datasets/gist_abc/images/B/*.jpg
  datasets/gist_abc/images/C/*.jpg

Output CSV columns:
  image_path,latitude,longitude,yaw,timestamp,building_id,source_file
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ExifTags


GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}
TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def _ratio_to_float(x: Any) -> float:
    try:
        return float(x)
    except TypeError:
        return float(x[0]) / float(x[1])


def _dms_to_deg(dms: Any, ref: str) -> Optional[float]:
    if not dms:
        return None
    deg = _ratio_to_float(dms[0])
    minute = _ratio_to_float(dms[1])
    sec = _ratio_to_float(dms[2])
    val = deg + minute / 60.0 + sec / 3600.0
    if ref in ("S", "W"):
        val = -val
    return val


def read_exif(path: Path) -> dict[str, Any]:
    try:
        image = Image.open(path)
        exif = image.getexif()
    except Exception as exc:
        return {"error": str(exc)}

    out: dict[str, Any] = {}

    dt_tag = TAGS.get("DateTimeOriginal") or TAGS.get("DateTime")
    if dt_tag and dt_tag in exif:
        out["timestamp"] = str(exif.get(dt_tag))
    else:
        out["timestamp"] = ""

    gps_tag = TAGS.get("GPSInfo")
    gps = exif.get_ifd(gps_tag) if gps_tag and gps_tag in exif else {}

    def gps_get(name: str) -> Any:
        tag = GPS_TAGS.get(name)
        return gps.get(tag) if tag is not None else None

    lat = _dms_to_deg(gps_get("GPSLatitude"), gps_get("GPSLatitudeRef"))
    lon = _dms_to_deg(gps_get("GPSLongitude"), gps_get("GPSLongitudeRef"))

    direction = gps_get("GPSImgDirection")
    yaw = _ratio_to_float(direction) if direction is not None else ""

    out.update({
        "latitude": lat if lat is not None else "",
        "longitude": lon if lon is not None else "",
        "yaw": yaw,
    })
    return out


def infer_building_id(image_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(image_root)
        return rel.parts[0]
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"])
    args = parser.parse_args()

    image_root: Path = args.image_root
    paths = sorted([p for p in image_root.rglob("*") if p.suffix in args.extensions])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "latitude",
                "longitude",
                "yaw",
                "timestamp",
                "building_id",
                "source_file",
                "error",
            ],
        )
        writer.writeheader()

        for p in paths:
            exif = read_exif(p)
            row = {
                "image_path": str(p.relative_to(image_root.parent)),
                "latitude": exif.get("latitude", ""),
                "longitude": exif.get("longitude", ""),
                "yaw": exif.get("yaw", ""),
                "timestamp": exif.get("timestamp", ""),
                "building_id": infer_building_id(image_root, p),
                "source_file": str(p),
                "error": exif.get("error", ""),
            }
            writer.writerow(row)

    print(f"Wrote {len(paths)} rows to {args.out}")


if __name__ == "__main__":
    main()
