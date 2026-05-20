#!/usr/bin/env python3
"""Sanity-check GIST ABC metadata CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = ["image_path", "latitude", "longitude", "building_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--image-root", default=None, type=Path)
    args = parser.parse_args()

    with args.metadata.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns: {missing}")

        rows = list(reader)

    counts = Counter(r.get("building_id", "UNKNOWN") for r in rows)
    invalid_gps = []
    missing_files = []

    for r in rows:
        try:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                invalid_gps.append(r["image_path"])
        except Exception:
            invalid_gps.append(r["image_path"])

        if args.image_root is not None:
            p1 = args.image_root / r["image_path"]
            p2 = args.image_root.parent / r["image_path"]
            if not p1.exists() and not p2.exists():
                missing_files.append(r["image_path"])

    print("Rows:", len(rows))
    print("Building counts:", dict(counts))
    print("Invalid GPS rows:", len(invalid_gps))
    if invalid_gps[:10]:
        print("  examples:", invalid_gps[:10])
    print("Missing files:", len(missing_files))
    if missing_files[:10]:
        print("  examples:", missing_files[:10])


if __name__ == "__main__":
    main()
