#!/usr/bin/env python3
"""Evaluate building-level prediction from predicted points.

Input predictions CSV should contain:
  image_path,building_id,pred_latitude,pred_longitude

If it already contains pred_building_id, this script will use that directly.
Otherwise it assigns pred_building_id by point-in-polygon using building_regions.geojson.

GeoJSON expected:
  FeatureCollection with each Feature:
    properties: {"building_id": "A"}
    geometry: Polygon or MultiPolygon in lon/lat coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def point_in_polygon(lon: float, lat: float, coords: Any) -> bool:
    if not coords:
        return False
    outer = coords[0]
    if not point_in_ring(lon, lat, outer):
        return False
    for hole in coords[1:]:
        if point_in_ring(lon, lat, hole):
            return False
    return True


def assign_building(lon: float, lat: float, regions: list[dict[str, Any]]) -> str:
    for feat in regions:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        bid = props.get("building_id", "")
        if not bid:
            continue

        if geom.get("type") == "Polygon":
            if point_in_polygon(lon, lat, geom.get("coordinates", [])):
                return bid
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                if point_in_polygon(lon, lat, poly):
                    return bid
    return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    regions_geo = json.loads(args.regions.read_text(encoding="utf-8"))
    regions = regions_geo.get("features", [])

    rows = []
    with args.predictions.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt = row.get("building_id") or row.get("gt_building_id")
            pred = row.get("pred_building_id")
            if not pred:
                try:
                    pred = assign_building(float(row["pred_longitude"]), float(row["pred_latitude"]), regions)
                except Exception:
                    pred = "UNKNOWN"
            rows.append({"gt": gt, "pred": pred, "image_path": row.get("image_path", "")})

    total = len(rows)
    correct = sum(1 for r in rows if r["gt"] == r["pred"])
    confusion = defaultdict(Counter)
    for r in rows:
        confusion[r["gt"]][r["pred"]] += 1

    result = {
        "num_samples": total,
        "accuracy": correct / total if total else 0.0,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
