#!/usr/bin/env python3
"""Evaluate zero-shot localization predictions against GT lat/lon."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--recall-thresholds",
        default="1,3,5,10,20,50",
        help="Comma-separated meter thresholds.",
    )
    return parser.parse_args()


def parse_float(row: dict[str, str], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(keys)


def maybe_float(row: dict[str, str], keys: tuple[str, ...]) -> Optional[float]:
    try:
        return parse_float(row, keys)
    except Exception:
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


def angle_error_deg(pred: float, gt: float) -> float:
    error = abs(pred % 360 - gt % 360)
    return min(error, 360 - error)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    thresholds = [float(v) for v in args.recall_thresholds.split(",") if v.strip()]

    with args.predictions.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    errors_m = []
    yaw_errors = []
    per_building: dict[str, list[float]] = defaultdict(list)
    skipped = 0

    for row in rows:
        if row.get("status") and row["status"] != "ok":
            skipped += 1
            continue

        try:
            gt_lat = parse_float(row, ("gt_latitude", "latitude", "lat"))
            gt_lon = parse_float(row, ("gt_longitude", "longitude", "lon"))
            pred_lat = parse_float(row, ("pred_latitude",))
            pred_lon = parse_float(row, ("pred_longitude",))
        except Exception:
            skipped += 1
            continue

        error_m = haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)
        errors_m.append(error_m)
        per_building[row.get("building_id") or "UNKNOWN"].append(error_m)

        gt_yaw = maybe_float(row, ("gt_yaw", "yaw"))
        pred_yaw = maybe_float(row, ("pred_yaw",))
        if gt_yaw is not None and pred_yaw is not None:
            yaw_errors.append(angle_error_deg(pred_yaw, gt_yaw))

    result: dict[str, Any] = {
        "num_rows": len(rows),
        "num_evaluated": len(errors_m),
        "num_skipped": skipped,
        "recall_thresholds_m": thresholds,
    }

    if errors_m:
        result["location_error_m"] = summarize(errors_m)
        result["recall"] = {
            f"within_{threshold:g}m": float(np.mean(np.asarray(errors_m) <= threshold))
            for threshold in thresholds
        }
        result["per_building"] = {
            building_id: {
                "num_evaluated": len(values),
                "location_error_m": summarize(values),
            }
            for building_id, values in sorted(per_building.items())
        }

    if yaw_errors:
        result["yaw_error_deg"] = summarize(yaw_errors)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
