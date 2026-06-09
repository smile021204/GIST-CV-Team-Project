#!/usr/bin/env python3
"""Select good and bad localization examples from a per-sample result CSV."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--metric",
        choices=("xy", "yaw", "combined"),
        default="combined",
        help="Metric used to rank good/bad examples.",
    )
    parser.add_argument(
        "--yaw-weight",
        type=float,
        default=0.1,
        help="Combined score = xy_error_m + yaw_weight * yaw_error_deg.",
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def score_results(df: pd.DataFrame, metric: str, yaw_weight: float) -> pd.DataFrame:
    df = df.copy()

    if "xy_error_m" not in df.columns:
        if "error_m" in df.columns:
            df["xy_error_m"] = df["error_m"]
        else:
            raise ValueError("Results CSV must contain xy_error_m or error_m.")

    if metric == "xy":
        df["example_score"] = df["xy_error_m"]
        return df

    if "yaw_error_deg" not in df.columns:
        raise ValueError("Results CSV must contain yaw_error_deg for yaw metrics.")

    if metric == "yaw":
        df["example_score"] = df["yaw_error_deg"]
    else:
        df["example_score"] = df["xy_error_m"] + yaw_weight * df["yaw_error_deg"]

    return df


def copy_example_images(df: pd.DataFrame, image_root: Path, out_dir: Path) -> None:
    image_out = out_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)

    for rank, row in enumerate(df.itertuples(index=False), start=1):
        image_path = Path(str(row.image_path))
        src = image_path if image_path.is_absolute() else image_root / image_path
        if not src.is_file():
            continue

        dst = image_out / f"{rank:03d}_{src.name}"
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results)
    df = score_results(df, args.metric, args.yaw_weight)
    df = df.dropna(subset=["example_score"]).copy()

    if len(df) == 0:
        raise SystemExit("No valid rows to select.")

    good = df.sort_values("example_score", ascending=True).head(args.top_k)
    bad = df.sort_values("example_score", ascending=False).head(args.top_k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    good_path = args.out_dir / "good_examples.csv"
    bad_path = args.out_dir / "bad_examples.csv"
    good.to_csv(good_path, index=False)
    bad.to_csv(bad_path, index=False)

    if args.copy_images:
        if args.image_root is None:
            raise SystemExit("--copy-images requires --image-root.")
        copy_example_images(good, args.image_root, args.out_dir / "good")
        copy_example_images(bad, args.image_root, args.out_dir / "bad")

    print(f"Saved {good_path} ({len(good)} rows)")
    print(f"Saved {bad_path} ({len(bad)} rows)")


if __name__ == "__main__":
    main()
