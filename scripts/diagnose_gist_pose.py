#!/usr/bin/env python3
"""Visual diagnostics for GIST ABC GPS/yaw/camera metadata."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap


def load_rows(data_dir: Path, image_dirname: str):
    image_root = data_dir / image_dirname
    rows = []
    with (data_dir / "metadata_full.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = image_root / row["filename"]
            if not path.exists():
                continue
            row = dict(row)
            row["path"] = path
            for key in [
                "lat",
                "lon",
                "roll",
                "pitch",
                "yaw",
                "flight_roll",
                "flight_pitch",
                "flight_yaw",
                "alt_abs",
                "alt_rel",
            ]:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def yaw_to_xy_delta(yaw_deg: np.ndarray):
    # DJI yaw is degrees clockwise from north. Local xy is east/north.
    yaw = np.deg2rad(yaw_deg)
    return np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)


def draw_arrows(ax, uv, yaw, length_px=28, color="red", width=0.0025, label=None):
    dxy = yaw_to_xy_delta(np.asarray(yaw))
    duv = np.stack([dxy[:, 0], -dxy[:, 1]], axis=-1) * length_px
    ax.quiver(
        uv[:, 0],
        uv[:, 1],
        duv[:, 0],
        duv[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=width,
        label=label,
    )


def make_full_map(rows, tm, out: Path, sample_every: int):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)
    latlon = np.array([[r["lat"], r["lon"]] for r in rows])
    xy = tm.projection.project(latlon)
    uv = canvas.to_uv(xy)
    yaw = np.array([r["yaw"] for r in rows])

    idx = np.arange(len(rows))[::sample_every]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb, origin="upper")
    ax.scatter(uv[:, 0], uv[:, 1], s=3, c="black", alpha=0.25)
    draw_arrows(ax, uv[idx], yaw[idx], length_px=32)
    ax.set_title(f"DJI yaw arrows on OSM raster (every {sample_every} frames)")
    ax.set_axis_off()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_variant_map(rows, tm, out: Path, sample_every: int):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)
    latlon = np.array([[r["lat"], r["lon"]] for r in rows])
    xy = tm.projection.project(latlon)
    uv = canvas.to_uv(xy)
    yaw = np.array([r["yaw"] for r in rows])
    idx = np.arange(len(rows))[::sample_every]
    variants = [
        ("yaw", yaw),
        ("-yaw", -yaw),
        ("yaw + 90", yaw + 90),
        ("yaw + 180", yaw + 180),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for ax, (title, values) in zip(axes.flat, variants):
        ax.imshow(rgb, origin="upper")
        ax.scatter(uv[:, 0], uv[:, 1], s=2, c="black", alpha=0.20)
        draw_arrows(ax, uv[idx], values[idx], length_px=30)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Yaw convention candidates on the same GPS points")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def resize_for_panel(image: Image.Image, width: int):
    scale = width / image.width
    return image.resize((width, int(image.height * scale)))


def make_sample_sheet(rows, tm, out: Path, num_samples: int):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)
    latlon = np.array([[r["lat"], r["lon"]] for r in rows])
    xy = tm.projection.project(latlon)
    uv_all = canvas.to_uv(xy)

    indices = np.linspace(0, len(rows) - 1, num_samples, dtype=int)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])

    for ax_img, ax_map, idx in zip(axes[:, 0], axes[:, 1], indices):
        row = rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = resize_for_panel(image, 700)
        ax_img.imshow(image)
        ax_img.set_title(
            f"{row['filename']}\n"
            f"gimbal r/p/y=({row['roll']:.1f}, {row['pitch']:.1f}, {row['yaw']:.1f})  "
            f"flight=({row['flight_roll']:.1f}, {row['flight_pitch']:.1f}, {row['flight_yaw']:.1f})"
        )
        ax_img.set_axis_off()

        uv = uv_all[idx]
        crop = 120
        x0, x1 = int(max(0, uv[0] - crop)), int(min(rgb.shape[1], uv[0] + crop))
        y0, y1 = int(max(0, uv[1] - crop)), int(min(rgb.shape[0], uv[1] + crop))
        ax_map.imshow(rgb[y0:y1, x0:x1], origin="upper")
        local_uv = np.array([[uv[0] - x0, uv[1] - y0]])
        ax_map.scatter(local_uv[:, 0], local_uv[:, 1], s=18, c="black")
        draw_arrows(ax_map, local_uv, np.array([row["yaw"]]), length_px=45)
        ax_map.set_title("OSM crop + DJI yaw arrow")
        ax_map.set_axis_off()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_stats(rows):
    print(f"usable images: {len(rows)}")
    for key in [
        "roll",
        "pitch",
        "yaw",
        "flight_roll",
        "flight_pitch",
        "flight_yaw",
        "alt_abs",
        "alt_rel",
    ]:
        values = np.array([r[key] for r in rows])
        print(
            f"{key:>12}: min={values.min():8.3f} "
            f"max={values.max():8.3f} mean={values.mean():8.3f} "
            f"std={values.std():8.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_one"))
    parser.add_argument("--image-dirname", default="dataset")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/pose_diagnostics"))
    parser.add_argument("--sample-every", type=int, default=12)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    rows = load_rows(args.data_dir, args.image_dirname)
    print_stats(rows)
    tm = TileManager.load(args.data_dir / "tiles.pkl")
    make_full_map(rows, tm, args.out_dir / "yaw_arrows_map.png", args.sample_every)
    make_variant_map(rows, tm, args.out_dir / "yaw_variants_map.png", args.sample_every)
    make_sample_sheet(rows, tm, args.out_dir / "pose_sample_sheet.png", args.num_samples)
    print(f"wrote {args.out_dir / 'yaw_arrows_map.png'}")
    print(f"wrote {args.out_dir / 'yaw_variants_map.png'}")
    print(f"wrote {args.out_dir / 'pose_sample_sheet.png'}")


if __name__ == "__main__":
    main()
