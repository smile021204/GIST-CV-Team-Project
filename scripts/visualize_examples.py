#!/usr/bin/env python3
"""Render selected good/bad examples with image and GT-vs-pred map crops."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

MAP_COLORS = {
    "building": (84, 155, 255),
    "parking": (255, 229, 145),
    "playground": (150, 133, 125),
    "grass": (188, 255, 143),
    "park": (0, 158, 16),
    "forest": (0, 92, 9),
    "water": (184, 213, 255),
    "fence": (238, 0, 255),
    "wall": (0, 0, 0),
    "hedge": (107, 68, 48),
    "kerb": (255, 234, 0),
    "building_outline": (0, 0, 255),
    "cycleway": (0, 251, 255),
    "path": (8, 237, 0),
    "road": (255, 0, 0),
    "tree_row": (0, 92, 9),
    "busway": (255, 128, 0),
    "void": (230, 230, 230),
}


class BoundaryBox:
    def __init__(self, min_, max_):
        self.min_ = np.asarray(min_, dtype=float)
        self.max_ = np.asarray(max_, dtype=float)

    @classmethod
    def from_string(cls, string: str):
        return cls(*np.split(np.array(string.split(","), float), 2))

    @property
    def size(self) -> np.ndarray:
        return self.max_ - self.min_

    def normalize(self, xy):
        return (xy - self.min_) / (self.max_ - self.min_)

    def __and__(self, other):
        return BoundaryBox(
            np.maximum(self.min_, other.min_), np.minimum(self.max_, other.max_)
        )


class Canvas:
    def __init__(self, bbox: BoundaryBox, ppm: float):
        self.bbox = bbox
        self.ppm = ppm
        self.scaling = bbox.size * ppm
        self.w, self.h = np.ceil(self.scaling).astype(int)
        self.raster = np.zeros((3, self.h, self.w), dtype=np.uint8)

    def to_uv(self, xy: np.ndarray):
        xy = self.bbox.normalize(xy)
        xy[..., 1] = 1 - xy[..., 1]
        return xy * self.scaling - 0.5


class SimpleProjection:
    def __init__(self, lat: float, lon: float):
        self.lat = float(lat)
        self.lon = float(lon)
        self.m_per_deg_lat = 110_540.0
        self.m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(self.lat))

    def project(self, latlon):
        latlon = np.asarray(latlon, dtype=float)
        x = (latlon[..., 1] - self.lon) * self.m_per_deg_lon
        y = (latlon[..., 0] - self.lat) * self.m_per_deg_lat
        return np.stack([x, y], axis=-1)


def bbox_to_slice(bbox: BoundaryBox, canvas: Canvas):
    uv_min = np.ceil(canvas.to_uv(bbox.min_)).astype(int)
    uv_max = np.ceil(canvas.to_uv(bbox.max_)).astype(int)
    return (slice(uv_max[1], uv_min[1]), slice(uv_min[0], uv_max[0]))


class SimpleTileManager:
    def __init__(self, dump):
        self.bbox = BoundaryBox.from_string(dump["bbox"])
        self.ppm = dump["ppm"]
        self.tile_size = dump["tile_size"]
        self.origin = self.bbox.min_
        self.groups = dump["groups"]
        self.projection = SimpleProjection(*dump["ref_latlonalt"][:2])
        self.tiles = {}

        for ij, bbox in dump["tiles_bbox"].items():
            canvas = Canvas(BoundaryBox.from_string(bbox), self.ppm)
            canvas.raster = np.asarray(Image.open(dump["tiles_raster"][ij])).transpose(
                2, 0, 1
            )
            self.tiles[ij] = canvas

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as f:
            return cls(pickle.load(f))

    def query(self, bbox: BoundaryBox):
        canvas = Canvas(bbox, self.ppm)
        bbox_all = bbox & self.bbox
        ij_min = np.floor((bbox_all.min_ - self.origin) / self.tile_size).astype(int)
        ij_max = np.ceil((bbox_all.max_ - self.origin) / self.tile_size).astype(int) - 1

        for i in range(ij_min[0], ij_max[0] + 1):
            for j in range(ij_min[1], ij_max[1] + 1):
                tile = self.tiles.get((i, j))
                if tile is None:
                    continue
                bbox_select = tile.bbox & bbox
                slice_query = bbox_to_slice(bbox_select, canvas)
                slice_tile = bbox_to_slice(bbox_select, tile)
                canvas.raster[(slice(None),) + slice_query] = tile.raster[
                    (slice(None),) + slice_tile
                ]
        return canvas

    def colorize(self, raster):
        area_names = ["void", *self.groups["areas"]]
        way_names = ["void", *self.groups["ways"]]
        area_colors = np.array(
            [MAP_COLORS.get(name, MAP_COLORS["void"]) for name in area_names]
        )
        way_colors = np.array(
            [MAP_COLORS.get(name, MAP_COLORS["void"]) for name in way_names]
        )
        return (
            np.where(
                raster[1, ..., None] > 0,
                way_colors[raster[1]],
                area_colors[raster[0]],
            )
            / 255.0
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--dataset-csv", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--tiles", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--crop-size-m", type=float, default=42.0)
    parser.add_argument("--arrow-len-m", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def yaw_vector(yaw_deg: float, length: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    return np.array([np.sin(yaw_rad), np.cos(yaw_rad)]) * length


def load_examples(examples_path: Path, dataset_path: Path, top_k: int | None):
    examples = pd.read_csv(examples_path)
    dataset = pd.read_csv(dataset_path)

    keep_cols = [
        "image_path",
        "gps_lat",
        "gps_lon",
        "target_lat",
        "target_lon",
        "target_yaw",
    ]
    missing = [col for col in keep_cols if col not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset CSV is missing required columns: {missing}")

    merged = examples.merge(dataset[keep_cols], on="image_path", how="left")
    if merged[["gps_lat", "gps_lon", "target_lat", "target_lon"]].isna().any().any():
        raise ValueError("Some examples did not match dataset rows by image_path.")

    if top_k is not None:
        merged = merged.head(top_k)
    return merged


def draw_map_crop(ax, tile_manager, row, crop_size_m: float, arrow_len_m: float):
    gt_xy = tile_manager.projection.project([row.target_lat, row.target_lon])
    gps_xy = tile_manager.projection.project([row.gps_lat, row.gps_lon])
    pred_xy = gps_xy + np.array([row.pred_delta_x_m, row.pred_delta_y_m])

    center = (gt_xy + pred_xy) / 2.0
    half = crop_size_m / 2.0
    bbox = BoundaryBox(center - half, center + half)
    canvas = tile_manager.query(bbox)
    rgb = tile_manager.colorize(canvas.raster)

    ax.imshow(rgb, origin="upper")

    gt_uv = canvas.to_uv(gt_xy.copy())
    pred_uv = canvas.to_uv(pred_xy.copy())

    ax.plot(
        [gt_uv[0], pred_uv[0]],
        [gt_uv[1], pred_uv[1]],
        color="black",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.scatter(gt_uv[0], gt_uv[1], s=35, c="#36d399", edgecolor="black", label="GT")
    ax.scatter(
        pred_uv[0], pred_uv[1], s=35, c="#ff4d4d", edgecolor="black", label="Pred"
    )

    for xy, yaw, color in (
        (gt_xy, row.target_yaw_deg, "#00a86b"),
        (pred_xy, row.pred_yaw_deg, "#ff4d4d"),
    ):
        if pd.isna(yaw):
            continue
        start = canvas.to_uv(xy.copy())
        end = canvas.to_uv((xy + yaw_vector(yaw, arrow_len_m)).copy())
        ax.arrow(
            start[0],
            start[1],
            end[0] - start[0],
            end[1] - start[1],
            color=color,
            width=0.6,
            head_width=4,
            length_includes_head=True,
        )

    ax.set_title("GT vs prediction crop", fontsize=10)
    ax.set_axis_off()


def draw_image(ax, image_root: Path, row):
    image_path = image_root / row.image_path
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_axis_off()

    loss = getattr(row, "example_score", np.nan)
    loss_text = f", loss={loss:.1f}" if not pd.isna(loss) else ""
    filename = Path(row.image_path).name
    ax.set_title(
        f"{filename}\nxy={row.xy_error_m:.1f}m, "
        f"yaw={row.yaw_error_deg:.1f}deg{loss_text}",
        fontsize=10,
        loc="left",
    )


def main() -> None:
    args = parse_args()
    rows = load_examples(args.examples, args.dataset_csv, args.top_k)
    if len(rows) == 0:
        raise SystemExit("No examples to visualize.")

    tile_manager = SimpleTileManager.load(args.tiles)

    fig_height = max(3.0 * len(rows), 4.0)
    fig, axes = plt.subplots(
        len(rows),
        2,
        figsize=(9.5, fig_height),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28, "hspace": 0.32},
        squeeze=False,
    )

    for i, row in enumerate(rows.itertuples(index=False)):
        draw_image(axes[i, 0], args.image_root, row)
        draw_map_crop(axes[i, 1], tile_manager, row, args.crop_size_m, args.arrow_len_m)
        if i == 0:
            axes[i, 1].legend(loc="lower right", fontsize=8, frameon=True)

    if args.title:
        fig.suptitle(args.title, fontsize=12, y=0.995)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
