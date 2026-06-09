#!/usr/bin/env python3
"""Visualize GIST ABC evaluation CSV outputs."""

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


def parse_float(row, key):
    return float(row[key])


def load_metadata(data_dir: Path):
    metadata = {}
    with (data_dir / "metadata_full.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            for key in [
                "lat",
                "lon",
                "roll",
                "pitch",
                "yaw",
                "flight_roll",
                "flight_pitch",
                "flight_yaw",
            ]:
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            metadata[row["filename"]] = row
    return metadata


def load_eval(csv_path: Path, metadata):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            meta = metadata[row["name"]]
            row["gt_lat"] = meta["lat"]
            row["gt_lon"] = meta["lon"]
            row["pitch"] = meta["pitch"]
            row["roll"] = meta["roll"]
            for key in [
                "loss_total",
                "xy_max_error_m",
                "xy_expectation_error_m",
                "yaw_max_error_deg",
                "yaw_expectation_error_deg",
                "gt_yaw_deg",
                "pred_yaw_max_deg",
                "pred_latitude",
                "pred_longitude",
            ]:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def yaw_to_uv_delta(yaw_deg, length_px):
    yaw = np.deg2rad(yaw_deg)
    dxy = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)
    return np.stack([dxy[:, 0], -dxy[:, 1]], axis=-1) * length_px


def latlon_to_uv(tm, canvas, latlon):
    xy = tm.projection.project(np.asarray(latlon))
    return canvas.to_uv(xy)


def add_yaw_arrows(ax, uv, yaw, length_px, color, label=None, width=0.004):
    if len(uv) == 0:
        return
    delta = yaw_to_uv_delta(np.asarray(yaw), length_px)
    ax.quiver(
        uv[:, 0],
        uv[:, 1],
        delta[:, 0],
        delta[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=width,
        label=label,
    )


def make_error_map(rows, tm, out):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)

    gt_latlon = np.array([[r["gt_lat"], r["gt_lon"]] for r in rows])
    pred_latlon = np.array([[r["pred_latitude"], r["pred_longitude"]] for r in rows])
    gt_uv = latlon_to_uv(tm, canvas, gt_latlon)
    pred_uv = latlon_to_uv(tm, canvas, pred_latlon)
    xy_err = np.array([r["xy_max_error_m"] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb, origin="upper")
    norm = plt.Normalize(vmin=0, vmax=max(20, float(np.percentile(xy_err, 95))))
    cmap = plt.get_cmap("magma")
    for a, b, err in zip(gt_uv, pred_uv, xy_err):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(norm(err)), alpha=0.55, lw=1.1)

    ax.scatter(gt_uv[:, 0], gt_uv[:, 1], s=18, c="#20c997", ec="black", lw=0.25, label="GT")
    sc = ax.scatter(
        pred_uv[:, 0],
        pred_uv[:, 1],
        s=28,
        c=xy_err,
        cmap=cmap,
        norm=norm,
        ec="white",
        lw=0.35,
        label="Pred",
    )
    worst = np.argsort(-xy_err)[: min(12, len(rows))]
    add_yaw_arrows(
        ax,
        gt_uv[worst],
        np.array([rows[i]["gt_yaw_deg"] for i in worst]),
        length_px=26,
        color="#20c997",
        label="GT yaw",
        width=0.003,
    )
    add_yaw_arrows(
        ax,
        pred_uv[worst],
        np.array([rows[i]["pred_yaw_max_deg"] for i in worst]),
        length_px=26,
        color="#ff4d4f",
        label="Pred yaw",
        width=0.003,
    )
    ax.set_title("GIST ABC test predictions on OSM map")
    ax.set_axis_off()
    ax.legend(loc="lower right", frameon=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("XY error (m)")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_histograms(rows, out):
    values = {
        "XY max error (m)": np.array([r["xy_max_error_m"] for r in rows]),
        "XY expectation error (m)": np.array([r["xy_expectation_error_m"] for r in rows]),
        "Yaw max error (deg)": np.array([r["yaw_max_error_deg"] for r in rows]),
        "Loss": np.array([r["loss_total"] for r in rows]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, arr) in zip(axes.flat, values.items()):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(finite, bins=18, color="#3b82f6", alpha=0.78, edgecolor="white")
            ax.axvline(
                np.mean(finite),
                color="#ef4444",
                lw=2,
                label=f"mean {np.mean(finite):.2f}",
            )
            ax.axvline(
                np.median(finite),
                color="#111827",
                lw=2,
                ls="--",
                label=f"median {np.median(finite):.2f}",
            )
            ax.legend()
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_recall_curves(rows, out):
    xy = np.array([r["xy_max_error_m"] for r in rows])
    yaw = np.array([r["yaw_max_error_deg"] for r in rows])
    xy = xy[np.isfinite(xy)]
    yaw = yaw[np.isfinite(yaw)]
    xy_thr = np.linspace(0, 40, 161)
    yaw_thr = np.linspace(0, 180, 181)
    xy_rec = np.array([np.mean(xy <= t) for t in xy_thr]) if xy.size else np.zeros_like(xy_thr)
    yaw_rec = np.array([np.mean(yaw <= t) for t in yaw_thr]) if yaw.size else np.zeros_like(yaw_thr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(xy_thr, xy_rec * 100, color="#2563eb", lw=2)
    for t in [2, 5, 10, 16, 20]:
        axes[0].axvline(t, color="#9ca3af", lw=0.8, alpha=0.7)
    axes[0].set_title("XY recall curve")
    axes[0].set_xlabel("threshold (m)")
    axes[0].set_ylabel("recall (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.25)

    axes[1].plot(yaw_thr, yaw_rec * 100, color="#7c3aed", lw=2)
    for t in [2, 5, 10, 20, 45, 90]:
        axes[1].axvline(t, color="#9ca3af", lw=0.8, alpha=0.7)
    axes[1].set_title("Yaw recall curve")
    axes[1].set_xlabel("threshold (deg)")
    axes[1].set_ylabel("recall (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_scatter(rows, out):
    xy = np.array([r["xy_max_error_m"] for r in rows])
    yaw = np.array([r["yaw_max_error_deg"] for r in rows])
    pitch = np.array([r["pitch"] for r in rows])
    loss = np.array([r["loss_total"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    mask = np.isfinite(xy) & np.isfinite(yaw) & np.isfinite(pitch)
    if np.any(mask):
        sc = axes[0].scatter(xy[mask], yaw[mask], c=pitch[mask], cmap="coolwarm", s=48, ec="white", lw=0.35)
        cb = fig.colorbar(sc, ax=axes[0], fraction=0.045, pad=0.02)
        cb.set_label("pitch (deg)")
    else:
        axes[0].text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xlabel("XY max error (m)")
    axes[0].set_ylabel("Yaw max error (deg)")
    axes[0].set_title("XY vs yaw error, colored by pitch")
    axes[0].grid(alpha=0.25)

    mask = np.isfinite(loss) & np.isfinite(xy) & np.isfinite(yaw)
    if np.any(mask):
        axes[1].scatter(loss[mask], xy[mask], c=yaw[mask], cmap="viridis", s=48, ec="white", lw=0.35)
    else:
        axes[1].text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel("loss")
    axes[1].set_ylabel("XY max error (m)")
    axes[1].set_title("Loss vs XY error, colored by yaw error")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def resize_for_panel(image, width):
    scale = width / image.width
    return image.resize((width, int(image.height * scale)))


def make_case_sheet(rows, tm, data_dir, image_dirname, out, selected_rows, title):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)
    top_k = len(selected_rows)
    if top_k == 0:
        return

    fig, axes = plt.subplots(top_k, 2, figsize=(12, 3.2 * top_k))
    if top_k == 1:
        axes = np.array([axes])

    for idx, row in enumerate(selected_rows):
        ax_img, ax_map = axes[idx]
        image = Image.open(data_dir / image_dirname / row["name"]).convert("RGB")
        ax_img.imshow(resize_for_panel(image, 720))
        ax_img.set_title(
            f"{row['name']}\n"
            f"xy={row['xy_max_error_m']:.1f}m, yaw={row['yaw_max_error_deg']:.1f}deg, "
            f"loss={row['loss_total']:.1f}"
        )
        ax_img.set_axis_off()

        gt_uv = latlon_to_uv(tm, canvas, [[row["gt_lat"], row["gt_lon"]]])[0]
        pred_uv = latlon_to_uv(tm, canvas, [[row["pred_latitude"], row["pred_longitude"]]])[0]
        center = (gt_uv + pred_uv) / 2
        crop = 135
        x0 = int(max(0, center[0] - crop))
        x1 = int(min(rgb.shape[1], center[0] + crop))
        y0 = int(max(0, center[1] - crop))
        y1 = int(min(rgb.shape[0], center[1] + crop))
        ax_map.imshow(rgb[y0:y1, x0:x1], origin="upper")
        local_gt = np.array([[gt_uv[0] - x0, gt_uv[1] - y0]])
        local_pred = np.array([[pred_uv[0] - x0, pred_uv[1] - y0]])
        ax_map.plot(
            [local_gt[0, 0], local_pred[0, 0]],
            [local_gt[0, 1], local_pred[0, 1]],
            color="#f97316",
            lw=2,
        )
        ax_map.scatter(local_gt[:, 0], local_gt[:, 1], s=45, c="#20c997", ec="black", label="GT")
        ax_map.scatter(local_pred[:, 0], local_pred[:, 1], s=45, c="#ff4d4f", ec="black", label="Pred")
        add_yaw_arrows(ax_map, local_gt, np.array([row["gt_yaw_deg"]]), 36, "#20c997", width=0.006)
        add_yaw_arrows(ax_map, local_pred, np.array([row["pred_yaw_max_deg"]]), 36, "#ff4d4f", width=0.006)
        ax_map.set_title("GT vs prediction crop")
        ax_map.set_axis_off()
        if idx == 0:
            ax_map.legend(loc="lower right")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_worst_cases(rows, tm, data_dir, image_dirname, out, top_k):
    worst = sorted(rows, key=lambda r: r["xy_max_error_m"], reverse=True)[:top_k]
    make_case_sheet(
        rows,
        tm,
        data_dir,
        image_dirname,
        out,
        worst,
        "Worst cases by XY error",
    )


def make_good_cases(rows, tm, data_dir, image_dirname, out, top_k):
    def score(row):
        return row["xy_max_error_m"] / 5.0 + row["yaw_max_error_deg"] / 20.0

    good = sorted(rows, key=score)[:top_k]
    make_case_sheet(
        rows,
        tm,
        data_dir,
        image_dirname,
        out,
        good,
        "Good cases by combined XY and yaw error",
    )


def write_summary(rows, out):
    xy = np.array([r["xy_max_error_m"] for r in rows])
    yaw = np.array([r["yaw_max_error_deg"] for r in rows])
    loss = np.array([r["loss_total"] for r in rows])
    xy_finite = xy[np.isfinite(xy)]
    yaw_finite = yaw[np.isfinite(yaw)]
    lines = [
        f"samples: {len(rows)}",
        f"nonfinite loss_total: {len(loss) - np.isfinite(loss).sum()}",
        f"nonfinite xy_max_error_m: {len(xy) - xy_finite.size}",
        f"nonfinite yaw_max_error_deg: {len(yaw) - yaw_finite.size}",
    ]
    if xy_finite.size:
        lines.extend(
            [
                f"xy mean/median: {xy_finite.mean():.3f} / {np.median(xy_finite):.3f} m",
                f"xy p75/p90/max: {np.percentile(xy_finite, 75):.3f} / {np.percentile(xy_finite, 90):.3f} / {xy_finite.max():.3f} m",
            ]
        )
    if yaw_finite.size:
        lines.extend(
            [
                f"yaw mean/median: {yaw_finite.mean():.3f} / {np.median(yaw_finite):.3f} deg",
                f"yaw p75/p90/max: {np.percentile(yaw_finite, 75):.3f} / {np.percentile(yaw_finite, 90):.3f} / {yaw_finite.max():.3f} deg",
            ]
        )
    for threshold in [2, 5, 10, 16, 20]:
        recall = np.mean(xy_finite <= threshold) if xy_finite.size else float("nan")
        lines.append(f"xy_recall_{threshold}m: {recall:.4f}")
    for threshold in [2, 5, 10, 20, 45, 90]:
        recall = np.mean(yaw_finite <= threshold) if yaw_finite.size else float("nan")
        lines.append(f"yaw_recall_{threshold}deg: {recall:.4f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data_one"))
    parser.add_argument("--image-dirname", default="dataset")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("outputs/eval_visualizations") / args.csv.parent.name / args.csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.data_dir)
    rows = load_eval(args.csv, metadata)
    tm = TileManager.load(args.data_dir / "tiles.pkl")

    make_error_map(rows, tm, out_dir / "error_map.png")
    make_histograms(rows, out_dir / "error_histograms.png")
    make_recall_curves(rows, out_dir / "recall_curves.png")
    make_scatter(rows, out_dir / "error_scatter.png")
    make_worst_cases(rows, tm, args.data_dir, args.image_dirname, out_dir / "worst_cases.png", args.top_k)
    make_good_cases(rows, tm, args.data_dir, args.image_dirname, out_dir / "good_cases.png", args.top_k)
    write_summary(rows, out_dir / "summary.txt")

    for name in [
        "error_map.png",
        "error_histograms.png",
        "recall_curves.png",
        "error_scatter.png",
        "worst_cases.png",
        "good_cases.png",
        "summary.txt",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
