#!/usr/bin/env python3
"""Compare two GIST ABC eval CSVs on a fixed set of selected images."""

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


NUMERIC_KEYS = [
    "loss_total",
    "xy_max_error_m",
    "xy_expectation_error_m",
    "yaw_max_error_deg",
    "yaw_expectation_error_deg",
    "gt_yaw_deg",
    "pred_yaw_max_deg",
    "pred_latitude",
    "pred_longitude",
]


def load_metadata(data_dir: Path):
    metadata = {}
    with (data_dir / "metadata_full.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            for key in ["lat", "lon", "pitch", "roll", "yaw"]:
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            metadata[row["filename"]] = row
    return metadata


def load_eval(csv_path: Path, metadata):
    rows = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            meta = metadata[row["name"]]
            row["gt_lat"] = meta["lat"]
            row["gt_lon"] = meta["lon"]
            row["pitch"] = meta["pitch"]
            row["roll"] = meta["roll"]
            for key in NUMERIC_KEYS:
                row[key] = float(row[key])
            rows[row["name"]] = row
    return rows


def good_score(row):
    return row["xy_max_error_m"] / 5.0 + row["yaw_max_error_deg"] / 20.0


def select_reference_cases(ref_rows, top_k):
    values = list(ref_rows.values())
    good = sorted(values, key=good_score)[:top_k]
    bad = sorted(values, key=lambda r: r["xy_max_error_m"], reverse=True)[:top_k]

    cases = []
    seen = set()
    for group, rows in [("old_good", good), ("old_bad", bad)]:
        for row in rows:
            if row["name"] in seen:
                continue
            seen.add(row["name"])
            cases.append((group, row["name"]))
    return cases


def is_success(row, xy_thr, yaw_thr):
    return row["xy_max_error_m"] <= xy_thr and row["yaw_max_error_deg"] <= yaw_thr


def yaw_to_uv_delta(yaw_deg, length_px):
    yaw = np.deg2rad(yaw_deg)
    dxy = np.array([np.sin(yaw), np.cos(yaw)])
    return np.array([dxy[0], -dxy[1]]) * length_px


def latlon_to_uv(tm, canvas, latlon):
    xy = tm.projection.project(np.asarray(latlon))
    return canvas.to_uv(xy)


def resize_for_panel(image, width):
    scale = width / image.width
    return image.resize((width, int(image.height * scale)))


def draw_prediction_crop(ax, rgb, tm, canvas, old_row, new_row, xy_thr, yaw_thr):
    gt_uv = latlon_to_uv(tm, canvas, [[old_row["gt_lat"], old_row["gt_lon"]]])[0]
    old_uv = latlon_to_uv(tm, canvas, [[old_row["pred_latitude"], old_row["pred_longitude"]]])[0]
    new_uv = latlon_to_uv(tm, canvas, [[new_row["pred_latitude"], new_row["pred_longitude"]]])[0]

    pts = np.stack([gt_uv, old_uv, new_uv])
    center = pts.mean(axis=0)
    crop = 155
    x0 = int(max(0, center[0] - crop))
    x1 = int(min(rgb.shape[1], center[0] + crop))
    y0 = int(max(0, center[1] - crop))
    y1 = int(min(rgb.shape[0], center[1] + crop))

    ax.imshow(rgb[y0:y1, x0:x1], origin="upper")
    local_gt = gt_uv - np.array([x0, y0])
    local_old = old_uv - np.array([x0, y0])
    local_new = new_uv - np.array([x0, y0])

    ax.plot([local_gt[0], local_old[0]], [local_gt[1], local_old[1]], c="#f97316", lw=2, alpha=0.75)
    ax.plot([local_gt[0], local_new[0]], [local_gt[1], local_new[1]], c="#2563eb", lw=2, alpha=0.85)
    ax.scatter([local_gt[0]], [local_gt[1]], s=48, c="#20c997", ec="black", label="GT", zorder=4)
    ax.scatter([local_old[0]], [local_old[1]], s=48, c="#f97316", ec="black", label="old", zorder=4)
    ax.scatter([local_new[0]], [local_new[1]], s=48, c="#2563eb", ec="black", label="new", zorder=4)

    for uv, yaw, color in [
        (local_gt, old_row["gt_yaw_deg"], "#20c997"),
        (local_old, old_row["pred_yaw_max_deg"], "#f97316"),
        (local_new, new_row["pred_yaw_max_deg"], "#2563eb"),
    ]:
        d = yaw_to_uv_delta(yaw, 35)
        ax.quiver(
            [uv[0]],
            [uv[1]],
            [d[0]],
            [d[1]],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.006,
        )

    old_ok = "OK" if is_success(old_row, xy_thr, yaw_thr) else "FAIL"
    new_ok = "OK" if is_success(new_row, xy_thr, yaw_thr) else "FAIL"
    ax.set_title(
        f"old {old_ok}: xy {old_row['xy_max_error_m']:.1f}m yaw {old_row['yaw_max_error_deg']:.1f}deg\n"
        f"new {new_ok}: xy {new_row['xy_max_error_m']:.1f}m yaw {new_row['yaw_max_error_deg']:.1f}deg",
        fontsize=9,
    )
    ax.set_axis_off()


def make_comparison_sheet(cases, old_rows, new_rows, data_dir, image_dirname, tm, out, xy_thr, yaw_thr):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 3.25 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])

    for idx, (group, name) in enumerate(cases):
        old_row = old_rows[name]
        new_row = new_rows[name]
        ax_img, ax_map = axes[idx]
        image = Image.open(data_dir / image_dirname / name).convert("RGB")
        ax_img.imshow(resize_for_panel(image, 760))
        ax_img.set_title(f"{group} | {name}", fontsize=10)
        ax_img.set_axis_off()
        draw_prediction_crop(ax_map, rgb, tm, canvas, old_row, new_row, xy_thr, yaw_thr)
        if idx == 0:
            ax_map.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Fixed old good/bad cases compared on old vs new weights "
        f"(success: XY<={xy_thr:g}m and yaw<={yaw_thr:g}deg)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_case_table(cases, old_rows, new_rows, out, xy_thr, yaw_thr):
    fieldnames = [
        "group",
        "name",
        "old_status",
        "new_status",
        "status_change",
        "old_xy_m",
        "new_xy_m",
        "delta_xy_m",
        "old_yaw_deg",
        "new_yaw_deg",
        "delta_yaw_deg",
        "old_loss",
        "new_loss",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, name in cases:
            old_row = old_rows[name]
            new_row = new_rows[name]
            old_ok = is_success(old_row, xy_thr, yaw_thr)
            new_ok = is_success(new_row, xy_thr, yaw_thr)
            if old_ok and new_ok:
                change = "both_good"
            elif old_ok and not new_ok:
                change = "old_only_good"
            elif not old_ok and new_ok:
                change = "new_only_good"
            else:
                change = "both_bad"
            writer.writerow(
                {
                    "group": group,
                    "name": name,
                    "old_status": "good" if old_ok else "bad",
                    "new_status": "good" if new_ok else "bad",
                    "status_change": change,
                    "old_xy_m": f"{old_row['xy_max_error_m']:.4f}",
                    "new_xy_m": f"{new_row['xy_max_error_m']:.4f}",
                    "delta_xy_m": f"{new_row['xy_max_error_m'] - old_row['xy_max_error_m']:.4f}",
                    "old_yaw_deg": f"{old_row['yaw_max_error_deg']:.4f}",
                    "new_yaw_deg": f"{new_row['yaw_max_error_deg']:.4f}",
                    "delta_yaw_deg": f"{new_row['yaw_max_error_deg'] - old_row['yaw_max_error_deg']:.4f}",
                    "old_loss": f"{old_row['loss_total']:.4f}",
                    "new_loss": f"{new_row['loss_total']:.4f}",
                }
            )


def write_summary(cases, old_rows, new_rows, out, xy_thr, yaw_thr):
    counts = {"both_good": 0, "old_only_good": 0, "new_only_good": 0, "both_bad": 0}
    deltas_xy = []
    deltas_yaw = []
    lines = [
        f"cases: {len(cases)}",
        f"success rule: xy <= {xy_thr:g} m and yaw <= {yaw_thr:g} deg",
        "",
    ]
    for group, name in cases:
        old_row = old_rows[name]
        new_row = new_rows[name]
        old_ok = is_success(old_row, xy_thr, yaw_thr)
        new_ok = is_success(new_row, xy_thr, yaw_thr)
        if old_ok and new_ok:
            key = "both_good"
        elif old_ok and not new_ok:
            key = "old_only_good"
        elif not old_ok and new_ok:
            key = "new_only_good"
        else:
            key = "both_bad"
        counts[key] += 1
        deltas_xy.append(new_row["xy_max_error_m"] - old_row["xy_max_error_m"])
        deltas_yaw.append(new_row["yaw_max_error_deg"] - old_row["yaw_max_error_deg"])
    for key, value in counts.items():
        lines.append(f"{key}: {value}")
    lines += [
        "",
        f"mean delta xy new-old: {np.mean(deltas_xy):.3f} m",
        f"median delta xy new-old: {np.median(deltas_xy):.3f} m",
        f"mean delta yaw new-old: {np.mean(deltas_yaw):.3f} deg",
        f"median delta yaw new-old: {np.median(deltas_yaw):.3f} deg",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data_one"))
    parser.add_argument("--image-dirname", default="dataset")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--xy-threshold", type=float, default=5.0)
    parser.add_argument("--yaw-threshold", type=float, default=20.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.data_dir)
    old_rows = load_eval(args.old_csv, metadata)
    new_rows = load_eval(args.new_csv, metadata)
    tm = TileManager.load(args.data_dir / "tiles.pkl")

    cases = select_reference_cases(old_rows, args.top_k)
    missing = [name for _, name in cases if name not in new_rows]
    if missing:
        raise SystemExit(f"Missing rows in new CSV: {missing}")

    make_comparison_sheet(
        cases,
        old_rows,
        new_rows,
        args.data_dir,
        args.image_dirname,
        tm,
        args.out_dir / "fixed_cases_old_vs_new.png",
        args.xy_threshold,
        args.yaw_threshold,
    )
    write_case_table(
        cases,
        old_rows,
        new_rows,
        args.out_dir / "fixed_cases_comparison.csv",
        args.xy_threshold,
        args.yaw_threshold,
    )
    write_summary(
        cases,
        old_rows,
        new_rows,
        args.out_dir / "summary.txt",
        args.xy_threshold,
        args.yaw_threshold,
    )
    for name in [
        "fixed_cases_old_vs_new.png",
        "fixed_cases_comparison.csv",
        "summary.txt",
    ]:
        print(args.out_dir / name)


if __name__ == "__main__":
    main()
