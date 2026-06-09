#!/usr/bin/env python3
"""Create a PDF report for top-ranked repair-guidance pairs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maploc.osm.viz import Colormap
from scripts import demo_repair_guidance as guidance


def load_pair_records(path: Path, top_k: int):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            record = dict(row)
            for key, value in list(record.items()):
                if key in {"rank", "report_name", "worker_name", "route_status", "direct_guidance", "route_guidance"}:
                    continue
                if value != "":
                    record[key] = float(value)
            records.append(record)
            if len(records) >= top_k:
                break
    return records


def photo_number(name):
    match = re.search(r"_(\d{4})_V", name)
    return match.group(1) if match else Path(name).stem


def resize_for_axis(image, max_width=900):
    scale = max_width / image.width
    if scale >= 1:
        return image
    return image.resize((max_width, int(image.height * scale)))


def guidance_for_pair(report, worker, report_xy, worker_xy, repair_xy, route):
    to_repair = repair_xy - worker_xy
    direct_distance = float(np.linalg.norm(to_repair))
    direct_bearing = guidance.bearing_from_xy_delta(to_repair)
    direct_relative = guidance.angle_wrap_deg(direct_bearing - worker["pred_yaw_max_deg"])
    route_bearing = guidance.route_initial_bearing(route["path_xy"])
    route_relative = guidance.angle_wrap_deg(route_bearing - worker["pred_yaw_max_deg"])
    return {
        "distance_m": direct_distance,
        "bearing_deg": direct_bearing,
        "relative_deg": direct_relative,
        "phrase": guidance.direction_phrase(direct_relative),
        "route_distance_m": route["length_m"],
        "route_bearing_deg": route_bearing,
        "route_relative_deg": route_relative,
        "route_phrase": guidance.direction_phrase(route_relative),
    }


def route_arrow_samples(route_uv, spacing_px=75):
    return guidance.route_arrow_samples(route_uv, spacing_px=spacing_px)


def draw_map(ax, tm, report, worker, repair_xy, route):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)

    report_xy = guidance.project_pred_xy(tm, report)
    worker_xy = guidance.project_pred_xy(tm, worker)
    route_xy = route["path_xy"]
    pts = np.concatenate([np.stack([report_xy, worker_xy, repair_xy]), route_xy], axis=0)
    uv = canvas.to_uv(pts)
    report_uv, worker_uv, repair_uv = uv[:3]
    route_uv = canvas.to_uv(route_xy)

    min_uv = uv.min(axis=0)
    max_uv = uv.max(axis=0)
    margin = 150
    x0 = int(max(0, min_uv[0] - margin))
    y0 = int(max(0, min_uv[1] - margin))
    x1 = int(min(rgb.shape[1], max_uv[0] + margin))
    y1 = int(min(rgb.shape[0], max_uv[1] + margin))

    ax.imshow(rgb[y0:y1, x0:x1], origin="upper")
    offset = np.array([x0, y0])
    local_report = report_uv - offset
    local_worker = worker_uv - offset
    local_repair = repair_uv - offset
    local_route = route_uv - offset

    ax.plot(
        local_route[:, 0],
        local_route[:, 1],
        c="#0f766e",
        lw=4.2,
        alpha=0.92,
        label="OSM routed path",
        zorder=4,
    )
    arrow_points, arrow_deltas = route_arrow_samples(local_route)
    if len(arrow_points) > 0:
        ax.quiver(
            arrow_points[:, 0],
            arrow_points[:, 1],
            arrow_deltas[:, 0],
            arrow_deltas[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#14b8a6",
            width=0.006,
            zorder=6,
        )
    ax.plot(
        [local_worker[0], local_repair[0]],
        [local_worker[1], local_repair[1]],
        c="#2563eb",
        lw=1.5,
        alpha=0.35,
        ls="--",
        label="direct line",
    )
    ax.plot(
        [local_report[0], local_repair[0]],
        [local_report[1], local_repair[1]],
        c="#f97316",
        lw=2.0,
        alpha=0.78,
        label="reported view ray",
    )
    guidance.draw_pose(ax, local_report, report["pred_yaw_max_deg"], "#f97316", "report pose")
    guidance.draw_pose(ax, local_worker, worker["pred_yaw_max_deg"], "#2563eb", "worker pose")
    ax.scatter(
        [local_repair[0]],
        [local_repair[1]],
        s=190,
        marker="*",
        c="#ef4444",
        ec="black",
        label="estimated repair spot",
        zorder=7,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_axis_off()


def result_text(record, guide, repair_latlon):
    return "\n".join(
        [
            f"Rank: {int(record['rank'])}",
            f"Final score: {record['final_score']:.3f}",
            "",
            f"Report photo number: #{photo_number(record['report_name'])}",
            f"Worker photo number: #{photo_number(record['worker_name'])}",
            "",
            f"Report pose error: XY {record['report_xy_error_m']:.2f} m / Yaw {record['report_yaw_error_deg']:.2f} deg",
            f"Worker pose error: XY {record['worker_xy_error_m']:.2f} m / Yaw {record['worker_yaw_error_deg']:.2f} deg",
            "",
            f"Direct distance: {guide['distance_m']:.2f} m",
            f"Direct guidance: {guide['phrase']}",
            f"OSM route distance: {guide['route_distance_m']:.2f} m",
            f"OSM route guidance: {guide['route_phrase']}",
            "",
            f"Repair anchor lat/lon: {repair_latlon[0]:.9f}, {repair_latlon[1]:.9f}",
            "",
            "Note: The repair anchor is placed in front of the reported photo",
            "using the predicted report pose and defect-distance setting.",
        ]
    )


def make_page(record, row_by_name, tm, args):
    report = row_by_name[record["report_name"]]
    worker = row_by_name[record["worker_name"]]
    report_xy = guidance.project_pred_xy(tm, report)
    worker_xy = guidance.project_pred_xy(tm, worker)
    repair_xy = report_xy + guidance.yaw_to_xy_delta(report["pred_yaw_max_deg"]) * args.defect_distance_m
    route = guidance.compute_osm_route(
        tm,
        worker_xy,
        repair_xy,
        margin_m=args.route_margin_m,
        building_clearance_m=args.building_clearance_m,
        barrier_clearance_m=args.barrier_clearance_m,
    )
    guide = guidance_for_pair(report, worker, report_xy, worker_xy, repair_xy, route)
    repair_latlon = tm.projection.unproject(repair_xy)

    fig = plt.figure(figsize=(16.5, 10.5))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.03, 1.45],
        height_ratios=[1, 1, 0.82],
        wspace=0.08,
        hspace=0.22,
    )
    ax_report = fig.add_subplot(grid[0, 0])
    ax_worker = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[2, 0])
    ax_map = fig.add_subplot(grid[:, 1])

    report_img = Image.open(args.data_dir / args.image_dirname / report["name"]).convert("RGB")
    worker_img = Image.open(args.data_dir / args.image_dirname / worker["name"]).convert("RGB")
    ax_report.imshow(resize_for_axis(report_img))
    ax_report.set_title(
        f"Report #{photo_number(report['name'])}: {report['name']}\n"
        f"pose error XY={report['xy_max_error_m']:.2f} m, yaw={report['yaw_max_error_deg']:.2f} deg",
        fontsize=9,
    )
    ax_report.set_axis_off()

    ax_worker.imshow(resize_for_axis(worker_img))
    ax_worker.set_title(
        f"Worker #{photo_number(worker['name'])}: {worker['name']}\n"
        f"pose error XY={worker['xy_max_error_m']:.2f} m, yaw={worker['yaw_max_error_deg']:.2f} deg",
        fontsize=9,
    )
    ax_worker.set_axis_off()

    ax_text.text(
        0,
        1,
        result_text(record, guide, repair_latlon),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        linespacing=1.18,
    )
    ax_text.set_axis_off()

    draw_map(ax_map, tm, report, worker, repair_xy, route)
    ax_map.set_title(
        f"OSM-aware Route | Route {guide['route_distance_m']:.1f} m | {guide['route_phrase']}",
        fontsize=12,
    )
    fig.suptitle(
        f"Top {int(record['rank'])} Repair Guidance Pair: "
        f"Report #{photo_number(report['name'])} -> Worker #{photo_number(worker['name'])}",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def write_top20_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=Path("outputs/repair_guidance_pair_ranking/xybest_epoch12/pair_ranking.csv"),
    )
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data_one"))
    parser.add_argument("--image-dirname", default="dataset")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/repair_guidance_pair_ranking/xybest_epoch12/top20_pdf"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--defect-distance-m", type=float, default=8.0)
    parser.add_argument("--route-margin-m", type=float, default=70.0)
    parser.add_argument("--building-clearance-m", type=float, default=1.5)
    parser.add_argument("--barrier-clearance-m", type=float, default=0.4)
    args = parser.parse_args()

    args.eval_csv = args.eval_csv or Path(
        "experiments/gist_abc_balanced_no_rectify_lr1e-5_init16_from_laststep_ep20_xybest/"
        "eval_test_xybest_epoch12.csv"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = guidance.load_metadata(args.data_dir)
    rows = guidance.load_eval(args.eval_csv, metadata)
    row_by_name = {row["name"]: row for row in rows}
    records = load_pair_records(args.ranking_csv, args.top_k)
    tm = guidance.TileManager.load(args.data_dir / "tiles.pkl")

    top_csv = args.out_dir / "top20_pairs.csv"
    pdf_path = args.out_dir / "top20_repair_guidance_pairs.pdf"
    write_top20_csv(top_csv, records)

    with PdfPages(pdf_path) as pdf:
        for idx, record in enumerate(records, 1):
            fig = make_page(record, row_by_name, tm, args)
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
            print(f"page {idx}/{len(records)}")

    print(pdf_path)
    print(top_csv)


if __name__ == "__main__":
    main()
