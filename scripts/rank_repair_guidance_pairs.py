#!/usr/bin/env python3
"""Rank report/worker photo pairs for the repair-guidance demo.

The model has already produced one row per test image in the eval CSV. This
script ranks ordered pairs of those rows, then runs OSM-aware routing for the
best candidates so the top of the CSV contains practical demo pairs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import demo_repair_guidance as guidance


def localization_score(row):
    return row["xy_max_error_m"] / 2.0 + row["yaw_max_error_deg"] / 5.0


def distance_penalty(distance_m, min_distance_m, max_distance_m):
    if distance_m < min_distance_m:
        return 50.0 + (min_distance_m - distance_m) / max(min_distance_m, 1.0) * 20.0
    if distance_m > max_distance_m:
        return (distance_m - max_distance_m) / 30.0
    return 0.0


def route_penalty(route_distance_m, direct_distance_m, max_route_distance_m):
    if direct_distance_m <= 1e-6:
        return 100.0
    ratio = route_distance_m / direct_distance_m
    penalty = 0.0
    if ratio > 2.5:
        penalty += (ratio - 2.5) * 8.0
    if route_distance_m > max_route_distance_m:
        penalty += (route_distance_m - max_route_distance_m) / 40.0
    return penalty


def pair_to_record(
    report,
    worker,
    report_xy,
    worker_xy,
    repair_xy,
    min_direct_distance_m,
    max_direct_distance_m,
):
    to_repair = repair_xy - worker_xy
    direct_distance = float(np.linalg.norm(to_repair))
    direct_bearing = guidance.bearing_from_xy_delta(to_repair)
    direct_relative = guidance.angle_wrap_deg(direct_bearing - worker["pred_yaw_max_deg"])

    report_score = localization_score(report)
    worker_score = localization_score(worker)
    pose_score = report_score + worker_score
    pre_route_score = pose_score + distance_penalty(
        direct_distance, min_direct_distance_m, max_direct_distance_m
    )

    return {
        "rank": "",
        "final_score": pre_route_score + 1000.0,
        "pre_route_score": pre_route_score,
        "pose_score": pose_score,
        "route_status": "not_routed",
        "report_name": report["name"],
        "worker_name": worker["name"],
        "report_xy_error_m": report["xy_max_error_m"],
        "report_yaw_error_deg": report["yaw_max_error_deg"],
        "worker_xy_error_m": worker["xy_max_error_m"],
        "worker_yaw_error_deg": worker["yaw_max_error_deg"],
        "direct_distance_m": direct_distance,
        "direct_bearing_deg": direct_bearing,
        "direct_relative_deg": direct_relative,
        "direct_guidance": guidance.direction_phrase(direct_relative),
        "route_distance_m": "",
        "route_ratio": "",
        "route_first_bearing_deg": "",
        "route_first_relative_deg": "",
        "route_guidance": "",
        "report_pred_latitude": report["pred_latitude"],
        "report_pred_longitude": report["pred_longitude"],
        "report_pred_yaw_deg": report["pred_yaw_max_deg"],
        "worker_pred_latitude": worker["pred_latitude"],
        "worker_pred_longitude": worker["pred_longitude"],
        "worker_pred_yaw_deg": worker["pred_yaw_max_deg"],
    }


def write_csv(path, records):
    fieldnames = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def format_value(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_top_summary(path, records, args):
    lines = [
        "Repair-guidance pair ranking",
        "",
        f"csv: {args.csv}",
        f"test_pairs: {len(records)}",
        f"routed_candidates: {sum(r['route_status'] == 'ok' for r in records)}",
        f"defect_distance_m: {args.defect_distance_m}",
        "",
        "Scoring: lower is better.",
        "final_score = report/worker localization score + distance penalty + route penalty",
        "",
        "Top pairs:",
    ]
    cols = [
        "rank",
        "final_score",
        "report_name",
        "worker_name",
        "report_xy_error_m",
        "report_yaw_error_deg",
        "worker_xy_error_m",
        "worker_yaw_error_deg",
        "direct_distance_m",
        "route_distance_m",
        "route_guidance",
    ]
    for record in records[: args.summary_top_k]:
        lines.append("")
        for col in cols:
            lines.append(f"{col}: {format_value(record[col])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def route_candidate(record, row_by_name, xy_by_name, repair_xy_by_name, tm, args):
    report = row_by_name[record["report_name"]]
    worker = row_by_name[record["worker_name"]]
    worker_xy = xy_by_name[worker["name"]]
    repair_xy = repair_xy_by_name[report["name"]]

    route = guidance.compute_osm_route(
        tm,
        worker_xy,
        repair_xy,
        margin_m=args.route_margin_m,
        building_clearance_m=args.building_clearance_m,
        barrier_clearance_m=args.barrier_clearance_m,
    )
    route_bearing = guidance.route_initial_bearing(route["path_xy"])
    if route_bearing is None:
        raise RuntimeError("Route has no initial bearing.")

    route_distance = route["length_m"]
    route_relative = guidance.angle_wrap_deg(route_bearing - worker["pred_yaw_max_deg"])
    direct_distance = float(record["direct_distance_m"])

    record["route_status"] = "ok"
    record["route_distance_m"] = route_distance
    record["route_ratio"] = route_distance / max(direct_distance, 1e-6)
    record["route_first_bearing_deg"] = route_bearing
    record["route_first_relative_deg"] = route_relative
    record["route_guidance"] = guidance.direction_phrase(route_relative)
    record["final_score"] = record["pre_route_score"] + route_penalty(
        route_distance, direct_distance, args.max_route_distance_m
    )
    return route


def create_best_visualization(best_record, row_by_name, repair_xy_by_name, tm, args, out_dir):
    report = row_by_name[best_record["report_name"]]
    worker = row_by_name[best_record["worker_name"]]
    worker_xy = guidance.project_pred_xy(tm, worker)
    repair_xy = repair_xy_by_name[report["name"]]
    route = guidance.compute_osm_route(
        tm,
        worker_xy,
        repair_xy,
        margin_m=args.route_margin_m,
        building_clearance_m=args.building_clearance_m,
        barrier_clearance_m=args.barrier_clearance_m,
    )

    to_repair = repair_xy - worker_xy
    direct_distance = float(np.linalg.norm(to_repair))
    direct_bearing = guidance.bearing_from_xy_delta(to_repair)
    direct_relative = guidance.angle_wrap_deg(direct_bearing - worker["pred_yaw_max_deg"])
    route_bearing = guidance.route_initial_bearing(route["path_xy"])
    route_relative = guidance.angle_wrap_deg(route_bearing - worker["pred_yaw_max_deg"])
    guide = {
        "distance_m": direct_distance,
        "bearing_deg": direct_bearing,
        "relative_deg": direct_relative,
        "phrase": guidance.direction_phrase(direct_relative),
        "route_distance_m": route["length_m"],
        "route_bearing_deg": route_bearing,
        "route_relative_deg": route_relative,
        "route_phrase": guidance.direction_phrase(route_relative),
    }
    repair_latlon = tm.projection.unproject(repair_xy)

    best_dir = out_dir / "best_pair_visualization"
    best_dir.mkdir(parents=True, exist_ok=True)
    guidance.make_guidance_figure(
        report,
        worker,
        repair_xy,
        guide,
        tm,
        args.data_dir,
        args.image_dirname,
        best_dir / "repair_guidance_demo.png",
        route=route,
    )
    guidance.write_summary(report, worker, repair_latlon, guide, best_dir / "guidance_summary.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "experiments/gist_abc_balanced_no_rectify_lr1e-5_init16_from_laststep_ep20_xybest/"
            "eval_test_xybest_epoch12.csv"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_one"))
    parser.add_argument("--image-dirname", default="dataset")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/repair_guidance_pair_ranking"))
    parser.add_argument("--defect-distance-m", type=float, default=8.0)
    parser.add_argument("--min-direct-distance-m", type=float, default=20.0)
    parser.add_argument("--max-direct-distance-m", type=float, default=250.0)
    parser.add_argument("--max-route-distance-m", type=float, default=330.0)
    parser.add_argument("--route-top-k", type=int, default=250)
    parser.add_argument("--summary-top-k", type=int, default=20)
    parser.add_argument("--route-margin-m", type=float, default=70.0)
    parser.add_argument("--building-clearance-m", type=float, default=1.5)
    parser.add_argument("--barrier-clearance-m", type=float, default=0.4)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--worker-name", default=None)
    parser.add_argument("--no-visualization", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = guidance.load_metadata(args.data_dir)
    rows = guidance.load_eval(args.csv, metadata)
    tm = guidance.TileManager.load(args.data_dir / "tiles.pkl")

    row_by_name = {row["name"]: row for row in rows}
    xy_by_name = {row["name"]: guidance.project_pred_xy(tm, row) for row in rows}
    repair_xy_by_name = {
        row["name"]: xy_by_name[row["name"]]
        + guidance.yaw_to_xy_delta(row["pred_yaw_max_deg"]) * args.defect_distance_m
        for row in rows
    }

    report_rows = rows
    if args.report_name:
        report_rows = [guidance.find_row_by_name(rows, args.report_name, "Report")]
    worker_rows = rows
    if args.worker_name:
        worker_rows = [guidance.find_row_by_name(rows, args.worker_name, "Worker")]

    records = []
    for report in report_rows:
        report_xy = xy_by_name[report["name"]]
        repair_xy = repair_xy_by_name[report["name"]]
        for worker in worker_rows:
            if worker["name"] == report["name"]:
                continue
            records.append(
                pair_to_record(
                    report,
                    worker,
                    report_xy,
                    xy_by_name[worker["name"]],
                    repair_xy,
                    args.min_direct_distance_m,
                    args.max_direct_distance_m,
                )
            )
    if len(records) == 0:
        raise SystemExit("No report/worker pairs to rank.")

    records.sort(key=lambda r: (r["pre_route_score"], r["direct_distance_m"]))
    routed = 0
    for idx, record in enumerate(records[: args.route_top_k], 1):
        try:
            route_candidate(record, row_by_name, xy_by_name, repair_xy_by_name, tm, args)
            routed += 1
        except Exception as exc:  # noqa: BLE001 - keep ranking robust for demo search.
            record["route_status"] = f"failed: {exc}"
            record["final_score"] = record["pre_route_score"] + 900.0
        if idx % 25 == 0 or idx == args.route_top_k:
            print(f"routed {idx}/{args.route_top_k}")

    records.sort(key=lambda r: (r["final_score"], r["pre_route_score"], r["direct_distance_m"]))
    for rank, record in enumerate(records, 1):
        record["rank"] = rank

    csv_path = args.out_dir / "pair_ranking.csv"
    summary_path = args.out_dir / "summary.txt"
    write_csv(csv_path, records)
    write_top_summary(summary_path, records, args)

    if records and not args.no_visualization:
        create_best_visualization(records[0], row_by_name, repair_xy_by_name, tm, args, args.out_dir)

    print(csv_path)
    print(summary_path)
    if not args.no_visualization:
        print(args.out_dir / "best_pair_visualization" / "repair_guidance_demo.png")
        print(args.out_dir / "best_pair_visualization" / "guidance_summary.txt")
    print(f"routed_ok={routed}")


if __name__ == "__main__":
    main()
