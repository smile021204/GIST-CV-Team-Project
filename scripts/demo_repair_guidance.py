#!/usr/bin/env python3
"""Create a facility repair guidance demo from localized GIST ABC photos.

This does not detect defects. It demonstrates how localized camera poses can
anchor a reported repair spot and guide a worker from their current photo pose.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
from pathlib import Path

import cv2
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
from maploc.utils.geo import BoundaryBox


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


def find_row_by_name(rows, requested_name, kind):
    if requested_name is None:
        return None
    requested = Path(requested_name)
    candidates = [requested_name]
    if requested.suffix == "":
        candidates.extend([requested_name + ".JPG", requested_name + ".jpg"])
    candidate_set = set(candidates)
    for row in rows:
        if row["name"] in candidate_set or Path(row["name"]).stem == requested.stem:
            return row
    raise SystemExit(f"{kind} image not found in CSV: {requested_name}")


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
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            meta = metadata[row["name"]]
            row["gt_lat"] = meta["lat"]
            row["gt_lon"] = meta["lon"]
            row["pitch"] = meta["pitch"]
            row["roll"] = meta["roll"]
            row["time"] = meta.get("time", "")
            for key in NUMERIC_KEYS:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def yaw_to_xy_delta(yaw_deg):
    yaw = math.radians(yaw_deg)
    return np.array([math.sin(yaw), math.cos(yaw)], dtype=float)


def yaw_to_uv_delta(yaw_deg, length_px):
    dxy = yaw_to_xy_delta(yaw_deg)
    return np.array([dxy[0], -dxy[1]]) * length_px


def bearing_from_xy_delta(delta_xy):
    return (math.degrees(math.atan2(delta_xy[0], delta_xy[1])) + 360.0) % 360.0


def angle_wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def direction_phrase(relative_bearing):
    angle = angle_wrap_deg(relative_bearing)
    abs_angle = abs(angle)
    side = "right" if angle > 0 else "left"
    if abs_angle <= 20:
        return f"go mostly straight ({angle:+.1f} deg)"
    if abs_angle <= 60:
        return f"turn slightly {side} ({angle:+.1f} deg)"
    if abs_angle <= 135:
        return f"turn {side} ({angle:+.1f} deg)"
    return f"turn around ({angle:+.1f} deg)"


def score_localization(row):
    return row["xy_max_error_m"] / 5.0 + row["yaw_max_error_deg"] / 20.0


def project_pred_xy(tm, row):
    return tm.projection.project(np.array([[row["pred_latitude"], row["pred_longitude"]]]))[0]


def select_report(rows, requested_name):
    if requested_name:
        return find_row_by_name(rows, requested_name, "Report")
    return min(rows, key=score_localization)


def select_worker(rows, requested_name, report_name, repair_xy, tm, target_distance_m, max_xy_error, max_yaw_error):
    if requested_name:
        return find_row_by_name(rows, requested_name, "Worker")

    candidates = [
        row
        for row in rows
        if row["name"] != report_name
        and row["xy_max_error_m"] <= max_xy_error
        and row["yaw_max_error_deg"] <= max_yaw_error
    ]
    if not candidates:
        candidates = [row for row in rows if row["name"] != report_name]

    def candidate_score(row):
        worker_xy = project_pred_xy(tm, row)
        distance = float(np.linalg.norm(repair_xy - worker_xy))
        return abs(distance - target_distance_m) + score_localization(row)

    return min(candidates, key=candidate_score)


def resize_for_panel(image, width):
    scale = width / image.width
    return image.resize((width, int(image.height * scale)))


def latlon_to_uv(tm, canvas, latlon):
    xy = tm.projection.project(np.asarray(latlon))
    return canvas.to_uv(xy)


def xy_to_uv(canvas, xy):
    return canvas.to_uv(np.asarray(xy))


def draw_pose(ax, uv, yaw, color, label, length_px=42):
    ax.scatter([uv[0]], [uv[1]], s=70, c=color, ec="black", label=label, zorder=5)
    duv = yaw_to_uv_delta(yaw, length_px)
    ax.quiver(
        [uv[0]],
        [uv[1]],
        [duv[0]],
        [duv[1]],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=0.006,
        zorder=6,
    )


def group_value(groups, channel, label):
    labels = ["void"] + groups[channel]
    return labels.index(label) if label in labels else None


def dilate_mask(mask, radius_px):
    if radius_px <= 0:
        return mask
    size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def build_route_cost(canvas, groups, building_clearance_m=1.5, barrier_clearance_m=0.4):
    area = canvas.raster[0]
    way = canvas.raster[1]
    cost = np.full(area.shape, 7.0, dtype=np.float32)

    area_costs = {
        "parking": 2.8,
        "playground": 6.0,
        "grass": 9.0,
        "park": 6.0,
        "forest": 13.0,
    }
    for label, value in area_costs.items():
        idx = group_value(groups, "areas", label)
        if idx is not None:
            cost[area == idx] = value

    way_costs = {
        "cycleway": 1.2,
        "path": 1.0,
        "road": 1.8,
        "busway": 3.0,
    }
    for label, value in way_costs.items():
        idx = group_value(groups, "ways", label)
        if idx is not None:
            cost[way == idx] = value

    blocked = np.zeros(area.shape, dtype=bool)
    for label in ["building", "water"]:
        idx = group_value(groups, "areas", label)
        if idx is not None:
            blocked |= area == idx
    blocked = dilate_mask(blocked, int(round(building_clearance_m * canvas.ppm)))

    barriers = np.zeros(area.shape, dtype=bool)
    for label in ["fence", "wall", "hedge", "building_outline"]:
        idx = group_value(groups, "ways", label)
        if idx is not None:
            barriers |= way == idx
    barriers = dilate_mask(barriers, int(round(barrier_clearance_m * canvas.ppm)))

    cost[blocked | barriers] = np.inf
    return cost


def uv_to_rc(uv):
    return int(round(float(uv[1]))), int(round(float(uv[0])))


def rc_to_uv(rc):
    return np.array([rc[1], rc[0]], dtype=float)


def nearest_walkable(cost, uv, max_radius_px=120):
    h, w = cost.shape
    y, x = uv_to_rc(uv)
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)
    if np.isfinite(cost[y, x]):
        return y, x
    for radius in range(1, max_radius_px + 1):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        window = np.isfinite(cost[y0:y1, x0:x1])
        if not window.any():
            continue
        ys, xs = np.where(window)
        ys = ys + y0
        xs = xs + x0
        best = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
        return int(ys[best]), int(xs[best])
    raise RuntimeError("Could not snap point to a walkable OSM cell.")


def astar_route(cost, start_rc, goal_rc):
    h, w = cost.shape
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    g = np.full((h, w), np.inf, dtype=np.float32)
    parent_y = np.full((h, w), -1, dtype=np.int32)
    parent_x = np.full((h, w), -1, dtype=np.int32)
    closed = np.zeros((h, w), dtype=bool)

    gy, gx = goal_rc

    def heuristic(y, x):
        dx = abs(x - gx)
        dy = abs(y - gy)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    sy, sx = start_rc
    g[sy, sx] = 0.0
    heap = [(heuristic(sy, sx), 0.0, sy, sx)]

    while heap:
        _, current_g, y, x = heapq.heappop(heap)
        if closed[y, x]:
            continue
        if (y, x) == goal_rc:
            break
        closed[y, x] = True
        for dy, dx, step_len in neighbors:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or closed[ny, nx]:
                continue
            if not np.isfinite(cost[ny, nx]):
                continue
            step_cost = step_len * float((cost[y, x] + cost[ny, nx]) * 0.5)
            next_g = current_g + step_cost
            if next_g < g[ny, nx]:
                g[ny, nx] = next_g
                parent_y[ny, nx] = y
                parent_x[ny, nx] = x
                heapq.heappush(heap, (next_g + heuristic(ny, nx), next_g, ny, nx))

    if not np.isfinite(g[gy, gx]):
        raise RuntimeError("Could not find a walkable route between the selected photos.")

    path = []
    y, x = goal_rc
    while y >= 0 and x >= 0:
        path.append((y, x))
        if (y, x) == start_rc:
            break
        py, px = parent_y[y, x], parent_x[y, x]
        y, x = int(py), int(px)
    path.reverse()
    return np.array([rc_to_uv(rc) for rc in path], dtype=float)


def simplify_polyline(points, epsilon):
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    line = end - start
    norm = np.linalg.norm(line)
    if norm < 1e-6:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        rel = start - points
        distances = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / norm
    idx = int(np.argmax(distances))
    if distances[idx] <= epsilon:
        return np.stack([start, end])
    left = simplify_polyline(points[: idx + 1], epsilon)
    right = simplify_polyline(points[idx:], epsilon)
    return np.concatenate([left[:-1], right], axis=0)


def route_bbox(tm, points_xy, margin_m):
    bbox = BoundaryBox(points_xy.min(0), points_xy.max(0)) + margin_m
    return bbox & tm.bbox


def route_length(path_xy):
    if path_xy is None or len(path_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())


def compute_osm_route(
    tm,
    worker_xy,
    repair_xy,
    margin_m=70.0,
    building_clearance_m=1.5,
    barrier_clearance_m=0.4,
):
    bbox = route_bbox(tm, np.stack([worker_xy, repair_xy]), margin_m)
    canvas = tm.query(bbox)
    cost = build_route_cost(canvas, tm.groups, building_clearance_m, barrier_clearance_m)
    start_rc = nearest_walkable(cost, canvas.to_uv(worker_xy), int(round(25 * canvas.ppm)))
    goal_rc = nearest_walkable(cost, canvas.to_uv(repair_xy), int(round(25 * canvas.ppm)))
    path_uv = astar_route(cost, start_rc, goal_rc)
    path_xy = canvas.to_xy(path_uv)
    path_xy = simplify_polyline(path_xy, epsilon=1.5)
    return {
        "path_xy": path_xy,
        "length_m": route_length(path_xy),
        "start_xy": canvas.to_xy(rc_to_uv(start_rc)),
        "goal_xy": canvas.to_xy(rc_to_uv(goal_rc)),
        "bbox": bbox,
    }


def route_initial_bearing(path_xy, lookahead_m=6.0):
    if path_xy is None or len(path_xy) < 2:
        return None
    total = 0.0
    for idx in range(1, len(path_xy)):
        delta = path_xy[idx] - path_xy[idx - 1]
        step = float(np.linalg.norm(delta))
        if step <= 1e-6:
            continue
        total += step
        if total >= lookahead_m:
            return bearing_from_xy_delta(path_xy[idx] - path_xy[0])
    return bearing_from_xy_delta(path_xy[-1] - path_xy[0])


def route_arrow_samples(route_uv, spacing_px=90):
    if route_uv is None or len(route_uv) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    points = []
    deltas = []
    dist_since = 0.0
    for idx in range(1, len(route_uv)):
        prev = route_uv[idx - 1]
        cur = route_uv[idx]
        segment = cur - prev
        seg_len = float(np.linalg.norm(segment))
        if seg_len <= 1e-6:
            continue
        dist_since += seg_len
        if dist_since >= spacing_px:
            direction = segment / seg_len
            points.append(prev + segment * 0.55)
            deltas.append(direction * 28)
            dist_since = 0.0
    return np.asarray(points), np.asarray(deltas)


def make_guidance_figure(report, worker, repair_xy, guidance, tm, data_dir, image_dirname, out, route=None):
    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)

    report_xy = project_pred_xy(tm, report)
    worker_xy = project_pred_xy(tm, worker)
    pts = np.stack([report_xy, worker_xy, repair_xy])
    if route is not None and route.get("path_xy") is not None:
        pts = np.concatenate([pts, route["path_xy"]], axis=0)
    uv_pts = xy_to_uv(canvas, pts)
    report_uv, worker_uv, repair_uv = uv_pts[:3]
    route_uv = None
    if route is not None and route.get("path_xy") is not None:
        route_uv = xy_to_uv(canvas, route["path_xy"])

    min_uv = uv_pts.min(axis=0)
    max_uv = uv_pts.max(axis=0)
    margin = 170
    x0 = int(max(0, min_uv[0] - margin))
    y0 = int(max(0, min_uv[1] - margin))
    x1 = int(min(rgb.shape[1], max_uv[0] + margin))
    y1 = int(min(rgb.shape[0], max_uv[1] + margin))

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.55])
    ax_report = fig.add_subplot(grid[0, 0])
    ax_worker = fig.add_subplot(grid[1, 0])
    ax_map = fig.add_subplot(grid[:, 1])

    report_img = Image.open(data_dir / image_dirname / report["name"]).convert("RGB")
    worker_img = Image.open(data_dir / image_dirname / worker["name"]).convert("RGB")
    ax_report.imshow(resize_for_panel(report_img, 720))
    ax_report.set_title(
        "Reported repair photo\n"
        f"{report['name']}\n"
        f"pose err xy={report['xy_max_error_m']:.1f}m, yaw={report['yaw_max_error_deg']:.1f}deg",
        fontsize=10,
    )
    ax_report.set_axis_off()

    ax_worker.imshow(resize_for_panel(worker_img, 720))
    ax_worker.set_title(
        "Worker current photo\n"
        f"{worker['name']}\n"
        f"pose err xy={worker['xy_max_error_m']:.1f}m, yaw={worker['yaw_max_error_deg']:.1f}deg",
        fontsize=10,
    )
    ax_worker.set_axis_off()

    ax_map.imshow(rgb[y0:y1, x0:x1], origin="upper")
    local_report = report_uv - np.array([x0, y0])
    local_worker = worker_uv - np.array([x0, y0])
    local_repair = repair_uv - np.array([x0, y0])
    if route_uv is not None:
        local_route = route_uv - np.array([x0, y0])
        ax_map.plot(
            local_route[:, 0],
            local_route[:, 1],
            c="#0f766e",
            lw=4,
            alpha=0.9,
            label="OSM routed path",
            zorder=4,
        )
        arrow_points, arrow_deltas = route_arrow_samples(local_route)
        if len(arrow_points) > 0:
            ax_map.quiver(
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
    ax_map.plot(
        [local_worker[0], local_repair[0]],
        [local_worker[1], local_repair[1]],
        c="#2563eb",
        lw=1.6,
        alpha=0.35,
        ls="--",
        label="direct line",
    )
    ax_map.plot(
        [local_report[0], local_repair[0]],
        [local_report[1], local_repair[1]],
        c="#f97316",
        lw=2,
        alpha=0.7,
        label="reported view ray",
    )
    draw_pose(ax_map, local_report, report["pred_yaw_max_deg"], "#f97316", "report pose")
    draw_pose(ax_map, local_worker, worker["pred_yaw_max_deg"], "#2563eb", "worker pose")
    ax_map.scatter(
        [local_repair[0]],
        [local_repair[1]],
        s=180,
        marker="*",
        c="#ef4444",
        ec="black",
        label="estimated repair spot",
        zorder=7,
    )
    ax_map.set_title(
        "Campus repair guidance with OSM-aware route\n"
        f"route {guidance.get('route_distance_m', guidance['distance_m']):.1f} m, "
        f"first direction: {guidance.get('route_phrase', guidance['phrase'])}",
        fontsize=11,
    )
    ax_map.legend(loc="lower right", fontsize=8)
    ax_map.set_axis_off()

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(report, worker, repair_latlon, guidance, out):
    lines = [
        "Campus repair guidance demo",
        "",
        "This demo localizes photos and uses the predicted camera pose to anchor a repair spot.",
        "It does not detect the defect itself.",
        "The routed path blocks OSM building/water cells and prefers OSM path/road/parking cells.",
        "",
        f"report_photo: {report['name']}",
        f"worker_photo: {worker['name']}",
        "",
        f"report_pred_latlon: {report['pred_latitude']:.9f}, {report['pred_longitude']:.9f}",
        f"report_pred_yaw_deg: {report['pred_yaw_max_deg']:.3f}",
        f"worker_pred_latlon: {worker['pred_latitude']:.9f}, {worker['pred_longitude']:.9f}",
        f"worker_pred_yaw_deg: {worker['pred_yaw_max_deg']:.3f}",
        f"repair_anchor_latlon: {repair_latlon[0]:.9f}, {repair_latlon[1]:.9f}",
        "",
        f"direct_distance_worker_to_repair_m: {guidance['distance_m']:.3f}",
        f"direct_bearing_to_repair_deg: {guidance['bearing_deg']:.3f}",
        f"direct_relative_to_worker_view_deg: {guidance['relative_deg']:.3f}",
        f"direct_guidance: {guidance['phrase']}",
    ]
    if "route_distance_m" in guidance:
        lines.extend(
            [
                "",
                f"osm_route_distance_m: {guidance['route_distance_m']:.3f}",
                f"osm_route_first_bearing_deg: {guidance['route_bearing_deg']:.3f}",
                f"osm_route_first_relative_to_worker_view_deg: {guidance['route_relative_deg']:.3f}",
                f"osm_route_guidance: {guidance['route_phrase']}",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/repair_guidance_demo"))
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--worker-name", default=None)
    parser.add_argument("--defect-distance-m", type=float, default=8.0)
    parser.add_argument("--target-worker-distance-m", type=float, default=22.0)
    parser.add_argument("--max-xy-error", type=float, default=5.0)
    parser.add_argument("--max-yaw-error", type=float, default=20.0)
    parser.add_argument("--route-margin-m", type=float, default=70.0)
    parser.add_argument("--building-clearance-m", type=float, default=1.5)
    parser.add_argument("--barrier-clearance-m", type=float, default=0.4)
    parser.add_argument("--no-route", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.data_dir)
    rows = load_eval(args.csv, metadata)
    tm = TileManager.load(args.data_dir / "tiles.pkl")

    report = select_report(rows, args.report_name)
    report_xy = project_pred_xy(tm, report)
    repair_xy = report_xy + yaw_to_xy_delta(report["pred_yaw_max_deg"]) * args.defect_distance_m
    worker = select_worker(
        rows,
        args.worker_name,
        report["name"],
        repair_xy,
        tm,
        args.target_worker_distance_m,
        args.max_xy_error,
        args.max_yaw_error,
    )
    worker_xy = project_pred_xy(tm, worker)
    to_repair = repair_xy - worker_xy
    distance = float(np.linalg.norm(to_repair))
    bearing = bearing_from_xy_delta(to_repair)
    relative = angle_wrap_deg(bearing - worker["pred_yaw_max_deg"])
    guidance = {
        "distance_m": distance,
        "bearing_deg": bearing,
        "relative_deg": relative,
        "phrase": direction_phrase(relative),
    }

    route = None
    if not args.no_route:
        route = compute_osm_route(
            tm,
            worker_xy,
            repair_xy,
            margin_m=args.route_margin_m,
            building_clearance_m=args.building_clearance_m,
            barrier_clearance_m=args.barrier_clearance_m,
        )
        route_bearing = route_initial_bearing(route["path_xy"])
        if route_bearing is not None:
            route_relative = angle_wrap_deg(route_bearing - worker["pred_yaw_max_deg"])
            guidance.update(
                {
                    "route_distance_m": route["length_m"],
                    "route_bearing_deg": route_bearing,
                    "route_relative_deg": route_relative,
                    "route_phrase": direction_phrase(route_relative),
                }
            )
    repair_latlon = tm.projection.unproject(repair_xy)

    make_guidance_figure(
        report,
        worker,
        repair_xy,
        guidance,
        tm,
        args.data_dir,
        args.image_dirname,
        args.out_dir / "repair_guidance_demo.png",
        route=route,
    )
    write_summary(report, worker, repair_latlon, guidance, args.out_dir / "guidance_summary.txt")
    print(args.out_dir / "repair_guidance_demo.png")
    print(args.out_dir / "guidance_summary.txt")


if __name__ == "__main__":
    main()
