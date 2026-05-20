"""
Build a single tiles.pkl covering all sessions in a custom dataset.

Reads lat/lon from every poses.txt passed via --poses, computes the union
bbox, downloads OSM data via the public API on first run (cached after that),
and writes tiles.pkl + a sanity HTML overlay.

Reference: maploc/data/kitti/prepare.py::prepare_osm (lines 22-51).
"""

import argparse
from pathlib import Path

import numpy as np

from maploc.osm.tiling import TileManager
from maploc.osm.viz import GeoPlotter
from maploc.utils.geo import BoundaryBox, Projection


def read_latlons(poses_path: Path) -> np.ndarray:
    """poses.txt line = '<filename> <lat> <lon> <alt> <roll> <pitch> <yaw>'"""
    ll = []
    for line in poses_path.read_text().strip().splitlines():
        parts = line.split()
        ll.append([float(parts[1]), float(parts[2])])
    return np.array(ll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, nargs="+", required=True,
                    help="One or more poses.txt files (combine into one tile set).")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--margin_m", type=float, default=128.0,
                    help="Extra meters around the trajectory bbox.")
    ap.add_argument("--ppm", type=int, default=2,
                    help="Pixels per meter (must match data config).")
    ap.add_argument("--tile_size", type=int, default=128,
                    help="Storage tile size in meters (no impact on training).")
    args = ap.parse_args()

    # 1) Collect lat/lons
    parts = []
    for p in args.poses:
        ll = read_latlons(p)
        print(f"  {p}: {len(ll)} frames")
        parts.append(ll)
    all_ll = np.vstack(parts)
    print(f"Total: {len(all_ll)} frames")
    print(f"  lat: {all_ll[:,0].min():.6f} ~ {all_ll[:,0].max():.6f}")
    print(f"  lon: {all_ll[:,1].min():.6f} ~ {all_ll[:,1].max():.6f}")

    # 2) Local metric frame + bbox + margin
    projection = Projection.from_points(all_ll)
    xy = projection.project(all_ll)
    bbox = BoundaryBox(xy.min(0), xy.max(0)) + args.margin_m
    print(f"bbox size (m): {bbox.size}  (margin={args.margin_m})")

    # 3) Build tiles (auto-downloads OSM JSON if cache file missing)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    osm_cache  = args.out_dir / "area.osm.json"
    tiles_path = args.out_dir / "tiles.pkl"

    tm = TileManager.from_bbox(
        projection=projection,
        bbox=bbox,
        ppm=args.ppm,
        path=osm_cache,
        tile_size=args.tile_size,
    )
    tm.save(tiles_path)
    print(f"Saved {tiles_path}")

    # 4) HTML overlay for visual verification
    plotter = GeoPlotter()
    plotter.points(all_ll, "red", name="frames")
    plotter.bbox(projection.unproject(bbox), "blue", "tile bbox")
    preview = args.out_dir / "tiles_preview.html"
    plotter.fig.write_html(preview)
    print(f"Wrote {preview}")


if __name__ == "__main__":
    main()
