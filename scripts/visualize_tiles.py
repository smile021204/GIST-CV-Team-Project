"""
Render a tiles.pkl as a static PNG with the GPS trajectory overlaid.
Useful when working on a remote server where the HTML preview can't open.

Output: <out_dir>/tiles_preview.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap


def read_latlons(poses_path: Path) -> np.ndarray:
    ll = []
    for line in poses_path.read_text().strip().splitlines():
        parts = line.split()
        ll.append([float(parts[1]), float(parts[2])])
    return np.array(ll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--poses", type=Path, nargs="*", default=[],
                    help="Optional poses.txt files; GPS overlaid in red.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: <tiles_dir>/tiles_preview.png)")
    args = ap.parse_args()

    tm = TileManager.load(args.tiles)
    print(f"Loaded {args.tiles}")
    print(f"  bbox (m): {tm.bbox.size}  ppm={tm.ppm}  tile_size={tm.tile_size}")

    canvas = tm.query(tm.bbox)
    rgb = Colormap.apply(canvas.raster)  # (H, W, 3) in [0,1]
    print(f"  raster size: {canvas.raster.shape} -> rgb {rgb.shape}")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb, origin="upper")

    for p in args.poses:
        ll = read_latlons(p)
        xy = tm.projection.project(ll)
        uv = canvas.to_uv(xy)
        ax.scatter(uv[:, 0], uv[:, 1], s=4, c="red", alpha=0.7,
                   label=f"{p.parent.name} (n={len(ll)})")

    ax.set_title(f"tiles.pkl  bbox={tuple(np.round(tm.bbox.size).astype(int))} m, "
                 f"ppm={tm.ppm}")
    ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    if args.poses:
        ax.legend(loc="upper right")

    Colormap.add_colorbar()

    out = args.out or args.tiles.parent / "tiles_preview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
