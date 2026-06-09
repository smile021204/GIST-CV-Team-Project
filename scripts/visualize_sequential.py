"""Sequential viz that mirrors notebooks/visualize_predictions_sequences.ipynb.

Per-frame MOVING canvas (each frame's canvas is centered on its own GT camera
position, matching KITTI/Mapillary semantics). RigidAligner.update_with_ref
warps the belief between per-frame canvases as the camera moves; this is what
the aligner was designed for, so the log-prob accumulation stays numerically
sane (no NaN explosion).

For visualization, the broader "trajectory" canvas (canvas_total) is the
union bbox of all GT positions + crop_size_meters + 16 m, exactly as the
notebook does it. Per-frame likelihood heatmaps are overlaid on the wider
background with extent = per_frame_canvas.bbox -- so each heatmap appears as
a localized patch on the full-trajectory map.

Color scheme (per user spec):
    GT     = BLACK
    SINGLE = GREEN
    SEQ    = RED

Example:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.visualize_sequential \\
        --experiment OrienterNet_MGL \\
        --split test --stride 1 \\
        --splits_filename splits_no_skybridge.json \\
        --output outputs/viz_seq/mgl_seq.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

from maploc import logger
from maploc.data import CustomDjiDataModule
from maploc.data.torch import collate
from maploc.evaluation.run import pretrained_models, resolve_checkpoint_path
from maploc.models.metrics import angle_error
from maploc.models.sequential import RigidAligner
from maploc.models.voting import argmax_xyr, log_softmax_spatial
from maploc.module import GenericModule
from maploc.osm.viz import Colormap
from maploc.utils.geo import BoundaryBox
from maploc.utils.viz_2d import features_to_RGB


# -------------------------- model & dataset --------------------------

def load_model(experiment: str, num_rotations: int):
    cfg_override = {"model": {"num_rotations": num_rotations, "apply_map_prior": True}}
    if experiment in pretrained_models:
        ckpt_name, defaults = pretrained_models[experiment]
        cfg_override["model"].update(
            {k: v for k, v in defaults.items() if k != "num_rotations"}
        )
        experiment = ckpt_name
    path = resolve_checkpoint_path(experiment)
    logger.info("Loading checkpoint %s", path)
    model = GenericModule.load_from_checkpoint(
        path,
        strict=True,
        find_best=not str(experiment).endswith(".ckpt"),
        cfg=OmegaConf.create(cfg_override),
    )
    return model.eval().cuda()


def snap_to_multiple(crop_size_meters: float, ppm: int = 2, multiple_of: int = 32) -> float:
    """Round crop_size_meters so canvas pixels (= 4 * crop * ppm / 2 = 4 * crop) is divisible by 32.

    The map encoder downsamples by 8 internally; if canvas H/W is not divisible
    the FFT-conv crops scores while map_mask stays full-size -> tensor mismatch.
    We snap UP so the requested area is fully covered.
    """
    canvas_px = 4 * crop_size_meters
    snapped_px = int(np.ceil(canvas_px / multiple_of)) * multiple_of
    return snapped_px / 4


def build_dataset_fixed_center(
    crop_size_meters: float,
    max_init_error_rotation,
    splits_filename: str = "splits.json",
):
    """Build DataModule with init_from_gps=True. The caller must inject
    gps_position so every frame's canvas is centered on the same point."""
    crop_size_meters = snap_to_multiple(crop_size_meters)
    data_cfg = {
        "crop_size_meters": float(crop_size_meters),
        "max_init_error": 0,
        "max_init_error_rotation": max_init_error_rotation,
        "prior_range_rotation": (
            (max_init_error_rotation + 1) if max_init_error_rotation is not None else None
        ),
        "add_map_mask": False,
        "init_from_gps": True,
        "splits_filename": splits_filename,
        "loading": {"test": {"batch_size": 1, "num_workers": 0}},
    }
    cfg = OmegaConf.create({"data": data_cfg})
    dm = CustomDjiDataModule(cfg.data)
    dm.prepare_data()
    dm.setup()
    return dm


def inject_fixed_center(dm, split: str, center_xy):
    center_latlon = dm.tile_manager.projection.unproject(np.array([center_xy]))[0]
    N = len(dm.splits[split])
    dm.data[split]["gps_position"] = (
        torch.tensor(center_latlon, dtype=torch.float64).expand(N, 2).clone().float()
    )


def build_dataset(
    crop_size_meters: float = 96.0,
    max_init_error_rotation=10,
    add_map_mask: bool = False,
    splits_filename: str = "splits.json",
):
    """Notebook-style sequential eval cfg with BROAD per-frame search.

    crop_size_meters        half-width of the per-frame canvas
                            default 96 m -> 192x192 m canvas (covers ~3 buildings)
    max_init_error_rotation yaw prior half-width in degrees (default 10)
    add_map_mask            default False so spatial search is unconstrained
                            inside the canvas (the canvas itself defines the
                            broad search region).
    """
    crop_size_meters = snap_to_multiple(crop_size_meters)
    data_cfg = {
        "crop_size_meters": float(crop_size_meters),
        "max_init_error": 0,
        "max_init_error_rotation": max_init_error_rotation,
        "prior_range_rotation": (
            (max_init_error_rotation + 1) if max_init_error_rotation is not None else None
        ),
        "add_map_mask": bool(add_map_mask),
        "init_from_gps": False,
        "splits_filename": splits_filename,
        "loading": {"test": {"batch_size": 1, "num_workers": 0}},
    }
    cfg = OmegaConf.create({"data": data_cfg})
    dm = CustomDjiDataModule(cfg.data)
    dm.prepare_data()
    dm.setup()
    return dm


# -------------------------- viz helpers --------------------------

def bbox_to_extent(bbox):
    return np.r_[bbox.min_, bbox.max_][[0, 2, 1, 3]]


def heatmap_overlay(lp_3d, p_alpha_exp=0.5, viz_range_nats=100.0):
    """Build an RGBA heatmap from 3D log-probs.

    Color = jet of log-prob, CLIPPED to top viz_range_nats below max.
    Alpha = normalized prob ** p_alpha_exp, so the map shows through where prob is low.

    Clipping is important because the accumulated seq belief can have a huge
    dynamic range that washes the colormap into uniform red.
    """
    lp_2d = lp_3d.max(-1).values.cpu().numpy()
    finite = lp_2d[np.isfinite(lp_2d)]
    if finite.size == 0:
        return np.zeros((*lp_2d.shape, 4)), 0, 0
    lp_max = float(finite.max())
    lp_min = lp_max - viz_range_nats
    lp_2d = np.clip(lp_2d, lp_min, lp_max)
    span = max(lp_max - lp_min, 1e-6)
    norm = (lp_2d - lp_min) / span
    rgba = plt.get_cmap("jet")(norm)
    rgba[..., -1] = norm ** p_alpha_exp
    return rgba, lp_min, lp_max


def draw_pose_marker_xy(ax, xy, yaw_deg, color, arrow_len_m=8.0, dot_size=140,
                        marker="o", edge="white", zorder=20):
    """Marker + yaw arrow in METRIC coords on a metric-extent axis."""
    yaw_rad = np.deg2rad(yaw_deg)
    # In our north-up canvas, yaw=0 -> +y (north).
    # imshow uses origin='upper' with extent=[xmin,xmax,ymin,ymax], so visual "up" is +y.
    dx = np.sin(yaw_rad) * arrow_len_m
    dy = np.cos(yaw_rad) * arrow_len_m
    ax.scatter(xy[0], xy[1], s=dot_size, c=color, edgecolors=edge,
               linewidths=1.5, marker=marker, zorder=zorder)
    ax.annotate(
        "", xy=(xy[0] + dx, xy[1] + dy), xytext=(xy[0], xy[1]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                        mutation_scale=14),
        zorder=zorder + 1,
    )


def draw_pose_marker_uv(ax, uv, yaw_deg, color, arrow_len_px=22, dot_size=110,
                        marker="o", edge="white", zorder=20):
    """Marker + yaw arrow in PIXEL coords (for the per-frame log-P panels)."""
    yaw_rad = np.deg2rad(yaw_deg)
    dx = np.sin(yaw_rad) * arrow_len_px
    dy = -np.cos(yaw_rad) * arrow_len_px
    ax.scatter(uv[0], uv[1], s=dot_size, c=color, edgecolors=edge,
               linewidths=1.5, marker=marker, zorder=zorder)
    ax.annotate(
        "", xy=(uv[0] + dx, uv[1] + dy), xytext=(uv[0], uv[1]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                        mutation_scale=14),
        zorder=zorder + 1,
    )


def draw_pose_candidates(ax, log_probs_3d, top_k=20, min_alpha=0.15,
                         arrow_len_px=16, c_face="white", c_edge="black",
                         zorder=12):
    prob_3d = log_probs_3d.exp()
    yaw_idx = torch.argmax(prob_3d, -1)
    yaws = yaw_idx.numpy() / prob_3d.shape[-1] * 360
    prob_2d = prob_3d.max(-1).values
    prob_2d = (prob_2d / prob_2d.max()).numpy()

    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(prob_2d, size=3) == prob_2d
    local_max &= prob_2d > 0.01
    ys, xs = np.where(local_max)
    if len(xs) == 0:
        return
    vals = prob_2d[ys, xs]
    order = np.argsort(-vals)[:top_k]
    xs, ys, vals = xs[order], ys[order], vals[order]
    yaw_at = yaws[ys, xs]
    for x, y, p, yaw_deg in zip(xs, ys, vals, yaw_at):
        alpha = float(min_alpha + (1 - min_alpha) * p)
        size = 30 + 90 * p
        yaw_rad = np.deg2rad(yaw_deg)
        dx = np.sin(yaw_rad) * arrow_len_px
        dy = -np.cos(yaw_rad) * arrow_len_px
        ax.scatter(x, y, s=size, c=c_face, edgecolors=c_edge,
                   linewidths=0.8, alpha=alpha, zorder=zorder)
        ax.annotate(
            "", xy=(x + dx, y + dy), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=c_edge, lw=1.2,
                            mutation_scale=8, alpha=alpha),
            zorder=zorder + 1,
        )


def compute_loglikelihood_2d(log_probs_3d, viz_range_nats=100.0):
    """2D log-likelihood (max over rotation), clipped to top viz_range_nats below max.

    Why clipping: after many frames the additive aligner gives the belief a huge
    dynamic range (e.g. [-6700, -1.4] at frame 25). Jet on the full range buries
    the sharp peak in red. Clipping to top 100 nats below max keeps the colormap
    meaningful: max=red, max-100=blue, anything lower = clipped to blue. The
    underlying belief is unchanged.
    """
    lp = log_probs_3d.max(-1).values
    arr = lp.numpy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, float("nan"), float("nan")
    lp_max = float(finite.max())
    lp_min_true = float(finite.min())
    lp_min_viz = lp_max - viz_range_nats
    arr_viz = np.clip(arr, lp_min_viz, lp_max)
    return arr_viz, lp_min_true, lp_max


# -------------------------- pages --------------------------

def draw_header_page(pdf, canvas_total, map_total, xy_gt_metric, label,
                     crop_size_m, has_yaw_prior, yaw_prior_range, add_map_mask):
    fig, ax = plt.subplots(figsize=(11, 9))
    extent_total = bbox_to_extent(canvas_total.bbox)
    ax.imshow(map_total, extent=extent_total, origin="upper")
    ax.scatter(*xy_gt_metric.T, c="black", s=12, lw=0)
    ax.plot(*xy_gt_metric.T, c="black", lw=0.8, alpha=0.6,
            label=f"GT trajectory ({len(xy_gt_metric)} frames)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    prior_str = (
        f"yaw_prior = GT_yaw ± {yaw_prior_range:.0f}°"
        if has_yaw_prior else "yaw_prior = None (blind 360° search)"
    )
    bbox_m = canvas_total.bbox.size
    mask_str = (
        "map_mask: ±crop (tight)" if add_map_mask else "map_mask: OFF (broad search)"
    )
    ax.set_title(
        f"{label}\n"
        f"VIS canvas {int(bbox_m[0])}x{int(bbox_m[1])} m  "
        f"(trajectory bbox + {int(crop_size_m + 16)} m)\n"
        f"per-frame SEARCH canvas (moving with camera) = "
        f"{int(2*crop_size_m)}x{int(2*crop_size_m)} m  {mask_str}  {prior_str}",
        fontsize=10,
    )
    fig.text(
        0.5, 0.02,
        "Per-frame pages: input image | map+single-frame likelihood overlay | "
        "map+sequential likelihood overlay  +  feature maps  +  raw log-P panels.\n"
        "Black=GT, Green=single-frame pred, Red=sequential pred.",
        ha="center", fontsize=8,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_frame_page(
    pdf, *, label, frame_idx, total_frames, image_np,
    canvas_total, map_total, per_frame_canvas,
    pred_log_probs, seq_log_probs,
    xy_gt_metric, xy_p_so_far, xy_seq_so_far,
    xy_gt_now, xy_s_now, xy_q_now,
    yaw_gt, yaw_s, yaw_q,
    uvr_single, uvr_seq,
    err_xy_single, err_yaw_single, err_xy_seq, err_yaw_seq,
    features_bev=None, valid_bev=None, map_features=None,
):
    """3 rows × 3 cols.

    Row 1 (trajectory + likelihood overlay in METRIC coords):
        [image]  [map_total + single overlay + trajectories]  [metrics text]
    Row 2 (learned features):
        [BEV feature]  [map feature]  [feature explanation]
    Row 3 (raw log-P heatmaps in PIXEL coords of the per-frame canvas):
        [single log P]  [seq log P]  [legend]
    """
    fig, axes = plt.subplots(3, 3, figsize=(22, 20),
                             gridspec_kw={"width_ratios": [1, 1, 0.4]})
    ax_img, ax_map_single, ax_info = axes[0]
    ax_fbev, ax_fmap, ax_info_feat = axes[1]
    ax_lp_single, ax_lp_seq, ax_info2 = axes[2]
    for a in (ax_info, ax_info_feat, ax_info2):
        a.axis("off")

    extent_total = bbox_to_extent(canvas_total.bbox)
    extent_frame = bbox_to_extent(per_frame_canvas.bbox)

    # --- top-left: input image
    ax_img.imshow(image_np)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"frame {frame_idx+1:03d}/{total_frames}", fontsize=11)

    # --- top-middle: clean OSM map + trajectories + current poses (no heatmap overlay)
    ax_map_single.imshow(map_total, extent=extent_total, origin="upper")
    ax_map_single.plot(*xy_gt_metric.T, c="black", lw=0.8, marker="o", ms=3,
                       mfc="none", zorder=6, label="GT (full)")
    ax_map_single.plot(*xy_p_so_far.T, c="green", lw=1.2, marker="o", ms=3,
                       zorder=7, label=f"single pred (0..{frame_idx})")
    ax_map_single.plot(*xy_seq_so_far.T, c="red", lw=1.2, marker="o", ms=3,
                       zorder=8, label=f"seq pred (0..{frame_idx})")
    draw_pose_marker_xy(ax_map_single, xy_gt_now, yaw_gt, "black",
                        marker="*", dot_size=220, zorder=20)
    draw_pose_marker_xy(ax_map_single, xy_s_now, yaw_s, "green",
                        marker="o", zorder=21)
    draw_pose_marker_xy(ax_map_single, xy_q_now, yaw_q, "red",
                        marker="o", zorder=22)
    ax_map_single.set_aspect("equal")
    ax_map_single.set_xticks([]); ax_map_single.set_yticks([])
    ax_map_single.set_xlim(canvas_total.bbox.min_[0], canvas_total.bbox.max_[0])
    ax_map_single.set_ylim(canvas_total.bbox.min_[1], canvas_total.bbox.max_[1])
    ax_map_single.legend(loc="upper right", fontsize=8)
    ax_map_single.set_title(
        "OSM map + trajectories + current poses  (black=GT, green=single, red=seq)",
        fontsize=10,
    )

    # --- top-right: metrics text
    ax_info.text(
        0.02, 0.95,
        f"frame {frame_idx+1}/{total_frames}\n\n"
        f"GT pose (metric)\n  xy=({xy_gt_now[0]:.1f}, {xy_gt_now[1]:.1f}) m\n"
        f"  yaw={yaw_gt:.1f}°\n\n"
        f"SINGLE pred\n  xy=({xy_s_now[0]:.1f}, {xy_s_now[1]:.1f}) m\n"
        f"  yaw={yaw_s:.1f}°\n"
        f"  Δxy={err_xy_single:.1f} m  Δθ={err_yaw_single:.1f}°\n\n"
        f"SEQ pred\n  xy=({xy_q_now[0]:.1f}, {xy_q_now[1]:.1f}) m\n"
        f"  yaw={yaw_q:.1f}°\n"
        f"  Δxy={err_xy_seq:.1f} m  Δθ={err_yaw_seq:.1f}°",
        fontfamily="monospace", fontsize=10,
        va="top", ha="left", transform=ax_info.transAxes,
    )

    # --- middle row: learned features
    def _squeeze_to_chw(F):
        while F.ndim > 3:
            F = F[0]
        return F

    if features_bev is not None:
        fbev_chw = _squeeze_to_chw(features_bev).cpu().numpy()
        try:
            if valid_bev is not None:
                mask = _squeeze_to_chw(valid_bev.unsqueeze(0)).cpu().numpy().astype(bool)
                if mask.ndim == 3:
                    mask = mask[0]
                fbev_rgb = features_to_RGB(fbev_chw, masks=[mask])[0]
            else:
                fbev_rgb = features_to_RGB(fbev_chw)[0]
            ax_fbev.imshow(fbev_rgb, origin="upper")
            ax_fbev.set_title(
                f"BEV feature (PCA->RGB)  shape={tuple(fbev_chw.shape)}", fontsize=10,
            )
        except Exception as e:
            ax_fbev.text(0.5, 0.5, f"(BEV viz failed: {type(e).__name__})",
                         ha="center", transform=ax_fbev.transAxes)
    ax_fbev.set_xticks([]); ax_fbev.set_yticks([])

    if map_features is not None:
        fmap_chw = _squeeze_to_chw(map_features).cpu().numpy()
        try:
            fmap_rgb = features_to_RGB(fmap_chw)[0]
            ax_fmap.imshow(fmap_rgb, origin="upper")
            ax_fmap.set_title(
                f"map feature (PCA->RGB)  shape={tuple(fmap_chw.shape)}\n"
                "from (areas, ways, nodes) class indices", fontsize=10,
            )
        except Exception as e:
            ax_fmap.text(0.5, 0.5, f"(map viz failed: {type(e).__name__})",
                         ha="center", transform=ax_fmap.transAxes)
    ax_fmap.set_xticks([]); ax_fmap.set_yticks([])

    ax_info_feat.text(
        0.02, 0.95,
        "LEARNED FEATURE PANELS\n\n"
        "PCA(8-D) -> RGB. Similar color =\n"
        "similar position in matching space,\n"
        "loosely tracks semantic class.\n\n"
        "Semantic class indices live in\n"
        "canvas.raster (3 channels):\n"
        "  ch 0 = areas (parks, water, ...)\n"
        "  ch 1 = ways  (road, sidewalk, ...)\n"
        "  ch 2 = nodes (point landmarks)",
        fontfamily="monospace", fontsize=9,
        va="top", ha="left", transform=ax_info_feat.transAxes,
    )

    # --- bottom row: raw log-P heatmaps in per-frame canvas PIXEL coords
    H, W = per_frame_canvas.raster.shape[-2:]
    uv_gt_now_px = per_frame_canvas.to_uv(np.asarray(xy_gt_now))
    uv_s_now_px = (float(uvr_single[0]), float(uvr_single[1]))
    uv_q_now_px = (float(uvr_seq[0]), float(uvr_seq[1]))

    lp_s_2d, lp_s_true_min, lp_s_max = compute_loglikelihood_2d(pred_log_probs)
    im_s = ax_lp_single.imshow(lp_s_2d, cmap="jet", origin="upper")
    draw_pose_candidates(ax_lp_single, pred_log_probs, top_k=20)
    draw_pose_marker_uv(ax_lp_single, uv_gt_now_px, yaw_gt, "black",
                        marker="*", dot_size=200, zorder=20)
    draw_pose_marker_uv(ax_lp_single, uv_s_now_px, yaw_s, "green",
                        marker="o", zorder=21)
    ax_lp_single.set_xticks([]); ax_lp_single.set_yticks([])
    ax_lp_single.set_xlim(0, W); ax_lp_single.set_ylim(H, 0)
    ax_lp_single.set_title(
        f"single-frame  log P(pose)  (per-frame canvas)\n"
        f"true range [{lp_s_true_min:.1f}, {lp_s_max:.1f}]   "
        f"viz clipped to top 100 nats   "
        f"argmax→GT  Δxy={err_xy_single:.1f}m  Δθ={err_yaw_single:.1f}°\n"
        f"(white dots = top-20 candidates, black star=GT, green=pred)",
        fontsize=10,
    )
    plt.colorbar(im_s, ax=ax_lp_single, fraction=0.04, pad=0.02)

    lp_q_2d, lp_q_true_min, lp_q_max = compute_loglikelihood_2d(seq_log_probs)
    im_q = ax_lp_seq.imshow(lp_q_2d, cmap="jet", origin="upper")
    draw_pose_candidates(ax_lp_seq, seq_log_probs, top_k=20)
    draw_pose_marker_uv(ax_lp_seq, uv_gt_now_px, yaw_gt, "black",
                        marker="*", dot_size=200, zorder=20)
    draw_pose_marker_uv(ax_lp_seq, uv_q_now_px, yaw_q, "red",
                        marker="o", zorder=21)
    ax_lp_seq.set_xticks([]); ax_lp_seq.set_yticks([])
    ax_lp_seq.set_xlim(0, W); ax_lp_seq.set_ylim(H, 0)
    ax_lp_seq.set_title(
        f"sequential  log P(pose)  (accumulated through frame {frame_idx+1})\n"
        f"true range [{lp_q_true_min:.1f}, {lp_q_max:.1f}]   "
        f"viz clipped to top 100 nats   "
        f"argmax→GT  Δxy={err_xy_seq:.1f}m  Δθ={err_yaw_seq:.1f}°\n"
        f"(white dots = top-20 candidates, black star=GT, red=pred)",
        fontsize=10,
    )
    plt.colorbar(im_q, ax=ax_lp_seq, fraction=0.04, pad=0.02)

    ax_info2.text(
        0.02, 0.95,
        "LOG-LIKELIHOOD PANELS\n\n"
        "Each pixel:\n"
        "  log P(camera at this pixel)\n"
        "  max-reduced over rotation.\n\n"
        "Colormap auto-scaled per panel;\n"
        "colorbar shows actual log-P range.\n\n"
        "Markers:\n"
        "  black star = GT pose\n"
        "  green dot  = single argmax\n"
        "  red dot    = sequential argmax\n"
        "  white dot  = top-20 candidates\n"
        "    (size/alpha ∝ likelihood)",
        fontfamily="monospace", fontsize=9,
        va="top", ha="left", transform=ax_info2.transAxes,
    )

    fig.suptitle(f"{label}   -   frame {frame_idx+1:03d}/{total_frames}",
                 fontsize=12)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -------------------------- top-level loop --------------------------

def run_and_render(
    model, dm, split, out_pdf: Path, stride: int, label: str,
    max_init_error_rotation, fixed_search: bool = False,
):
    dset, chunk2idx = dm.sequence_dataset(split, max_length=10_000)
    keys = sorted(chunk2idx)
    assert len(keys) == 1, f"expected 1 chunk, got {len(keys)}"
    indices = chunk2idx[keys[0]]
    N = len(indices)
    logger.info(
        "Running %d frames in %s mode", N,
        "FIXED-CENTER (all frames share one canvas, no warping)"
        if fixed_search else "MOVING canvas",
    )

    # GT trajectory in metric coords
    xy_gt_metric = dm.data[split]["t_c2w"][:, :2].numpy()[indices]
    yaws_gt = dm.data[split]["roll_pitch_yaw"][:, -1].numpy()[indices]

    crop_m = float(dm.cfg.crop_size_meters)
    if fixed_search:
        # canvas_total is the SAME as the per-frame search canvas.
        # The dataset already injected gps_position pointing every frame to
        # the same center, so dset[indices[0]].canvas covers the trajectory.
        sample0 = dset[indices[0]]
        canvas_total = sample0["canvas"]
    else:
        bbox_seq = BoundaryBox(xy_gt_metric.min(0), xy_gt_metric.max(0)) + crop_m + 16
        canvas_total = dm.tile_manager.query(bbox_seq)
    map_total = Colormap.apply(canvas_total.raster)
    logger.info(
        "Vis canvas: bbox size = (%.1f, %.1f) m  ->  raster (%d, %d)",
        canvas_total.bbox.size[0], canvas_total.bbox.size[1],
        canvas_total.raster.shape[-2], canvas_total.raster.shape[-1],
    )

    # Storage for trajectories in metric coords
    xy_p_metric = np.zeros((N, 2))
    xy_seq_metric = np.zeros((N, 2))

    stride_set = set(range(0, N, stride)) | {N - 1}
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        draw_header_page(
            pdf, canvas_total, map_total, xy_gt_metric, label,
            crop_size_m=crop_m,
            has_yaw_prior=(max_init_error_rotation is not None),
            yaw_prior_range=(max_init_error_rotation or 0),
            add_map_mask=bool(dm.cfg.add_map_mask),
        )

        aligner = RigidAligner(num_rotations=model.model.conf.num_rotations)
        for i, idx in enumerate(tqdm(indices, desc="seq inference + render")):
            data = dset[idx]
            batch = model.transfer_batch_to_device(collate([data]), model.device, 0)
            pred = model(batch)

            # Proper sequential Bayesian update used in BOTH modes:
            # aligner.update_with_ref warps the previous belief by the camera's
            # motion (xy_source -> xy_target, yaw_source -> yaw_target), then
            # adds the new observation. That warping IS the motion model. In
            # fixed_search mode the canvas is the same every frame, so the warp
            # just shifts the belief by camera motion in canvas pixel coords.
            obs = pred["scores"][0]
            xy_frame = data["canvas"].to_xy(batch["uv"].squeeze(0).double())
            yaw_frame = batch["roll_pitch_yaw"].squeeze(0)[-1].double()
            aligner.update_with_ref(obs, data["canvas"], xy_frame, yaw_frame)
            # Safety: only intervene if NaN/inf actually appears.
            if not torch.isfinite(aligner.belief).all():
                aligner.belief = aligner.belief.nan_to_num(
                    nan=-1e4, posinf=-1e4, neginf=-1e4
                )
            seq_logits = aligner.belief

            uvr_single = pred["uvr_max"][0].detach().cpu()
            xy_p_metric[i] = data["canvas"].to_xy(uvr_single[:2].double()).numpy()

            lps_uvt = log_softmax_spatial(seq_logits)
            uvr_seq = argmax_xyr(lps_uvt).detach().cpu()
            xy_seq_metric[i] = data["canvas"].to_xy(uvr_seq[:2].double()).numpy()

            err_xy_single = float(
                np.linalg.norm(xy_gt_metric[i] - xy_p_metric[i])
            )
            err_xy_seq = float(
                np.linalg.norm(xy_gt_metric[i] - xy_seq_metric[i])
            )
            yaw_gt_t = torch.tensor([yaws_gt[i]], dtype=torch.float64)
            err_yaw_single = float(
                angle_error(yaw_gt_t, uvr_single[2:3].double()).item()
            )
            err_yaw_seq = float(
                angle_error(yaw_gt_t, uvr_seq[2:3].double()).item()
            )

            if i < 5 or i % 25 == 0:
                logger.info(
                    "f%03d  GT xy=(%6.1f,%6.1f) yaw=%6.1f  "
                    "single xy=(%6.1f,%6.1f) yaw=%6.1f Δxy=%5.1fm Δθ=%5.1f°  "
                    "seq xy=(%6.1f,%6.1f) yaw=%6.1f Δxy=%5.1fm Δθ=%5.1f°",
                    i, xy_gt_metric[i, 0], xy_gt_metric[i, 1], yaws_gt[i],
                    xy_p_metric[i, 0], xy_p_metric[i, 1], float(uvr_single[2]),
                    err_xy_single, err_yaw_single,
                    xy_seq_metric[i, 0], xy_seq_metric[i, 1], float(uvr_seq[2]),
                    err_xy_seq, err_yaw_seq,
                )

            if i in stride_set:
                fbev = pred.get("features_bev")
                vbev = pred.get("valid_bev")
                map_feat = (
                    pred["map"]["map_features"]
                    if isinstance(pred.get("map"), dict)
                    and "map_features" in pred["map"]
                    else None
                )

                draw_frame_page(
                    pdf,
                    label=label,
                    frame_idx=i,
                    total_frames=N,
                    image_np=data["image"].permute(1, 2, 0).cpu().numpy(),
                    canvas_total=canvas_total,
                    map_total=map_total,
                    per_frame_canvas=data["canvas"],
                    pred_log_probs=pred["log_probs"][0].detach().cpu(),
                    seq_log_probs=lps_uvt.detach().cpu(),
                    xy_gt_metric=xy_gt_metric,
                    xy_p_so_far=xy_p_metric[: i + 1],
                    xy_seq_so_far=xy_seq_metric[: i + 1],
                    xy_gt_now=xy_gt_metric[i],
                    xy_s_now=xy_p_metric[i],
                    xy_q_now=xy_seq_metric[i],
                    yaw_gt=float(yaws_gt[i]),
                    yaw_s=float(uvr_single[2]),
                    yaw_q=float(uvr_seq[2]),
                    uvr_single=uvr_single,
                    uvr_seq=uvr_seq,
                    err_xy_single=err_xy_single,
                    err_yaw_single=err_yaw_single,
                    err_xy_seq=err_xy_seq,
                    err_yaw_seq=err_yaw_seq,
                    features_bev=(fbev[0].detach() if fbev is not None else None),
                    valid_bev=(vbev[0].detach() if vbev is not None else None),
                    map_features=(map_feat[0].detach() if map_feat is not None else None),
                )

            del pred, batch, lps_uvt, data
            torch.cuda.empty_cache()

    rendered = sum(1 for i in range(N) if i in stride_set)
    logger.info("Wrote %s  (1 header + %d frame pages of %d total).",
                out_pdf, rendered, N)


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    ap.add_argument(
        "--num_rotations", type=int, default=64,
        help="Default 64 (training value). Bump to 128/256 for sharper yaw.",
    )
    ap.add_argument(
        "--crop_size_meters", type=float, default=96.0,
        help="Per-frame search canvas half-width in meters. Default 96 -> 192m x "
             "192m search canvas (covers ~3 buildings on campus). Use 64 for a "
             "tighter notebook-style search, or 128 to span the whole campus area "
             "(more memory).",
    )
    ap.add_argument(
        "--add_map_mask", action="store_true",
        help="Enable the map_mask (mask_radius defaults to max_init_error). "
             "Default OFF for broad spatial search.",
    )
    ap.add_argument(
        "--fixed_search", action="store_true",
        help="FIXED search canvas centered at trajectory centroid. All frames "
             "share the same canvas. Sequential filtering still uses the "
             "RigidAligner with motion-model warping (belief is shifted by "
             "camera motion in canvas pixel coords each frame), so seq_pred "
             "tracks the current camera position just like moving-canvas mode "
             "but on a single fixed map.",
    )
    ap.add_argument(
        "--max_init_error_rotation", type=int, default=10,
        help="Yaw-prior half-width around GT yaw. Default 10 (notebook KITTI default).",
    )
    ap.add_argument(
        "--no_yaw_prior", action="store_true",
        help="Disable the yaw prior entirely (model searches the full 360 deg).",
    )
    ap.add_argument(
        "--stride", type=int, default=5,
        help="Render every Nth frame. Inference runs on every frame.",
    )
    ap.add_argument(
        "--splits_filename", type=str, default="splits_balanced.json",
        help="JSON file inside datasets/gist_abc/ that defines train/val/test "
             "lists.",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    seed_everything(0)
    torch.set_grad_enabled(False)

    max_init_error_rotation = None if args.no_yaw_prior else args.max_init_error_rotation

    model = load_model(args.experiment, args.num_rotations)
    if args.fixed_search:
        dm = build_dataset_fixed_center(
            crop_size_meters=args.crop_size_meters,
            max_init_error_rotation=max_init_error_rotation,
            splits_filename=args.splits_filename,
        )
        # Inject the trajectory centroid as the fixed canvas center for every frame.
        xy_all = dm.data[args.split]["t_c2w"][:, :2].numpy()
        center_xy = xy_all.mean(0)
        logger.info(
            "Fixed-search center (trajectory centroid): xy=(%.2f, %.2f) m  "
            "canvas half-width=%.1f m",
            center_xy[0], center_xy[1], float(dm.cfg.crop_size_meters),
        )
        inject_fixed_center(dm, args.split, center_xy)
    else:
        dm = build_dataset(
            crop_size_meters=args.crop_size_meters,
            max_init_error_rotation=max_init_error_rotation,
            add_map_mask=args.add_map_mask,
            splits_filename=args.splits_filename,
        )

    out = args.output
    if out.suffix.lower() != ".pdf":
        out = out.with_suffix(".pdf")
        logger.warning("Coercing output extension to .pdf -> %s", out)

    label = (
        args.experiment
        if args.experiment in pretrained_models
        else Path(args.experiment).name
    )
    run_and_render(
        model, dm, args.split, out, stride=args.stride, label=label,
        max_init_error_rotation=max_init_error_rotation,
        fixed_search=args.fixed_search,
    )


if __name__ == "__main__":
    main()
