#!/usr/bin/env python3
"""Publication-style matplotlib figures for server PINN-TFI outputs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "outputs_server"

BOUNDARY = np.array(
    [
        [0.0, -1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [0.0, -1.0],
    ]
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def closed_lshape_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    in_box = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    in_removed_quadrant = (x > 0.0) & (y < 0.0)
    return in_box & (~in_removed_quadrant)


def lshape_polygon_patch(ax) -> PathPatch:
    return PathPatch(MplPath(BOUNDARY), transform=ax.transData, facecolor="none", edgecolor="none")


def apply_clip(contour_set, patch: PathPatch) -> None:
    if hasattr(contour_set, "set_clip_path"):
        contour_set.set_clip_path(patch)
        return
    for collection in getattr(contour_set, "collections", []):
        collection.set_clip_path(patch)


def fill_plot_field(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    plot_mask = closed_lshape_mask(x, y)
    if np.all(np.isfinite(data[plot_mask])):
        filled = np.array(data, dtype=float, copy=True)
        filled[~plot_mask] = np.nan
    else:
        filled = fill_boundary_nans(data, plot_mask)
    if np.any(~np.isfinite(filled)):
        vals = filled[np.isfinite(filled)]
        fallback = float(np.mean(vals)) if vals.size else 0.0
        filled = np.where(np.isfinite(filled), filled, fallback)
    return filled


def lshape_triangulation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[mtri.Triangulation, np.ndarray]:
    base_mask = closed_lshape_mask(x, y)
    if np.all(np.isfinite(z[base_mask])):
        filled = np.array(z, dtype=float, copy=True)
        filled[~base_mask] = np.nan
    else:
        filled = fill_boundary_nans(z, base_mask)
    inside = base_mask & np.isfinite(filled)
    points_x = x[inside].ravel()
    points_y = y[inside].ravel()
    points_z = filled[inside].ravel()

    boundary_z = []
    for bx, by in BOUNDARY[:-1]:
        dist2 = (points_x - bx) ** 2 + (points_y - by) ** 2
        boundary_z.append(points_z[int(np.argmin(dist2))])
    points_x = np.concatenate([points_x, BOUNDARY[:-1, 0]])
    points_y = np.concatenate([points_y, BOUNDARY[:-1, 1]])
    points_z = np.concatenate([points_z, np.asarray(boundary_z)])

    triang = mtri.Triangulation(points_x, points_y)
    tri = triang.triangles
    cx = points_x[tri].mean(axis=1)
    cy = points_y[tri].mean(axis=1)
    edge_lengths = np.sqrt(
        np.maximum.reduce(
            [
                (points_x[tri[:, 0]] - points_x[tri[:, 1]]) ** 2 + (points_y[tri[:, 0]] - points_y[tri[:, 1]]) ** 2,
                (points_x[tri[:, 1]] - points_x[tri[:, 2]]) ** 2 + (points_y[tri[:, 1]] - points_y[tri[:, 2]]) ** 2,
                (points_x[tri[:, 2]] - points_x[tri[:, 0]]) ** 2 + (points_y[tri[:, 2]] - points_y[tri[:, 0]]) ** 2,
            ]
        )
    )
    max_allowed = 4.0 / max(x.shape[0] - 1, 1)
    triang.set_mask((~closed_lshape_mask(cx, cy)) | (edge_lengths > max_allowed))
    return triang, points_z


def triangle_face_values(triang: mtri.Triangulation, values: np.ndarray) -> np.ndarray:
    return values[triang.triangles].mean(axis=1)


def fill_boundary_nans(data: np.ndarray, plot_mask: np.ndarray) -> np.ndarray:
    filled = np.array(data, dtype=float, copy=True)
    filled[~plot_mask] = np.nan
    for _ in range(4):
        missing = plot_mask & (~np.isfinite(filled))
        if not np.any(missing):
            break
        new = filled.copy()
        count = np.zeros_like(filled, dtype=float)
        accum = np.zeros_like(filled, dtype=float)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.roll(np.roll(filled, di, axis=0), dj, axis=1)
            if di > 0:
                shifted[:di, :] = np.nan
            elif di < 0:
                shifted[di:, :] = np.nan
            if dj > 0:
                shifted[:, :dj] = np.nan
            elif dj < 0:
                shifted[:, dj:] = np.nan
            valid = missing & np.isfinite(shifted)
            accum[valid] += shifted[valid]
            count[valid] += 1.0
        fillable = missing & (count > 0.0)
        new[fillable] = accum[fillable] / count[fillable]
        filled = new
    return filled


def masked_field(data: np.ndarray, plot_mask: np.ndarray) -> np.ma.MaskedArray:
    filled = fill_boundary_nans(data, plot_mask)
    return np.ma.masked_where((~plot_mask) | (~np.isfinite(filled)), filled)


def finite_values(field: np.ndarray, active: np.ndarray) -> np.ndarray:
    vals = field[active & np.isfinite(field)]
    return vals[np.isfinite(vals)]


def format_linear_colorbar(cbar, vmin: float, vmax: float, fmt: str = "%.3f") -> None:
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    cbar.ax.tick_params(labelsize=12, direction="in")


def format_log_colorbar(cbar) -> None:
    vmin, vmax = cbar.mappable.norm.vmin, cbar.mappable.norm.vmax
    locator_ticks = LogLocator(base=10.0, numticks=6).tick_values(vmin, vmax)
    locator_ticks = locator_ticks[(locator_ticks >= vmin) & (locator_ticks <= vmax)]
    if vmin > 0.0 and vmax > vmin:
        lo = np.log10(vmin)
        hi = np.log10(vmax)
        min_gap = 0.08 * max(hi - lo, 1.0)
        locator_ticks = np.array(
            [t for t in locator_ticks if abs(np.log10(t) - lo) > min_gap and abs(np.log10(t) - hi) > min_gap]
        )
    ticks = np.unique(np.concatenate(([vmin], locator_ticks, [vmax])))
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    cbar.ax.tick_params(labelsize=12, direction="in")


def save_pdf_png(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def style_lshape_axes(ax: plt.Axes) -> None:
    ax.plot(BOUNDARY[:, 0], BOUNDARY[:, 1], color="black", lw=1.4, solid_joinstyle="miter")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])


def plot_solution_contour(npz: np.lib.npyio.NpzFile, case_id: int, out_base: Path) -> None:
    x = npz["X"]
    y = npz["Y"]
    z = np.array(npz["U_pred"], dtype=float)
    triang, points_z = lshape_triangulation(x, y, z)
    zmin = float(np.nanmin(points_z))
    zmax = float(np.nanmax(points_z))
    contour_levels = np.linspace(zmin, zmax, 32)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cf = ax.tricontourf(
        triang,
        points_z,
        levels=contour_levels,
        cmap="jet",
        vmin=zmin,
        vmax=zmax,
    )
    style_lshape_axes(ax)
    ax.set_title(rf"Case {case_id}: $u_{{PINN}}$ on the L-shaped domain")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"$u_{PINN}$")
    format_linear_colorbar(cbar, zmin, zmax)
    save_pdf_png(fig, out_base.with_name(out_base.name + "_solution_contour"))


def plot_solution_surface(npz: np.lib.npyio.NpzFile, case_id: int, out_base: Path) -> None:
    x = npz["X"]
    y = npz["Y"]
    z = np.array(npz["U_pred"], dtype=float)
    triang, points_z = lshape_triangulation(x, y, z)

    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        triang,
        points_z,
        cmap="jet",
        linewidth=0.0,
        edgecolor="none",
        antialiased=False,
        shade=False,
    )
    surf.set_edgecolor("face")
    zmin = float(np.nanmin(points_z))
    ax.plot(BOUNDARY[:, 0], BOUNDARY[:, 1], zs=np.full(len(BOUNDARY), zmin), color="black", lw=1.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"$x$", labelpad=4)
    ax.set_ylabel(r"$y$", labelpad=4)
    ax.set_zlabel(r"$u_{PINN}$", labelpad=4)
    ax.set_title(rf"Case {case_id}: solution surface", pad=10)
    ax.view_init(elev=32, azim=-125)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 0.42))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color("0.25")
    ax.tick_params(axis="both", which="major", pad=1)
    cbar = fig.colorbar(surf, ax=ax, fraction=0.04, pad=0.06, shrink=0.78)
    cbar.set_label(r"$u_{PINN}$")
    format_linear_colorbar(cbar, float(np.nanmin(points_z)), float(np.nanmax(points_z)))
    save_pdf_png(fig, out_base.with_name(out_base.name + "_solution_surface"))


def plot_error_contour(npz: np.lib.npyio.NpzFile, case_id: int, out_base: Path) -> None:
    x = npz["X"]
    y = npz["Y"]
    err_raw = np.array(npz["U_err"], dtype=float)
    plot_mask = closed_lshape_mask(x, y)
    err_filled = fill_plot_field(err_raw, x, y)
    vals = err_filled[plot_mask & np.isfinite(err_filled)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return
    vmin = max(float(vals.min()), 1.0e-12)
    vmax = float(vals.max())
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 32)
    err = np.ma.masked_where((~plot_mask) | (~np.isfinite(err_filled)), np.maximum(err_filled, vmin))

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    err_triang, err_values = lshape_triangulation(x, y, np.asarray(err.filled(np.nan), dtype=float))
    err_values = np.maximum(err_values, vmin)
    cf = ax.tricontourf(
        err_triang,
        err_values,
        levels=contour_levels,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="jet",
    )
    style_lshape_axes(ax)
    ax.set_title(rf"Case {case_id}: pointwise error")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"$|u_{PINN}-u_{ref}|$")
    format_log_colorbar(cbar)
    save_pdf_png(fig, out_base.with_name(out_base.name + "_error_contour"))


def read_history(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {
        "epoch": [],
        "loss": [],
        "residual_rms": [],
        "boundary_max": [],
        "rel_l2": [],
    }
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key in cols:
                value = row[key]
                cols[key].append(float(value) if value.lower() != "nan" else np.nan)
    return {key: np.array(value, dtype=float) for key, value in cols.items()}


def plot_loss(history_csv: Path, case_id: int, out_base: Path) -> None:
    hist = read_history(history_csv)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogy(hist["epoch"], hist["loss"], color="#1f77b4", lw=1.8, label="loss")
    ax.semilogy(hist["epoch"], hist["boundary_max"], color="#2ca02c", lw=1.4, ls="--", label="boundary max")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(rf"Case {case_id}: training history")
    ax.grid(True, which="both", color="0.86", lw=0.5)
    ax.legend(frameon=False, ncol=2)
    save_pdf_png(fig, out_base.with_name(out_base.name + "_loss"))

def plot_case(npz_path: Path, plots_root: Path) -> None:
    case_id = int(npz_path.parent.name.replace("case", ""))
    history_csv = npz_path.with_name(npz_path.stem + "_history.csv")
    if not history_csv.exists():
        raise FileNotFoundError(history_csv)
    case_dir = plots_root / f"case{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    out_base = case_dir / npz_path.stem

    with np.load(npz_path, allow_pickle=True) as npz:
        plot_solution_contour(npz, case_id, out_base)
        plot_solution_surface(npz, case_id, out_base)
        if case_id in (1, 2) and np.isfinite(npz["U_err"][npz["active"].astype(bool)]).any():
            plot_error_contour(npz, case_id, out_base)
    plot_loss(history_csv, case_id, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render matplotlib PDF/PNG plots for outputs_server.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()

    configure_matplotlib()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    plots_root = root / "plots_mpl"
    if plots_root.exists():
        shutil.rmtree(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    case_dirs = ["case1", "case2", "case3"] if args.case == "all" else [f"case{args.case}"]
    npz_paths: list[Path] = []
    for case_dir in case_dirs:
        npz_paths.extend(sorted((root / case_dir).glob("*.npz")))
    if not npz_paths:
        raise RuntimeError(f"No npz files found under {root}")

    for npz_path in npz_paths:
        plot_case(npz_path, plots_root)
    print(f"Rendered {len(npz_paths)} case(s) into {plots_root}")


if __name__ == "__main__":
    main()
