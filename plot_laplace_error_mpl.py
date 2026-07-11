#!/usr/bin/env python3
"""Horizontal overview of pointwise Laplace residual errors for cases 1--3."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_server_results_mpl import (
    configure_matplotlib,
    format_log_colorbar,
    lshape_triangulation,
    save_pdf_png,
    style_lshape_axes,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "outputs_pinn_tfi_final"


def find_case_npz(root: Path, case_id: int) -> Path:
    files = sorted((root / f"case{case_id}").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found under {root / f'case{case_id}'}")
    return files[0]


def plot_residual_panel(ax: plt.Axes, npz_path: Path, case_id: int) -> None:
    with np.load(npz_path, allow_pickle=True) as npz:
        x = npz["X"]
        y = npz["Y"]
        residual = np.abs(np.array(npz["U_residual"], dtype=float))

    finite = residual[np.isfinite(residual)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        raise RuntimeError(f"No positive finite residual values in {npz_path}")
    vmin = max(float(np.nanpercentile(positive, 2.0)), 1.0e-12)
    vmax = float(np.nanmax(positive))
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 32)
    plot_field = np.maximum(residual, vmin)
    triang, values = lshape_triangulation(x, y, plot_field)
    values = np.maximum(values, vmin)

    cf = ax.tricontourf(
        triang,
        values,
        levels=levels,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="jet",
        extend="min",
    )
    style_lshape_axes(ax)
    title = rf"Case {case_id}: $|-\Delta u_{{PINN}}-f|$"
    cbar_label = rf"$|-\Delta u_{{PINN}}-f|$"
    if case_id == 3:
        title = rf"Case {case_id}: $|-\Delta u_r|$"
        cbar_label = rf"$|-\Delta u_r|$"
    ax.set_title(title)
    cbar = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(cbar_label)
    format_log_colorbar(cbar)


def plot_laplace_error_overview(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1), constrained_layout=True)
    for ax, case_id in zip(axes, (1, 2, 3), strict=True):
        plot_residual_panel(ax, find_case_npz(root, case_id), case_id)
    fig.suptitle("Pointwise Laplace residual error", fontsize=16)
    save_pdf_png(fig, out_dir / "laplace_error_cases123")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 1x3 Laplace residual error overview.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    configure_matplotlib()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    out_dir = root / "plots_laplace_error"
    plot_laplace_error_overview(root, out_dir)
    print(f"Rendered Laplace residual overview into {out_dir}")


if __name__ == "__main__":
    main()
