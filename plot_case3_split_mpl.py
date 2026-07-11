#!/usr/bin/env python3
"""Publication-style split plots for case 3 PINN-TFI outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_server_results_mpl import (
    configure_matplotlib,
    format_linear_colorbar,
    lshape_triangulation,
    save_pdf_png,
    style_lshape_axes,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "outputs_pinn_tfi_final"


def plot_linear_field(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    title: str,
    cbar_label: str,
) -> None:
    triang, values = lshape_triangulation(x, y, field)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    levels = np.linspace(vmin, vmax, 32)
    cf = ax.tricontourf(triang, values, levels=levels, cmap="jet", vmin=vmin, vmax=vmax)
    style_lshape_axes(ax)
    ax.set_title(title)
    cbar = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(cbar_label)
    format_linear_colorbar(cbar, vmin, vmax)


def plot_case3_split(npz_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / npz_path.stem
    with np.load(npz_path, allow_pickle=True) as npz:
        required = ("X", "Y", "U_s", "U_regular")
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(f"{npz_path} missing required fields: {missing}")
        x = npz["X"]
        y = npz["Y"]
        u_s = np.array(npz["U_s"], dtype=float)
        u_r = np.array(npz["U_regular"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    plot_linear_field(axes[0], x, y, u_s, r"Singular part $u_s$", r"$u_s$")
    plot_linear_field(axes[1], x, y, u_r, r"Regular part $u_r$", r"$u_r$")
    fig.suptitle("Case 3 singular decomposition", fontsize=16)
    save_pdf_png(fig, out_base.with_name(out_base.name + "_split_contours"))


def find_case3_npz(root: Path) -> list[Path]:
    case3_dir = root / "case3"
    return sorted(case3_dir.glob("*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render case 3 singular decomposition plots.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--npz", type=Path, default=None, help="Optional single case3 npz file.")
    args = parser.parse_args()

    configure_matplotlib()
    root = args.root.resolve()
    if args.npz is not None:
        npz_paths = [args.npz.resolve()]
        out_dir = root / "plots_case3_split"
    else:
        if not root.exists():
            raise FileNotFoundError(root)
        npz_paths = find_case3_npz(root)
        out_dir = root / "plots_case3_split"
    if not npz_paths:
        raise RuntimeError(f"No case3 npz files found under {root / 'case3'}")

    for npz_path in npz_paths:
        plot_case3_split(npz_path, out_dir)
    print(f"Rendered {len(npz_paths)} case3 split plot(s) into {out_dir}")


if __name__ == "__main__":
    main()
