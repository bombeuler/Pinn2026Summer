#!/usr/bin/env python3
r"""Draw a single publication-style geometry sketch for Case 3.

The figure shows the L-shaped domain, the removed quadrant, the reentrant
corner, and the Dirac line source Gamma used in pinn_lshape_tfi_all.py.

Usage
-----
    pixi run python lshape_tfi/plot_case3_example.py
    pixi run python lshape_tfi/plot_case3_example.py --out outputs_pinn_tfi_4060/plots_case3_example
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from pinn_lshape_tfi_all import GAMMA_SEGMENTS
from plot_server_results_mpl import BOUNDARY, configure_matplotlib, save_pdf_png


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "outputs_pinn_tfi_4060" / "plots_case3_example"


def draw_gamma(ax: plt.Axes) -> None:
    for (ax0, ay0), (bx, by), _ in GAMMA_SEGMENTS:
        ax.plot(
            [ax0, bx],
            [ay0, by],
            color="#FF6B6B",
            lw=3.2,
            solid_capstyle="round",
            zorder=5,
        )
    pts = np.array([p for seg in GAMMA_SEGMENTS for p in seg[:2]], dtype=float)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=14,
        color="#FF6B6B",
        edgecolor="white",
        lw=0.5,
        zorder=6,
    )


def plot_case3_example(out_dir: Path) -> None:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    ax.add_patch(
        PathPatch(
            MplPath(BOUNDARY),
            facecolor="#EAF6FF",
            edgecolor="#243B53",
            lw=1.6,
            joinstyle="miter",
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, -1.0),
            1.0,
            1.0,
            facecolor="white",
            edgecolor="none",
            lw=0.0,
            zorder=2,
        )
    )
    ax.plot(BOUNDARY[:, 0], BOUNDARY[:, 1], color="#243B53", lw=1.6, solid_joinstyle="miter", zorder=4)

    draw_gamma(ax)
    ax.text(-0.55, 0.45, r"$\Omega$", fontsize=23, ha="center", va="center", color="#243B53")
    ax.text(-0.32, -0.36, r"$\gamma_{l}$", fontsize=20, color="#FF6B6B", ha="left", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.axis("off")

    save_pdf_png(fig, out_dir / "case3_example_geometry")
    print(f"Saved {out_dir / 'case3_example_geometry.pdf'}")
    print(f"Saved {out_dir / 'case3_example_geometry.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the Case 3 geometry example figure.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()
    plot_case3_example(args.out.resolve())


if __name__ == "__main__":
    main()
