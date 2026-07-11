r"""
Compare PINN TFI model outputs (4060 run, float64) against high-precision FD
references on fine grids (801x801 and 1601x1601).

For each case x n_fd:
  1. Load the trained .pt checkpoint, rebuild TFIModel with float64.
  2. Load the FD reference from reference_case{id}_{n_fd}.npz.
  3. Sampling set = all grid points where U_ref_fd is finite
     (automatically excludes outer-boundary NaN, domain-exterior NaN,
      and any Dirac-line-source NaN if present — no manual exclusion).
  4. Evaluate the PINN model directly on those coordinates (batch, no_grad).
  5. Compute rel_L2 and max_err over the FULL sampling set — no region
     exclusion whatsoever (no corner cutout, no gamma cutout).
  6. Save comparison .npz and 3 publication-style plots (solution contour,
     reference contour, error contour) into {root}/plots_fd{n_fd}/case{id}/.

Usage
-----
    pixi run python lshape_tfi/compare_pinn_vs_fd_4060.py
    pixi run python lshape_tfi/compare_pinn_vs_fd_4060.py --case 3
    pixi run python lshape_tfi/compare_pinn_vs_fd_4060.py --n-fd 801 1601
    pixi run python lshape_tfi/compare_pinn_vs_fd_4060.py --root /path/to/other_outputs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pinn_lshape_tfi_all import case2_exact_np, configure_dtype, make_model  # noqa: E402

# ---- import publication-style helpers from plot_server_results_mpl ----
from plot_server_results_mpl import (  # noqa: E402
    BOUNDARY,
    closed_lshape_mask,
    configure_matplotlib,
    fill_plot_field,
    format_linear_colorbar,
    format_log_colorbar,
    lshape_triangulation,
    save_pdf_png,
    style_lshape_axes,
)


# ============================================================================
# Model loading / evaluation
# ============================================================================

def load_pinn_model(case_id: int, ckpt_path: str, dtype: str = "float64"):
    configure_dtype(dtype)
    cfg = _infer_model_config(ckpt_path)
    model = make_model(
        case_id,
        hidden=cfg.get("hidden", 64),
        depth=cfg.get("depth", 4),
        arch=cfg.get("arch", "mlp"),
        network_input=cfg.get("network_input", "lambda"),
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


def _infer_model_config(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", ck.get("final_metrics", {}))
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    return cfg


def batched_eval(model, x_np, y_np, batch: int = 8192, dtype=torch.float64):
    N = len(x_np)
    out = np.empty(N, dtype=np.float64)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        xt = torch.tensor(x_np[s:e], dtype=dtype)
        yt = torch.tensor(y_np[s:e], dtype=dtype)
        with torch.no_grad():
            u = model(xt, yt)
        out[s:e] = u.detach().cpu().numpy().astype(np.float64)
    return out


# ============================================================================
# Path helpers
# ============================================================================

def find_pinn_npz(root: Path, case_id: int) -> str:
    d = root / f"case{case_id}"
    files = sorted(glob.glob(str(d / "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {d}")
    return files[0]


def find_pinn_pt(root: Path, case_id: int) -> str:
    d = root / f"case{case_id}"
    files = sorted(glob.glob(str(d / "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt in {d}")
    return files[0]


def find_fd_reference(case_id: int, n_fd: int = 801) -> Path:
    p = SCRIPT_DIR / f"reference_case{case_id}_{n_fd}.npz"
    if not p.exists():
        raise FileNotFoundError(f"FD reference not found: {p}")
    return p


# ============================================================================
# Plotting (publication style, imported from plot_server_results_mpl)
# ============================================================================

def _downsample_grid(xg, U, target_n: int = 401):
    """Stride-subsample a square grid to ~target_n x target_n for plotting."""
    n = len(xg)
    stride = max(1, (n - 1) // (target_n - 1))
    xs = xg[::stride]
    Us = U[::stride].T  # transpose for matplotlib (x=columns convention handled by triangulation)
    # Actually triangulation takes meshgrid arrays directly; keep ij convention
    Us = U[::stride, ::stride]
    xs = xg[::stride]
    return xs, Us


def plot_solution_contour(xg, U, case_id: int, out_base: Path, tag: str) -> None:
    """tricontourf of a solution field with L-shape clip, publication style."""
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    x_ds, U_ds = _downsample_grid(xg, U, 401)
    X_ds, Y_ds = np.meshgrid(x_ds, x_ds, indexing="ij")
    triang, points_z = lshape_triangulation(X_ds, Y_ds, U_ds)
    zmin = float(np.nanmin(points_z))
    zmax = float(np.nanmax(points_z))
    levels = np.linspace(zmin, zmax, 32)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cf = ax.tricontourf(
        triang, points_z, levels=levels, cmap="jet", vmin=zmin, vmax=zmax,
    )
    style_lshape_axes(ax)
    label = r"$u_{PINN}$" if tag == "pinn" else r"$u_{\mathrm{ref}}$"
    if tag == "pinn":
        title = rf"Case {case_id}: PINN $u_{{PINN}}$"
    elif case_id == 2:
        title = rf"Case {case_id}: exact reference"
    else:
        title = rf"Case {case_id}: FD reference"
    ax.set_title(title)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(label)
    format_linear_colorbar(cbar, zmin, zmax)
    save_pdf_png(fig, out_base.with_name(out_base.name + f"_{tag}_contour"))


def plot_error_contour(xg, U_err, case_id: int, out_base: Path) -> None:
    """LogNorm tricontourf of pointwise error, publication style."""
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    mask = closed_lshape_mask(X, Y)
    err_filled = fill_plot_field(U_err, X, Y)
    vals = err_filled[mask & np.isfinite(err_filled)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return
    vmax = float(vals.max())
    vmin = max(float(np.percentile(vals, 2)), 1.0e-12)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 32)

    x_ds, err_ds = _downsample_grid(xg, U_err, 401)
    X_ds, Y_ds = np.meshgrid(x_ds, x_ds, indexing="ij")
    err_filled_ds = fill_plot_field(err_ds, X_ds, Y_ds)
    err_ds_clipped = np.maximum(err_filled_ds, vmin)
    triang, err_values = lshape_triangulation(X_ds, Y_ds, err_ds_clipped)
    err_values = np.maximum(err_values, vmin)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cf = ax.tricontourf(
        triang, err_values, levels=levels, norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="jet", extend="min",
    )
    style_lshape_axes(ax)
    ax.set_title(rf"Case {case_id}: pointwise error")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"$|u_{PINN}-u_{\mathrm{ref}}|$")
    format_log_colorbar(cbar)
    save_pdf_png(fig, out_base.with_name(out_base.name + "_error_contour"))


# ============================================================================
# Main compare logic
# ============================================================================

def compare_case(
    case_id: int,
    root: Path,
    n_fd: int = 801,
    save_outputs: bool = True,
    plots_dir: Path | None = None,
):
    pt_path = find_pinn_pt(root, case_id)
    npz_path = find_pinn_npz(root, case_id)
    fd_path = find_fd_reference(case_id, n_fd)

    print(f"\n{'='*70}")
    print(f"Case {case_id}  (FD {n_fd}x{n_fd})")
    print(f"  PINN: {os.path.basename(pt_path)}")
    print(f"  FD  : {fd_path.name}")
    print(f"{'='*70}")

    # ---- FD reference ----
    fd = np.load(str(fd_path), allow_pickle=True)
    xg_fd = fd["xg"]
    U_ref_fd = fd["U_ref"]
    h = float(xg_fd[1] - xg_fd[0])
    print(f"  FD grid: {len(xg_fd)}x{len(xg_fd)}  h={h:.6f}")

    # ---- sampling set ----
    X, Y = np.meshgrid(xg_fd, xg_fd, indexing="ij")
    finite_mask = np.isfinite(U_ref_fd)
    ref_source = "FD"
    if case_id == 2:
        U_ref_exact = np.full_like(U_ref_fd, np.nan)
        U_ref_exact[finite_mask] = case2_exact_np(X[finite_mask], Y[finite_mask])
        U_ref_fd = U_ref_exact
        ref_source = "exact"
        print("  reference: analytic exact solution on FD grid")
    xa = X[finite_mask].copy()
    ya = Y[finite_mask].copy()
    ur = U_ref_fd[finite_mask].copy()
    n_pts = len(xa)
    print(f"  sampling on {n_pts} finite-ref points (no exclusion)")

    # ---- model ----
    t0 = time.time()
    model, cfg = load_pinn_model(case_id, pt_path, dtype="float64")
    print(f"  model loaded ({time.time()-t0:.2f}s)  "
          f"arch={cfg.get('arch')} h={cfg.get('hidden')} d={cfg.get('depth')}")

    # ---- evaluate ----
    t0 = time.time()
    up = batched_eval(model, xa, ya, batch=8192, dtype=torch.float64)
    eval_sec = time.time() - t0
    nan_pinn = int(np.isnan(up).sum())
    print(f"  PINN sampled {n_pts} pts in {eval_sec:.2f}s  (NaN: {nan_pinn})")

    # ---- metrics ----
    diff = up - ur
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(ur))
    max_err = float(np.max(np.abs(diff)))
    rms_err = float(np.sqrt(np.mean(diff ** 2)))
    max_up = float(np.max(np.abs(up)))
    max_ur = float(np.max(np.abs(ur)))
    print(f"  rel_L2  = {rel_l2:.6e}")
    print(f"  max_err = {max_err:.6e}")
    print(f"  rms_err = {rms_err:.6e}")
    print(f"  max|U_pinn|={max_up:.6e}  max|U_ref|={max_ur:.6e}")

    # ---- reconstruct full grids ----
    U_pinn_grid = np.full_like(U_ref_fd, np.nan)
    U_err_grid = np.full_like(U_ref_fd, np.nan)
    U_pinn_grid[finite_mask] = up
    U_err_grid[finite_mask] = np.abs(diff)

    # ---- save npz + plots ----
    if save_outputs and plots_dir is not None:
        case_plots = plots_dir / f"case{case_id}"
        case_plots.mkdir(parents=True, exist_ok=True)
        run_stem = Path(pt_path).stem
        out_base = case_plots / f"{run_stem}_vs_fd{n_fd}"

        np.savez_compressed(
            str(out_base.with_suffix(".npz")),
            xg_fd=xg_fd,
            U_pinn_fd=U_pinn_grid,
            U_ref_fd=U_ref_fd,
            U_err_fd=U_err_grid,
            finite_mask=finite_mask,
            n_pts=n_pts,
            rel_l2_fd=rel_l2,
            max_err_fd=max_err,
            rms_err_fd=rms_err,
            max_abs_upinn=max_up,
            max_abs_uref=max_ur,
            nan_in_pinn=nan_pinn,
            eval_sec=eval_sec,
            config_json=json.dumps({
                "case": case_id,
                "fd_grid_n": n_fd,
                "mesh_h": h,
                "model_config": cfg,
                "pinn_npz": os.path.basename(npz_path),
                "fd_npz": fd_path.name,
                "reference_source": ref_source,
                "exclusion": "none",
            }),
        )
        print(f"  saved {out_base.name}.npz")

        # 3 plots: solution, reference, error
        plot_solution_contour(xg_fd, U_pinn_grid, case_id, out_base, tag="pinn")
        print(f"  saved {out_base.name}_pinn_contour.pdf/.png")
        plot_solution_contour(xg_fd, U_ref_fd, case_id, out_base, tag="ref")
        print(f"  saved {out_base.name}_ref_contour.pdf/.png")
        plot_error_contour(xg_fd, U_err_grid, case_id, out_base)
        print(f"  saved {out_base.name}_error_contour.pdf/.png")

    return {
        "case": case_id,
        "n_fd": n_fd,
        "n_pts": n_pts,
        "rel_l2": rel_l2,
        "max_err": max_err,
        "rms_err": rms_err,
        "max_abs_upinn": max_up,
        "max_abs_uref": max_ur,
        "nan_in_pinn": nan_pinn,
        "eval_sec": eval_sec,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare PINN TFI (float64) vs FD references. "
                    "Publication-style plots + error npz."
    )
    p.add_argument(
        "--root", type=Path,
        default=SCRIPT_DIR / "outputs_pinn_tfi_4060",
        help="root directory containing case1/case2/case3 subdirs "
             "(default: outputs_pinn_tfi_4060)",
    )
    p.add_argument("--case", choices=("1", "2", "3", "all"), default="all")
    p.add_argument(
        "--n-fd", nargs="+", type=int, default=[801, 1601],
        help="FD reference grid resolution(s) (default: 801 1601)",
    )
    p.add_argument("--no-save", action="store_true",
                   help="skip saving .npz/.png (just print metrics)")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    cases = [1, 2, 3] if args.case == "all" else [int(args.case)]
    save = not args.no_save

    configure_matplotlib()

    all_results = []
    for n_fd in args.n_fd:
        plots_dir = root / f"plots_fd{n_fd}"
        if save and plots_dir.exists():
            shutil.rmtree(plots_dir)
        if save:
            plots_dir.mkdir(parents=True, exist_ok=True)
        for cid in cases:
            r = compare_case(cid, root, n_fd=n_fd, save_outputs=save,
                             plots_dir=plots_dir)
            all_results.append(r)

    # ---- summary table ----
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'case':>5} {'n_fd':>6} {'n_pts':>10} {'rel_L2':>12} {'max_err':>12} "
          f"{'rms_err':>12} {'max|U_pred|':>14} {'max|U_ref|':>12}")
    for r in all_results:
        print(f"{r['case']:>5d} {r['n_fd']:>6d} {r['n_pts']:>10d} "
              f"{r['rel_l2']:>12.4e} {r['max_err']:>12.4e} "
              f"{r['rms_err']:>12.4e} {r['max_abs_upinn']:>14.4e} "
              f"{r['max_abs_uref']:>12.4e}")


if __name__ == "__main__":
    main()
