r"""
Unified hard-boundary TFI PINN solver for the three L-shape exercises.

All cases use the transfinite trial form

    u = L[g] + N(lambda) - L[N](lambda),

with a lambda-space ResNet. Case 3 uses singularity splitting:

    u = u_s + u_r,  u_s = 5 * int_gamma Phi(x-y) ds_y,
    -Delta u_r = 0,  u_r = -u_s on dOmega.

The PINN loss is the strong residual. For case 2, the residual is evaluated on a
punctured domain by default because the exact corner function has second
derivatives of size O(r^(-4/3)) near the reentrant corner.

Outputs are saved under lshape_tfi/outputs_pinn_tfi/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import site
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUT_ROOT = SCRIPT_DIR / "outputs_pinn_tfi"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib-cache"))

if ".pixi/envs/" in sys.executable:
    user_site = site.getusersitepackages()
    if user_site in sys.path:
        sys.path.remove(user_site)


AREA = 3.0
SQRT2 = float(np.sqrt(2.0))
N_VERTS = 6
TORCH_DTYPE = torch.float32
VERTICES = torch.tensor(
    [
        [0.0, -1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ],
    dtype=TORCH_DTYPE,
)


def in_lshape(x, y):
    return (x > -1) & (x < 1) & (y > -1) & (y < 1) & ~((x >= 0) & (y <= 0))


def in_lshape_closed(x, y):
    return (x >= -1) & (x <= 1) & (y >= -1) & (y <= 1) & ~((x > 0) & (y < 0))


def lshape_basis_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concave Wachspress-like coordinates lambda_0,...,lambda_5 on the L-shape."""
    z = torch.sqrt(torch.clamp(x * x + y * y, min=0.0))
    c = z - x + y
    r = 1.0 - x
    t = 1.0 - y
    left = 1.0 + x
    b = 1.0 + y

    a0 = 0.5 * SQRT2 * c - y
    a2 = x + 0.5 * SQRT2 * c
    a3 = 0.5 * (1.0 + z + (SQRT2 - 1.0) * y)
    a4 = 0.5 * (SQRT2 + z)
    a5 = 0.5 * (1.0 + z - (SQRT2 - 1.0) * x)

    nums = torch.stack(
        [
            r * t * left * a0,
            r * t * left * b,
            t * left * b * a2,
            c * left * b * a3,
            c * r * b * a4,
            c * r * t * a5,
        ],
        dim=-1,
    )
    den = nums.sum(dim=-1, keepdim=True)
    return nums / den


@dataclass
class RunConfig:
    case: int
    integrator: str
    quad_order: int
    n_random: int
    arch: str
    network_input: str
    hidden: int
    depth: int
    adam_epochs: int
    lbfgs_steps: int
    lbfgs_max_iter: int
    lr: float
    lr_schedule: str
    min_lr: float
    step_size: int
    step_gamma: float
    lbfgs_lr: float
    eval_grid: int
    eval_every: int
    seed: int
    device: str
    dtype: str
    corner_radius: float
    run_name: str


@dataclass
class TrainHistoryRow:
    epoch: int
    loss: float
    residual_rms: float
    boundary_max: float
    rel_l2: float
    lr: float
    time_sec: float


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    weights: torch.Tensor | None


def select_device() -> torch.device:
    requested = os.environ.get(
        "TFI_DEVICE", os.environ.get("PINN_DEVICE", "auto")
    ).lower()
    if requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TFI_DEVICE=cuda was requested, but CUDA is not available."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError("TFI_DEVICE must be one of: auto, cuda, gpu, cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("--dtype must be one of: float32, float64")


def configure_dtype(name: str) -> None:
    global TORCH_DTYPE, VERTICES
    TORCH_DTYPE = select_dtype(name)
    torch.set_default_dtype(TORCH_DTYPE)
    VERTICES = VERTICES.to(dtype=TORCH_DTYPE)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def gauss_legendre_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def gauss_on_rect(x_range: tuple[float, float], y_range: tuple[float, float], n: int):
    pts, ws = gauss_legendre_1d(n)
    nx, ny = np.meshgrid(pts, pts, indexing="ij")
    wx, wy = np.meshgrid(ws, ws, indexing="ij")
    p0, p1, w = nx.ravel(), ny.ravel(), (wx * wy).ravel()
    x0, x1 = x_range
    y0, y1 = y_range
    x = (x1 - x0) * 0.5 * (p0 + 1.0) + x0
    y = (y1 - y0) * 0.5 * (p1 + 1.0) + y0
    weights = w * (x1 - x0) * 0.5 * (y1 - y0) * 0.5
    return x, y, weights


def sample_boundary_gauss(n_per_edge: int):
    pts, ws = gauss_legendre_1d(n_per_edge)
    tm = (pts + 1.0) / 2.0
    one = np.ones_like(tm)
    edges = [
        (tm * 2.0 - 1.0, one * 1.0, 2.0),
        (one * -1.0, tm * 2.0 - 1.0, 2.0),
        (-tm, one * -1.0, 1.0),
        (one * 1.0, tm, 1.0),
        (one * 0.0, -tm, 1.0),
        (tm, one * 0.0, 1.0),
    ]
    xs_all, ys_all, ws_all = [], [], []
    for xs, ys, length in edges:
        xs_all.append(xs)
        ys_all.append(ys)
        ws_all.append(ws * length / 2.0)
    return np.concatenate(xs_all), np.concatenate(ys_all), np.concatenate(ws_all)


def sample_interior_gauss_local(n: int, k: int = 1):
    """Gauss points on L-shape via (2k)×(2k) uniform square cells.

    The square [-1,1]² is divided into (2k)×(2k) cells of size (1/k)×(1/k).
    Cells in the bottom-right quadrant [0,1]×[-1,0] (k×k cells) are excluded.
    Each remaining cell gets n×n Gauss--Legendre points.
    Total cells: 3k², total points: 3k²·n².
    """
    h = 1.0 / k  # half cell width
    xs_all, ys_all, ws_all = [], [], []
    for i in range(2 * k):       # x-index: 0..2k-1, centre x = -1 + (i+0.5)*h
        for j in range(2 * k):   # y-index: 0..2k-1, centre y = -1 + (j+0.5)*h
            if i >= k and j < k:  # bottom-right quadrant → excluded
                continue
            x0 = -1.0 + i * h
            x1 = x0 + h
            y0 = -1.0 + j * h
            y1 = y0 + h
            xg, yg, wg = gauss_on_rect((x0, x1), (y0, y1), n)
            xs_all.append(xg)
            ys_all.append(yg)
            ws_all.append(wg)
    return np.concatenate(xs_all), np.concatenate(ys_all), np.concatenate(ws_all)


def fd_reference_case1(n: int = 201):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    xg = np.linspace(-1.0, 1.0, n)
    h = xg[1] - xg[0]
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    active = in_lshape(X, Y)
    idx = -np.ones((n, n), dtype=np.int64)
    idx[active] = np.arange(active.sum())
    total = int(active.sum())
    ii, jj = np.where(active)
    k = idx[ii, jj]

    rows = [k]
    cols = [k]
    vals = [np.full(total, 4.0 / h**2)]
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni = ii + di
        nj = jj + dj
        valid = (ni >= 0) & (ni < n) & (nj >= 0) & (nj < n)
        nact = np.zeros(len(ii), dtype=bool)
        nact[valid] = active[ni[valid], nj[valid]]
        rows.append(k[nact])
        cols.append(idx[ni[nact], nj[nact]])
        vals.append(np.full(int(nact.sum()), -1.0 / h**2))
    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total, total),
    ).tocsr()
    u = spla.spsolve(K, np.ones(total))
    U = np.full((n, n), np.nan)
    U[active] = u
    return xg, X, Y, active, U


def sample_interior_random_np(n: int) -> tuple[np.ndarray, np.ndarray]:
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    total = 0
    while total < n:
        m = max(512, 2 * (n - total))
        x = np.random.rand(m) * 2.0 - 1.0
        y = np.random.rand(m) * 2.0 - 1.0
        mask = in_lshape(x, y)
        xs_all.append(x[mask])
        ys_all.append(y[mask])
        total += int(mask.sum())
    return np.concatenate(xs_all)[:n], np.concatenate(ys_all)[:n]


def sample_interior_sobol_np(
    n: int, seed: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    n_left = int(round(n * 2.0 / 3.0))
    n_right = n - n_left
    engine_left = torch.quasirandom.SobolEngine(
        2, scramble=True, seed=seed + 17 * step + 1
    )
    engine_right = torch.quasirandom.SobolEngine(
        2, scramble=True, seed=seed + 17 * step + 2
    )
    left = engine_left.draw(n_left).cpu().numpy()
    right = engine_right.draw(n_right).cpu().numpy()
    xl = -1.0 + left[:, 0]
    yl = -1.0 + 2.0 * left[:, 1]
    xr = right[:, 0]
    yr = right[:, 1]
    return np.concatenate([xl, xr]), np.concatenate([yl, yr])


def make_batch(args: argparse.Namespace, device: torch.device, step: int) -> Batch:
    if args.integrator == "gauss":
        x_np, y_np, w_np = sample_interior_gauss_local(args.quad_order)
        weights = torch.tensor(w_np, dtype=TORCH_DTYPE, device=device)
    elif args.integrator == "monte-carlo":
        x_np, y_np = sample_interior_random_np(args.n_random)
        weights = None
    elif args.integrator == "sobol":
        x_np, y_np = sample_interior_sobol_np(args.n_random, args.seed, step)
        weights = None
    else:
        raise ValueError("integrator must be one of: gauss, monte-carlo, sobol")
    x = torch.tensor(x_np, dtype=TORCH_DTYPE, device=device)
    y = torch.tensor(y_np, dtype=TORCH_DTYPE, device=device)
    return Batch(x=x, y=y, weights=weights)


class LambdaResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x + self.block(x))


class LambdaMLP(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 4, input_dim: int = N_VERTS):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        return self.net(lam).squeeze(-1)


class LambdaResNet(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 4, input_dim: int = N_VERTS):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([LambdaResidualBlock(hidden) for _ in range(depth)])
        self.output = nn.Linear(hidden, 1)

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.input(lam))
        for block in self.blocks:
            h = block(h)
        return self.output(h).squeeze(-1)


def case2_boundary_raw(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r2 = x * x + y * y
    near_corner = r2 < 1.0e-12
    x_safe = torch.where(near_corner, torch.ones_like(x), x)
    y_safe = torch.where(near_corner, torch.zeros_like(y), y)
    theta = torch.atan2(y_safe, x_safe)
    theta = torch.where(theta < 0.0, theta + 2.0 * torch.pi, theta)
    r = torch.sqrt(torch.clamp(r2, min=1.0e-24))
    value = torch.pow(r, 2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)
    return torch.where(near_corner, torch.zeros_like(value), value)


def case2_exact_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0.0, theta + 2.0 * np.pi, theta)
    r = np.sqrt(x * x + y * y)
    return np.power(r, 2.0 / 3.0) * np.sin(2.0 * theta / 3.0)


GAMMA_SEGMENTS = [
    ((-0.8, -0.8), (-0.2, -0.8), 0.6),
    ((-0.8, -0.8), (-0.8, -0.2), 0.6),
    ((-0.8, -0.2), (-0.5, -0.2), 0.3),
    ((-0.2, -0.8), (-0.2, -0.5), 0.3),
    ((-0.5, -0.5), (-0.2, -0.5), 0.3),
    ((-0.5, -0.5), (-0.5, -0.2), 0.3),
]


def _F_np(u: np.ndarray, d: np.ndarray) -> np.ndarray:
    return u * np.log(u * u + d * d) - 2.0 * u + 2.0 * d * np.arctan2(u, d)


def _F_zero_np(u: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(u == 0.0, 0.0, 2.0 * u * (np.log(np.abs(u)) - 1.0))


def singular_part_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    G = np.zeros_like(x)
    for (ax, ay), (bx, by), length in GAMMA_SEGMENTS:
        tau_x = (bx - ax) / length
        tau_y = (by - ay) / length
        dx = x - ax
        dy = y - ay
        p = tau_x * dx + tau_y * dy
        cross = dx * tau_y - dy * tau_x
        d = np.abs(cross)
        u1 = -p
        u2 = length - p
        eps = np.finfo(np.float64).eps * length * 1.0e3
        mask = d > eps
        if np.any(mask):
            G[mask] += _F_np(u2[mask], d[mask]) - _F_np(u1[mask], d[mask])
        if np.any(~mask):
            G[~mask] += _F_zero_np(u2[~mask]) - _F_zero_np(u1[~mask])
    return -5.0 * G / (4.0 * np.pi)


def _F_torch(u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return u * torch.log(u * u + d * d) - 2.0 * u + 2.0 * d * torch.atan2(u, d)


def _F_zero_torch(u: torch.Tensor) -> torch.Tensor:
    abs_u = torch.abs(u)
    safe_abs = torch.clamp(abs_u, min=1.0e-30)
    value = 2.0 * u * (torch.log(safe_abs) - 1.0)
    return torch.where(abs_u == 0.0, torch.zeros_like(value), value)


def singular_part_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    G = torch.zeros_like(x)
    for (ax, ay), (bx, by), length in GAMMA_SEGMENTS:
        tau_x = (bx - ax) / length
        tau_y = (by - ay) / length
        dx = x - ax
        dy = y - ay
        p = tau_x * dx + tau_y * dy
        cross = dx * tau_y - dy * tau_x
        # Clamp keeps the logarithmic line-integral expression finite on Gamma.
        d = torch.sqrt(torch.clamp(cross * cross, min=1.0e-24))
        u1 = -p
        u2 = length - p
        f2 = _F_torch(u2, d)
        f1 = _F_torch(u1, d)
        f2_zero = _F_zero_torch(u2)
        f1_zero = _F_zero_torch(u1)
        near_line = torch.abs(cross) < 1.0e-12
        G = G + torch.where(near_line, f2_zero - f1_zero, f2 - f1)
    return -5.0 * G / (4.0 * torch.pi)


class TFIModel(nn.Module):
    def __init__(
        self,
        case_id: int,
        hidden: int,
        depth: int,
        arch: str,
        network_input: str,
        boundary_edge_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
        vertex_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.case_id = case_id
        self.network_input = network_input
        input_dim = 2 if network_input == "xy" else N_VERTS
        if arch == "mlp":
            self.base = LambdaMLP(hidden=hidden, depth=depth, input_dim=input_dim)
        elif arch == "resnet":
            self.base = LambdaResNet(hidden=hidden, depth=depth, input_dim=input_dim)
        else:
            raise ValueError("--arch must be one of: mlp, resnet")
        self.arch = arch
        self.boundary_edge_fn = boundary_edge_fn
        self.vertex_fn = vertex_fn

    def lift_network(self, lam: torch.Tensor) -> torch.Tensor:
        lifted = torch.zeros(lam.shape[0], dtype=lam.dtype, device=lam.device)
        for i in range(N_VERTS):
            ip1 = (i + 1) % N_VERTS
            im1 = (i - 1) % N_VERTS

            edge_next = torch.zeros_like(lam)
            edge_next[:, i] = 1.0 - lam[:, ip1]
            edge_next[:, ip1] = lam[:, ip1]

            edge_prev = torch.zeros_like(lam)
            edge_prev[:, i] = 1.0 - lam[:, im1]
            edge_prev[:, im1] = lam[:, im1]

            vertex = torch.zeros_like(lam)
            vertex[:, i] = 1.0

            lifted = lifted + lam[:, i] * (
                self.base(edge_next) + self.base(edge_prev) - self.base(vertex)
            )
        return lifted

    def lift_network_xy(self, lam: torch.Tensor) -> torch.Tensor:
        verts = VERTICES.to(dtype=lam.dtype, device=lam.device)
        lifted = torch.zeros(lam.shape[0], dtype=lam.dtype, device=lam.device)
        for i in range(N_VERTS):
            ip1 = (i + 1) % N_VERTS
            im1 = (i - 1) % N_VERTS

            t_next = lam[:, ip1]
            p_next = (1.0 - t_next[:, None]) * verts[i] + t_next[:, None] * verts[ip1]

            t_prev = 1.0 - lam[:, im1]
            p_prev = (1.0 - t_prev[:, None]) * verts[im1] + t_prev[:, None] * verts[i]

            p_vertex = verts[i].expand(lam.shape[0], 2)

            lifted = lifted + lam[:, i] * (
                self.base(p_next) + self.base(p_prev) - self.base(p_vertex)
            )
        return lifted

    def lift_boundary_data(self, lam: torch.Tensor) -> torch.Tensor:
        lifted = torch.zeros(lam.shape[0], dtype=lam.dtype, device=lam.device)
        verts = VERTICES.to(dtype=lam.dtype, device=lam.device)
        for i in range(N_VERTS):
            ip1 = (i + 1) % N_VERTS
            im1 = (i - 1) % N_VERTS
            g_next = self.boundary_edge_fn(i, lam[:, ip1], verts)
            g_prev = self.boundary_edge_fn(im1, 1.0 - lam[:, im1], verts)
            g_vertex = self.vertex_fn(verts[i, 0], verts[i, 1])
            lifted = lifted + lam[:, i] * (g_next + g_prev - g_vertex)
        return lifted

    def regular_forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lam = lshape_basis_torch(x, y)
        if self.network_input == "xy":
            xy = torch.stack((x, y), dim=-1)
            return (
                self.lift_boundary_data(lam) + self.base(xy) - self.lift_network_xy(lam)
            )
        return self.lift_boundary_data(lam) + self.base(lam) - self.lift_network(lam)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.case_id == 3:
            return singular_part_torch(x, y) + self.regular_forward(x, y)
        return self.regular_forward(x, y)


def zero_edge(edge: int, t: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


def zero_vertex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def case2_edge(edge: int, t: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
    if edge in (0, 1):
        return torch.zeros_like(t)
    ip1 = (edge + 1) % N_VERTS
    p = (1.0 - t[:, None]) * verts[edge] + t[:, None] * verts[ip1]
    return case2_boundary_raw(p[:, 0], p[:, 1])


def case3_edge(edge: int, t: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
    ip1 = (edge + 1) % N_VERTS
    p = (1.0 - t[:, None]) * verts[edge] + t[:, None] * verts[ip1]
    return -singular_part_torch(p[:, 0], p[:, 1])


def case3_vertex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -singular_part_torch(x, y)


def make_model(
    case_id: int, hidden: int, depth: int, arch: str, network_input: str
) -> TFIModel:
    if case_id == 1:
        return TFIModel(
            case_id, hidden, depth, arch, network_input, zero_edge, zero_vertex
        )
    if case_id == 2:
        return TFIModel(
            case_id, hidden, depth, arch, network_input, case2_edge, case2_boundary_raw
        )
    if case_id == 3:
        return TFIModel(
            case_id, hidden, depth, arch, network_input, case3_edge, case3_vertex
        )
    raise ValueError("case must be one of 1, 2, 3")


def prepare_residual_batch(case_id: int, batch: Batch, corner_radius: float) -> Batch:
    if case_id not in (2, 3) or corner_radius <= 0.0:
        return batch
    keep = batch.x * batch.x + batch.y * batch.y > corner_radius * corner_radius
    if not bool(torch.any(keep).detach().cpu()):
        raise RuntimeError("No residual points remain after applying --corner-radius.")
    weights = batch.weights[keep] if batch.weights is not None else None
    return Batch(x=batch.x[keep], y=batch.y[keep], weights=weights)


def laplacian_of(u: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ux, uy = torch.autograd.grad(u.sum(), (x, y), create_graph=True)
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True, retain_graph=True)[0]
    uyy = torch.autograd.grad(uy.sum(), y, create_graph=True)[0]
    return uxx + uyy


def precompute_boundary_lifting(
    x: torch.Tensor,
    y: torch.Tensor,
    model: TFIModel,
    case_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del case_id
    with torch.no_grad():
        lam = lshape_basis_torch(x, y)
        g_vals = model.lift_boundary_data(lam).detach()

    x_graph = x.detach().clone().requires_grad_(True)
    y_graph = y.detach().clone().requires_grad_(True)
    lam_graph = lshape_basis_torch(x_graph, y_graph)
    g_vals_on_graph = model.lift_boundary_data(lam_graph)
    lap_g = laplacian_of(g_vals_on_graph, x_graph, y_graph).detach()
    return g_vals, lap_g


def pinn_loss(
    case_id: int,
    model: TFIModel,
    batch: Batch,
    corner_radius: float,
    lap_g_precomputed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = prepare_residual_batch(case_id, batch, corner_radius)
    x = batch.x.detach().clone().requires_grad_(True)
    y = batch.y.detach().clone().requires_grad_(True)
    if lap_g_precomputed is None:
        u = model.regular_forward(x, y) if case_id == 3 else model(x, y)
        lap = laplacian_of(u, x, y)
    else:
        lam = lshape_basis_torch(x, y)
        if model.network_input == "xy":
            xy = torch.stack((x, y), dim=-1)
            u_net = model.base(xy) - model.lift_network_xy(lam)
        else:
            u_net = model.base(lam) - model.lift_network(lam)
        lap = laplacian_of(u_net, x, y) + lap_g_precomputed
    if case_id == 1:
        res = lap + 1.0
    else:
        res = lap
    density = res * res
    if batch.weights is None:
        loss = torch.mean(density)
    else:
        loss = torch.sum(batch.weights * density) / AREA
    rms = torch.sqrt(torch.mean(density))
    return loss, rms


def boundary_error(
    case_id: int, model: TFIModel, device: torch.device, n_per_edge: int = 64
) -> float:
    xb_np, yb_np, _ = sample_boundary_gauss(n_per_edge)
    verts = VERTICES.cpu().numpy()
    xb_np = np.concatenate([xb_np, verts[:, 0]])
    yb_np = np.concatenate([yb_np, verts[:, 1]])
    xb = torch.tensor(xb_np, dtype=TORCH_DTYPE, device=device)
    yb = torch.tensor(yb_np, dtype=TORCH_DTYPE, device=device)
    with torch.no_grad():
        pred = model(xb, yb)
        if case_id == 1:
            target = torch.zeros_like(pred)
        elif case_id == 2:
            target = case2_boundary_raw(xb, yb)
        else:
            target = torch.zeros_like(pred)
        err = pred - target
    return float(torch.max(torch.abs(err)).cpu())


def basis_error(device: torch.device, n_per_edge: int = 64) -> float:
    xb_np, yb_np, _ = sample_boundary_gauss(n_per_edge)
    xb = torch.tensor(xb_np, dtype=TORCH_DTYPE, device=device)
    yb = torch.tensor(yb_np, dtype=TORCH_DTYPE, device=device)
    lam = lshape_basis_torch(xb, yb)
    finite = torch.isfinite(lam).all()
    if not bool(finite.cpu()):
        return float("inf")
    return float(torch.max(torch.abs(lam.sum(dim=1) - 1.0)).cpu())


def make_eval_grid(case_id: int, n_grid: int):
    if case_id == 1:
        xg, X, Y, interior_active, U_ref = fd_reference_case1(n_grid)
        active = in_lshape_closed(X, Y)
        boundary = active & ~interior_active
        U_ref[boundary] = 0.0
        return xg, X, Y, active, interior_active, U_ref
    if case_id == 2:
        xg = np.linspace(-1.0, 1.0, n_grid)
        X, Y = np.meshgrid(xg, xg, indexing="ij")
        active = in_lshape_closed(X, Y)
        interior_active = in_lshape(X, Y)
        U_ref = np.full(X.shape, np.nan)
        U_ref[active] = case2_exact_np(X[active], Y[active])
        return xg, X, Y, active, interior_active, U_ref
    candidates = [
        SCRIPT_DIR / "reference_case3_1601.npz",
        SCRIPT_DIR / "reference_case3.npz",
        ROOT_DIR / "lshape-poisson" / "reference_case3_1601.npz",
        ROOT_DIR / "lshape-poisson" / "reference_case3.npz",
    ]
    for path in candidates:
        if path.exists():
            ref = np.load(path, allow_pickle=True)
            xg_full = ref["xg"]
            U_full = ref["U_ref"]
            if len(xg_full) == n_grid:
                xg = xg_full
                U_ref = U_full
            else:
                idx = np.linspace(0, len(xg_full) - 1, n_grid).round().astype(int)
                xg = xg_full[idx]
                U_ref = U_full[np.ix_(idx, idx)]
            X, Y = np.meshgrid(xg, xg, indexing="ij")
            active = in_lshape_closed(X, Y)
            interior_active = in_lshape(X, Y)
            return xg, X, Y, active, interior_active, U_ref
    xg = np.linspace(-1.0, 1.0, n_grid)
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    active = in_lshape_closed(X, Y)
    interior_active = in_lshape(X, Y)
    return xg, X, Y, active, interior_active, np.full(X.shape, np.nan)


def evaluate(
    case_id: int,
    model: TFIModel,
    device: torch.device,
    n_grid: int,
    corner_radius: float = 0.0,
):
    xg, X, Y, active, interior_active, U_ref = make_eval_grid(case_id, n_grid)
    xa = X[active]
    ya = Y[active]
    xt = torch.tensor(xa, dtype=TORCH_DTYPE, device=device)
    yt = torch.tensor(ya, dtype=TORCH_DTYPE, device=device)
    preds: list[np.ndarray] = []
    regs: list[np.ndarray] = []
    step = 4096
    for start in range(0, len(xa), step):
        end = min(start + step, len(xa))
        with torch.no_grad():
            preds.append(model(xt[start:end], yt[start:end]).cpu().numpy())
            regs.append(
                model.regular_forward(xt[start:end], yt[start:end]).cpu().numpy()
            )
    up = np.concatenate(preds) if preds else np.array([])
    ur = np.concatenate(regs) if regs else np.array([])
    U_pred = np.full(X.shape, np.nan)
    U_pred[active] = up
    U_regular = np.full(X.shape, np.nan)
    U_regular[active] = ur
    U_s = np.full(X.shape, np.nan)
    if case_id == 3:
        U_s[active] = singular_part_np(xa, ya)
    finite = interior_active & np.isfinite(U_ref)
    if case_id in (2, 3) and corner_radius > 0.0:
        finite &= X * X + Y * Y > corner_radius * corner_radius
    U_err = np.full(X.shape, np.nan)
    if np.any(finite):
        diff = U_pred[finite] - U_ref[finite]
        U_err[finite] = np.abs(diff)
        rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(U_ref[finite]))
        max_err = float(np.max(np.abs(diff)))
    else:
        rel_l2 = float("nan")
        max_err = float("nan")
    return {
        "xg": xg,
        "X": X,
        "Y": Y,
        "active": active,
        "interior_active": interior_active,
        "U_pred": U_pred,
        "U_ref": U_ref,
        "U_err": U_err,
        "U_s": U_s,
        "U_regular": U_regular,
        "rel_l2": rel_l2,
        "max_err": max_err,
    }


def evaluate_residual_field(
    case_id: int,
    model: TFIModel,
    device: torch.device,
    eval_data: dict,
    corner_radius: float,
):
    X = eval_data["X"]
    Y = eval_data["Y"]
    active = eval_data.get("interior_active", eval_data["active"]).copy()
    if case_id in (2, 3) and corner_radius > 0.0:
        active &= X * X + Y * Y > corner_radius * corner_radius
    xa = X[active]
    ya = Y[active]
    residual_vals: list[np.ndarray] = []
    grad_norm_vals: list[np.ndarray] = []
    step = 2048
    for start in range(0, len(xa), step):
        end = min(start + step, len(xa))
        x = torch.tensor(
            xa[start:end], dtype=TORCH_DTYPE, device=device, requires_grad=True
        )
        y = torch.tensor(
            ya[start:end], dtype=TORCH_DTYPE, device=device, requires_grad=True
        )
        u = model.regular_forward(x, y) if case_id == 3 else model(x, y)
        ux, uy = torch.autograd.grad(u.sum(), (x, y), create_graph=True)
        uxx = torch.autograd.grad(ux.sum(), x, create_graph=True, retain_graph=True)[0]
        uyy = torch.autograd.grad(uy.sum(), y, create_graph=True)[0]
        lap = uxx + uyy
        residual = -lap
        if case_id == 1:
            residual = residual - 1.0
        residual_vals.append(residual.detach().cpu().numpy())
        grad_norm_vals.append(
            torch.sqrt(torch.clamp(ux * ux + uy * uy, min=0.0)).detach().cpu().numpy()
        )
    U_residual = np.full(X.shape, np.nan)
    U_grad_norm = np.full(X.shape, np.nan)
    if residual_vals:
        U_residual[active] = np.concatenate(residual_vals)
        U_grad_norm[active] = np.concatenate(grad_norm_vals)
    finite = np.isfinite(U_residual)
    if np.any(finite):
        rms = float(np.sqrt(np.nanmean(U_residual[finite] ** 2)))
        max_abs = float(np.nanmax(np.abs(U_residual[finite])))
    else:
        rms = float("nan")
        max_abs = float("nan")
    return U_residual, U_grad_norm, rms, max_abs


def write_history(path: Path, rows: list[TrainHistoryRow]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(TrainHistoryRow.__annotations__.keys())
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_preview(
    path: Path,
    case_id: int,
    eval_data: dict,
    U_residual: np.ndarray,
    history: list[TrainHistoryRow],
) -> None:
    xg = eval_data["xg"]
    U_pred = eval_data["U_pred"]
    U_ref = eval_data["U_ref"]
    U_err = eval_data["U_err"]
    vmin = (
        np.nanmin(U_pred)
        if not np.isfinite(U_ref).any()
        else min(np.nanmin(U_pred), np.nanmin(U_ref))
    )
    vmax = (
        np.nanmax(U_pred)
        if not np.isfinite(U_ref).any()
        else max(np.nanmax(U_pred), np.nanmax(U_ref))
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    residual_title = "|residual of u_r|" if case_id == 3 else "|strong residual|"
    for ax, U, title in [
        (axes[0, 0], U_pred, "TFI PINN"),
        (axes[0, 1], U_ref, "Reference"),
        (axes[1, 0], U_err, "|error|"),
        (axes[1, 1], np.abs(U_residual), residual_title),
    ]:
        pcm = ax.pcolormesh(
            xg,
            xg,
            U.T,
            cmap="jet",
            shading="auto",
            vmin=vmin if title in ("TFI PINN", "Reference") else None,
            vmax=vmax if title in ("TFI PINN", "Reference") else None,
        )
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.plot(0, 0, "r*", ms=10, mec="k")
        if case_id == 3:
            for sq in [([-0.8, -0.2], [-0.8, -0.2]), ([-0.5, -0.2], [-0.5, -0.2])]:
                x0, x1 = sq[0]
                y0, y1 = sq[1]
                ax.plot(
                    [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "w-", lw=0.6, alpha=0.6
                )
        plt.colorbar(pcm, ax=ax, shrink=0.8)
    if history:
        inset = axes[1, 1].inset_axes([0.05, 0.05, 0.45, 0.32])
        inset.semilogy(
            [r.epoch for r in history], [max(r.loss, 1.0e-300) for r in history]
        )
        inset.set_title("loss", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.grid(True)
    fig.suptitle(
        f"Case {case_id} TFI PINN | rel L2={eval_data['rel_l2']:.3e}, max={eval_data['max_err']:.3e}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run_tests(case_id: int, args: argparse.Namespace, device: torch.device) -> None:
    model = make_model(
        case_id, hidden=16, depth=2, arch=args.arch, network_input=args.network_input
    ).to(device)
    bdy = boundary_error(case_id, model, device)
    berr = basis_error(device)
    if bdy > 1.0e-5:
        raise AssertionError(f"case {case_id} boundary error too large: {bdy:.3e}")
    if berr > 1.0e-5:
        raise AssertionError(f"basis partition error too large: {berr:.3e}")
    print(f"case {case_id} tests passed: boundary={bdy:.3e}, basis={berr:.3e}")


def train_case(
    case_id: int, args: argparse.Namespace, device: torch.device
) -> tuple[TFIModel, list[TrainHistoryRow], RunConfig]:
    run_name = make_run_name(case_id, args)
    cfg = RunConfig(
        case=case_id,
        integrator=args.integrator,
        quad_order=args.quad_order,
        n_random=args.n_random,
        arch=args.arch,
        network_input=args.network_input,
        hidden=args.hidden,
        depth=args.depth,
        adam_epochs=args.adam_epochs,
        lbfgs_steps=args.lbfgs_steps,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        min_lr=args.min_lr,
        step_size=args.step_size,
        step_gamma=args.step_gamma,
        lbfgs_lr=args.lbfgs_lr,
        eval_grid=args.eval_grid,
        eval_every=args.eval_every,
        seed=args.seed,
        device=str(device),
        dtype=args.dtype,
        corner_radius=args.corner_radius if case_id in (2, 3) else 0.0,
        run_name=run_name,
    )
    model = make_model(
        case_id, args.hidden, args.depth, args.arch, args.network_input
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(args.adam_epochs, 1), eta_min=args.min_lr
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(args.step_size, 1), gamma=args.step_gamma
        )
    elif args.lr_schedule == "none":
        scheduler = None
    else:
        raise ValueError("--lr-schedule must be one of: cosine, step, none")
    history: list[TrainHistoryRow] = []
    fixed_batch = make_batch(args, device, 0) if args.integrator == "gauss" else None
    lap_g_precomputed: torch.Tensor | None = None
    if fixed_batch is not None:
        fixed_batch = prepare_residual_batch(case_id, fixed_batch, cfg.corner_radius)
        _, lap_g_precomputed = precompute_boundary_lifting(
            fixed_batch.x, fixed_batch.y, model, case_id
        )
    n_pts = fixed_batch.x.numel() if fixed_batch is not None else args.n_random
    if case_id in (2, 3) and args.corner_radius > 0.0:
        print(
            f"case {case_id} residual excludes r <= {args.corner_radius:g}", flush=True
        )
    print(
        f"Training case {case_id} PINN [{args.integrator}, {n_pts} pts], "
        f"{args.arch}/{args.network_input} h={args.hidden} d={args.depth}, Adam {args.adam_epochs} + "
        f"L-BFGS {args.lbfgs_steps}x{args.lbfgs_max_iter} on {device}\n"
        f"  lr={args.lr:g}, schedule={args.lr_schedule}, min_lr={args.min_lr:g}, "
        f"lbfgs_lr={args.lbfgs_lr:g}",
        flush=True,
    )
    t0 = time.time()
    for ep in range(args.adam_epochs):
        batch = fixed_batch if fixed_batch is not None else make_batch(args, device, ep)
        loss, residual_rms = pinn_loss(
            case_id, model, batch, cfg.corner_radius, lap_g_precomputed
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        print_due = ep % args.print_every == 0 or ep == args.adam_epochs - 1
        eval_due = args.eval_every > 0 and ep > 0 and ep % args.eval_every == 0
        if print_due or eval_due:
            bdy = boundary_error(case_id, model, device, n_per_edge=16)
            rel_l2 = float("nan")
            if eval_due:
                eval_data = evaluate(
                    case_id, model, device, args.eval_grid, cfg.corner_radius
                )
                rel_l2 = eval_data["rel_l2"]
            row = TrainHistoryRow(
                epoch=ep,
                loss=float(loss.detach().cpu()),
                residual_rms=float(residual_rms.detach().cpu()),
                boundary_max=bdy,
                rel_l2=rel_l2,
                lr=float(opt.param_groups[0]["lr"]),
                time_sec=time.time() - t0,
            )
            history.append(row)
            if print_due:
                rel_part = f" rel_l2={rel_l2:.4e}" if eval_due else ""
                print(
                    f"  adam {ep:5d}: loss={row.loss:.4e} rms={row.residual_rms:.4e} "
                    f"bdy={bdy:.2e}{rel_part} lr={row.lr:.1e}",
                    flush=True,
                )

    if args.lbfgs_steps > 0:
        lbfgs_batch = (
            fixed_batch
            if fixed_batch is not None
            else make_batch(args, device, args.adam_epochs + 1)
        )
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_max_iter,
            history_size=50,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt_lbfgs.zero_grad()
            loss_inner, _ = pinn_loss(
                case_id, model, lbfgs_batch, cfg.corner_radius, lap_g_precomputed
            )
            loss_inner.backward()
            return loss_inner

        print("L-BFGS refinement...", flush=True)
        for step in range(args.lbfgs_steps):
            loss_val = opt_lbfgs.step(closure)
            loss_float = (
                float(loss_val.detach().cpu()) if loss_val is not None else float("nan")
            )
            if step % 5 == 0 or step == args.lbfgs_steps - 1:
                with torch.no_grad():
                    bdy = boundary_error(case_id, model, device, n_per_edge=16)
                row = TrainHistoryRow(
                    epoch=args.adam_epochs + step,
                    loss=loss_float,
                    residual_rms=math.sqrt(max(loss_float, 0.0)),
                    boundary_max=bdy,
                    rel_l2=float("nan"),
                    lr=args.lbfgs_lr,
                    time_sec=time.time() - t0,
                )
                history.append(row)
                print(
                    f"  lbfgs {step:4d}: loss={loss_float:.4e} bdy={bdy:.2e}",
                    flush=True,
                )
    return model, history, cfg


def save_outputs(
    case_id: int,
    model: TFIModel,
    history: list[TrainHistoryRow],
    cfg: RunConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    out_dir = OUT_ROOT / f"case{case_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / cfg.run_name
    eval_data = evaluate(case_id, model, device, args.eval_grid, cfg.corner_radius)
    U_residual, U_grad_norm, residual_rms, residual_max = evaluate_residual_field(
        case_id, model, device, eval_data, cfg.corner_radius
    )
    final_bdy = boundary_error(case_id, model, device)
    if history:
        history[-1].boundary_max = final_bdy
        history[-1].rel_l2 = eval_data["rel_l2"]
        history[-1].residual_rms = residual_rms
    config_dict = asdict(cfg)
    config_dict["final_boundary_max"] = final_bdy
    config_dict["final_rel_l2"] = eval_data["rel_l2"]
    config_dict["final_max_err"] = eval_data["max_err"]
    config_dict["final_residual_rms"] = residual_rms
    config_dict["final_residual_max"] = residual_max
    config_dict["gamma_segments"] = GAMMA_SEGMENTS if case_id == 3 else None
    np.savez_compressed(
        base.with_suffix(".npz"),
        xg=eval_data["xg"],
        X=eval_data["X"],
        Y=eval_data["Y"],
        active=eval_data["active"],
        interior_active=eval_data["interior_active"],
        U_pred=eval_data["U_pred"],
        U_ref=eval_data["U_ref"],
        U_err=eval_data["U_err"],
        U_s=eval_data["U_s"],
        U_regular=eval_data["U_regular"],
        U_residual=U_residual,
        U_grad_norm=U_grad_norm,
        loss_hist=np.array([r.loss for r in history]),
        residual_rms_hist=np.array([r.residual_rms for r in history]),
        boundary_error_hist=np.array([r.boundary_max for r in history]),
        history_epoch=np.array([r.epoch for r in history]),
        rel_l2=eval_data["rel_l2"],
        max_err=eval_data["max_err"],
        residual_rms=residual_rms,
        residual_max=residual_max,
        boundary_max_abs=final_bdy,
        config_json=json.dumps(config_dict),
    )
    write_history(base.with_name(base.name + "_history.csv"), history)
    with base.with_name(base.name + "_config.json").open("w") as f:
        json.dump(config_dict, f, indent=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config_dict,
            "final_metrics": config_dict,
        },
        base.with_suffix(".pt"),
    )
    if not args.no_plot:
        save_preview(
            base.with_name(base.name + "_preview.png"),
            case_id,
            eval_data,
            U_residual,
            history,
        )
    print(
        f"case {case_id}: rel_L2={eval_data['rel_l2']:.4e}, max_err={eval_data['max_err']:.4e}, "
        f"res_rms={residual_rms:.4e}, boundary={final_bdy:.3e}",
        flush=True,
    )
    print(f"Saved outputs under {out_dir}", flush=True)


def make_run_name(case_id: int, args: argparse.Namespace) -> str:
    if args.integrator == "gauss":
        integ = f"gauss_q{args.quad_order}"
    else:
        integ = f"{args.integrator}_n{args.n_random}"
    corner = f"{args.corner_radius:g}".replace(".", "p").replace("-", "m")
    suffix = f"_cr{corner}" if case_id in (2, 3) and args.corner_radius > 0.0 else ""
    input_suffix = "" if args.network_input == "lambda" else f"_{args.network_input}"
    dtype_suffix = "" if args.dtype == "float32" else f"_{args.dtype}"
    return f"case{case_id}_{integ}_{args.arch}{input_suffix}_h{args.hidden}_d{args.depth}_seed{args.seed}{suffix}{dtype_suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified hard-boundary TFI PINN solver for L-shape cases 1-3."
    )
    parser.add_argument("--case", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument(
        "--integrator", choices=("gauss", "monte-carlo", "sobol"), default="gauss"
    )
    parser.add_argument("--quad-order", type=int, default=16)
    parser.add_argument("--n-random", type=int, default=4096)
    parser.add_argument("--arch", choices=("mlp", "resnet"), default="mlp")
    parser.add_argument("--network-input", choices=("lambda", "xy"), default="lambda")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--adam-epochs", "--epochs", dest="adam_epochs", type=int, default=5000
    )
    parser.add_argument("--lbfgs-steps", type=int, default=40)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--lr-schedule", choices=("cosine", "step", "none"), default="none"
    )
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--step-gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--corner-radius", type=float, default=0.0)
    parser.add_argument("--eval-grid", type=int, default=151)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_dtype(args.dtype)
    set_seed(args.seed)
    device = select_device()
    cases = [1, 2, 3] if args.case == "all" else [int(args.case)]
    if args.test or args.test_only:
        for case_id in cases:
            run_tests(case_id, args, device)
    if args.test_only:
        return
    for case_id in cases:
        model, history, cfg = train_case(case_id, args, device)
        save_outputs(case_id, model, history, cfg, args, device)


if __name__ == "__main__":
    main()
