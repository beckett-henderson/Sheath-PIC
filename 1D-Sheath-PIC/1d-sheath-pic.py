#!/usr/bin/env python3
# plasma_sheath_pic_dirichlet.py

"""
1D Plasma Sheath (PIC) with Dirichlet BCs, Leapfrog, and Sheath Accumulation
Author: Beckett Henderson
Date Created: 08/12/25
Date Last Modified: 09/16/25

Overview
--------
This script simulates a 1D plasma sheath with immobile ions and mobile electrons
on a uniform grid using a Particle‑in‑Cell (PIC) algorithm. Compared to the
original reference implementation, this **new functioning code** adopts these
key modeling choices (aligned with a treecode comparison workflow):

  • **Force model:** Electrons feel **electron–electron + sheath** fields only.
    Ions are *excluded* from the force (but can still be initialized for future
    comparisons). Use `--include_ions_in_force` to optionally include ions.

  • **Field convention:** **E = +∂φ/∂x** (matches the sign convention used by
    the treecode). This is different from the more common E = −∂φ/∂x.

  • **Boundary conditions:** Dirichlet walls with φ(0)=φ₀ and φ(L)=φ_b.
    Absorbed electron charge at each wall is re‑introduced into the field via
    either (a) **Dirichlet Green’s function** singular sources at x≈0 and x≈L,
    or (b) an optional **near‑wall smear layer** (`--sheath_mode smear`) with
    configurable `--smear_cells` depth.

  • **Time integrator:** Staggered **leapfrog (Drift–Kick)** by default with an
    **adaptive CFL time step** based on current max |v|. Optional
    **explicit Euler** (`--euler`) is provided for parity with the README and
    quick experiments.

  • **I/O & viz:** Optional MP4 phase‑space video (x,v). Optionally save PNG
    **snapshots** of φ(x) at specified times via `--snap_t`.

Non‑dimensionalization (same as reference, primes omitted in code):
  Reference scales: L_ref = λ_D, v_th, T_ref = L_ref/v_th, φ_0 = m_e v_th²/q_0.
  Governing equations in code units:
    ∂²φ/∂x² = −ρ,   E = +∂φ/∂x (this code),
    dx/dt = v,
    dv/dt = (q/m) E.

Leapfrog update (with adaptive Δt):
    v^{n+1/2} = v^{n−1/2} + (q/m) E^n Δt,
    x^{n+1}   = x^n       + v^{n+1/2} Δt.

Euler update (optional):
    v^{n+1} = v^n + (q/m) E^n Δt,
    x^{n+1} = x^n + v^{n+1} Δt.

CLI flags let you configure domain, resolution, runtime, randomness, I/O, and
whether to include ions in the force. Defaults reproduce the new code behavior.

Notes on parity with the README you provided:
  • Adds **`--euler`** integrator, **`--snap_t`** snapshots, **`--smear_cells`**
    with **`--sheath_mode {greens,smear}`**, optional fixed‑step **`--dt`** and
    **`--steps`** (legacy mode) and video cadence via **`--save_every`** (in
    steps) as an alternative to `--frame_dt`.

"""

# =============================================================================
# Imports & Logging
# =============================================================================
# pip install numpy matplotlib imageio tqdm
import argparse
import logging
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
        * parse_args
        * Parses command line arguments for simulation parameters and I/O.
        * Equations context:
            * Kinematics (dimensionless): dx/dt = v
            * Dynamics   (dimensionless): dv/dt = (q/m) E,  with E = +∂φ/∂x
        * Date Created: 08/12/25
        * Date Last Modified: 09/16/25
        * Author: Beckett Henderson
        * API:
            * Inputs: None
            * Outputs: args (argparse.Namespace)

    Default Initial Conditions (PIC units):
        * L=1.0; N_ion=N_electron=10000; N_grid=256
        * v_th=1.0; seed=0; φ(0)=φ(L)=0
        * m_e=1, q_e=−0.1, q_i=+0.1
        * Leapfrog (default), adaptive CFL with dt_max=0.01, CFL=0.1, eps_v=1e-12
        * T_final=5.0 (sim units)
        * Video on, fps=10, frame_dt=0.05 → electron_evolution.mp4
        * Absorption plot → loss_rate.png
    """
    p = argparse.ArgumentParser(description="1D Plasma Sheath PIC (Dirichlet, leapfrog/Euler, sheath accumulation)")

    # Domain & grid
    p.add_argument('--L', type=float, default=1.0, help='Domain length')
    p.add_argument('--N_ion', type=int, default=10_000, help='Number of immobile ions')
    p.add_argument('--N_electron', type=int, default=10_000, help='Number of electrons (mobile)')
    p.add_argument('--N_grid', type=int, default=256, help='Number of grid cells (nodes = N_grid+1)')

    # Species & ICs
    p.add_argument('--m_e', type=float, default=1.0, help='Electron mass (code units)')
    p.add_argument('--q_e', type=float, default=-1.0/10.0, help='Electron charge (code units)')
    p.add_argument('--q_i', type=float, default=+1.0/10.0, help='Ion charge (code units)')
    p.add_argument('--v_th', type=float, default=1.0, help='Electron thermal velocity for initialization')
    p.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    # Boundaries & sheath handling
    p.add_argument('--phi0', type=float, default=0.0, help='Left Dirichlet boundary φ(0)')
    p.add_argument('--phib', type=float, default=0.0, help='Right Dirichlet boundary φ(L)')
    p.add_argument('--sheath_mode', choices=['greens','smear'], default='greens', help='Sheath modeling mode: Dirichlet Green\'s sources or near‑wall smear layer')
    p.add_argument('--smear_cells', type=int, default=1, help='Smear depth (cells) when --sheath_mode=smear')

    # Time stepping / runtime
    p.add_argument('--T_final', type=float, default=5.0, help='Total simulation time (sim units) for adaptive CFL mode')
    p.add_argument('--CFL', type=float, default=0.1, help='CFL number for adaptive dt: dt ≤ CFL * dx / max(|v|)')
    p.add_argument('--dt_max', type=float, default=0.01, help='Maximum allowed dt (safeguard)')
    p.add_argument('--eps_v', type=float, default=1e-12, help='Velocity floor used in CFL calculation')

    # Legacy fixed‑step option (optional; matches README table)
    p.add_argument('--dt', type=float, default=None, help='If set (>0), use fixed dt and --steps instead of adaptive CFL')
    p.add_argument('--steps', type=int, default=None, help='If set (>0), number of steps for fixed‑step mode')

    # Integrator selection
    p.add_argument('--euler', action='store_true', help='Use explicit Euler instead of leapfrog')

    # Force model
    p.add_argument('--include_ions_in_force', action='store_true', help='If set, include ion charge in field seen by electrons')

    # Output / video / snapshots
    p.add_argument('--make_video', action='store_true', default=True, help='Write phase‑space MP4 video')
    p.add_argument('--fps', type=int, default=10, help='Frames per second for video')
    p.add_argument('--frame_dt', type=float, default=0.05, help='Time spacing between saved frames (sim units; ignored if --save_every is used)')
    p.add_argument('--save_every', type=int, default=None, help='If set, save a frame every N steps (overrides --frame_dt cadence)')
    p.add_argument('--output', type=str, default='electron_evolution.mp4', help='MP4 filename')
    p.add_argument('--snap_t', type=float, nargs='*', default=None, help='Times (sim units) to save PNG snapshots of φ(x)')
    p.add_argument('--ylim_v', type=float, nargs=2, default=[-10.0, 10.0], help='y‑limits for velocity plot')
    p.add_argument('--ylim_phi', type=float, nargs=2, default=[0.0, 1.2e4], help='y‑limits for potential plot')
    p.add_argument('--loss_png', type=str, default='loss_rate.png', help='PNG filename for absorption plot')

    args = p.parse_args()

    if args.N_grid < 2:
        p.error('--N_grid must be ≥ 2')
    if args.N_electron <= 0:
        p.error('--N_electron must be positive')
    if args.L <= 0:
        p.error('--L must be positive')
    if args.dt is not None and args.dt <= 0:
        p.error('--dt must be positive when provided')
    if (args.dt is None) ^ (args.steps is None):
        p.error('Provide both --dt and --steps for fixed‑step mode, or neither for adaptive CFL mode')
    if args.dt is not None and args.steps is not None and args.steps <= 0:
        p.error('--steps must be positive when using fixed‑step mode')

    return args

# =============================================================================
# Numerical Kernels (Thomas, CIC, Gather, Greens, Smear)
# =============================================================================

def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
        * thomas_solve
        * Solves a tridiagonal linear system A x = d using the Thomas algorithm.
        * Equations:
            * A is tridiagonal with sub‑diagonal a, main‑diagonal b, super‑diagonal c
            * Forward elimination, then back substitution
        * Date Created: 08/12/25
        * Date Last Modified: 09/16/25
        * Author: Beckett Henderson
        * API:
            * Inputs: a (n−1), b (n), c (n−1), d (n)
            * Outputs: x (n)

    Numerical Notes:
        * O(n) time and memory; stable for the Poisson FD system used here.
    """
    n = len(b)
    cc = c.astype(float).copy()
    dd = d.astype(float).copy()
    bb = b.astype(float).copy()
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    for i in range(1, n):
        denom = bb[i] - a[i-1]*cc[i-1]
        if i < n-1:
            cc[i] /= denom
        dd[i] = (dd[i] - a[i-1]*dd[i-1]) / denom
    x = np.empty(n)
    x[-1] = dd[-1]
    for i in range(n-2, -1, -1):
        x[i] = dd[i] - cc[i]*x[i+1]
    return x


def solve_phi_from_rho(rho_nodes: np.ndarray, dx: float, phi0: float, phib: float) -> np.ndarray:
    """
        * solve_phi_from_rho
        * Solves the Dirichlet Poisson problem −φ'' = ρ on [0, L] using node‑based
          finite differences and a Thomas tridiagonal solve on interior nodes.
        * Discrete interior: (−φ_{i−1} + 2φ_i − φ_{i+1})/dx² = ρ_i, i=1..N−1
        * BCs: φ_0 = φ(0) = phi0, φ_N = φ(L) = phib
        * Date Created: 08/12/25
        * Date Last Modified: 09/16/25
        * Author: Beckett Henderson
        * API:
            * Inputs: rho_nodes (N+1), dx, phi0, phib
            * Outputs: phi (N+1)

    Numerical Notes:
        * RHS includes boundary contributions: rhs[0]+=phi0/dx², rhs[-1]+=phib/dx².
    """
    N = rho_nodes.size - 1
    n = N - 1
    invdx2 = 1.0/(dx*dx)
    a = -np.ones(n-1)*invdx2
    b =  2.0*np.ones(n)*invdx2
    c = -np.ones(n-1)*invdx2
    rhs = rho_nodes[1:N].copy()
    rhs[0]  += phi0*invdx2
    rhs[-1] += phib*invdx2
    phi_inner = thomas_solve(a, b, c, rhs)
    phi = np.empty(N+1)
    phi[0] = phi0
    phi[1:N] = phi_inner
    phi[N] = phib
    return phi


def compute_grad(phi: np.ndarray, dx: float) -> np.ndarray:
    """
        * compute_grad
        * Computes ∂φ/∂x on nodes using centered differences in the interior and
          one‑sided at the boundaries.
        * E convention in this code: **E = +∂φ/∂x**.
        * API: Inputs: phi (N+1), dx → Outputs: grad (N+1)
    """
    G = np.empty_like(phi)
    G[1:-1] = (phi[2:] - phi[:-2])/(2*dx)
    G[0]    = (phi[1]  - phi[0]) / dx
    G[-1]   = (phi[-1] - phi[-2]) / dx
    return G


def cic_deposit_nodes(positions: np.ndarray, charges: np.ndarray, dx: float, N_grid: int) -> np.ndarray:
    """
        * cic_deposit_nodes
        * Cloud‑In‑Cell (CIC) deposit of particle charge onto grid **nodes**.
        * With s=x/dx, j=⌊s⌋, θ=s−j: ρ_j+= (q/dx)(1−θ), ρ_{j+1}+= (q/dx)θ
        * API: positions (Np), charges (Np), dx, N_grid → rho_nodes (N_grid+1)
    """
    rho = np.zeros(N_grid+1)
    s = np.clip(positions/dx, 0.0, N_grid-1-1e-12)
    j = np.floor(s).astype(int)
    th = (positions - j*dx)/dx
    qw = charges/dx
    np.add.at(rho, j,   qw*(1.0-th))
    np.add.at(rho, j+1, qw*th)
    return rho


def gather_linear(positions: np.ndarray, field_nodes: np.ndarray, dx: float, N_grid: int) -> np.ndarray:
    """
        * gather_linear
        * Linear (CIC‑consistent) interpolation of a nodal field back to particle
          positions: f(x) ≈ (1−θ)f_j + θ f_{j+1}.
        * API: positions (Np), field_nodes (N_grid+1), dx, N_grid → f_p (Np)
    """
    s = np.clip(positions/dx, 0.0, N_grid-1-1e-12)
    j = np.floor(s).astype(int)
    th = (positions - j*dx)/dx
    return (1.0-th)*field_nodes[j] + th*field_nodes[j+1]


def dirichlet_green_1d(x_nodes: np.ndarray, xp: float, L: float) -> np.ndarray:
    """
        * dirichlet_green_1d
        * Green's function for −φ'' = δ(x−x') on [0,L] with φ(0)=φ(L)=0.
          Returns G(x,x') evaluated on node array x_nodes.
        * Closed form: G(x,x') = { x(L−x')/L, x≤x' ; x'(L−x)/L, x≥x' }.
        * API: x_nodes (N+1), xp (source point), L → G (N+1)
    """
    xp_arr = np.array(xp)
    return np.where(x_nodes < xp_arr, x_nodes*(L - xp_arr)/L, xp_arr*(L - x_nodes)/L)


def deposit_sheath_layers(q_left: float, q_right: float, dx: float, N_grid: int, smear_cells: int) -> np.ndarray:
    """
        * deposit_sheath_layers
        * Converts accumulated wall (sheath) charges into a near‑wall nodal ρ array
          using a delta‑like layer smeared across a small number of cells.
        * API: q_left, q_right, dx, N_grid, smear_cells → rho_nodes (N_grid+1)
    """
    rho = np.zeros(N_grid + 1)
    # Left wall deposit (at ~0+)
    if q_left != 0.0:
        q = q_left / dx
        rho[0] += q * (1.0)
        for m in range(1, max(1, smear_cells)):
            w = max(0.0, 1.0 - m/float(max(1, smear_cells)))
            idx = min(m, N_grid)
            rho[idx] += q * 0.0 * w  # default keep delta‑like; set nonzero to truly smear
    # Right wall deposit (at ~L−)
    if q_right != 0.0:
        q = q_right / dx
        rho[-1] += q * (1.0)
        for m in range(1, max(1, smear_cells)):
            w = max(0.0, 1.0 - m/float(max(1, smear_cells)))
            idx = max(N_grid - m, 0)
            rho[idx] += q * 0.0 * w
    return rho

# =============================================================================
# Plotting Helpers
# =============================================================================

def render_frame(x_nodes: np.ndarray,
                 phi: np.ndarray,
                 ele_x: np.ndarray,
                 ele_v: np.ndarray,
                 t: float,
                 absorbed_count: int,
                 ylim_v,
                 ylim_phi):
    """
        * render_frame
        * Renders a phase‑space scatter (x,v) for the current step.
        * API: node positions, potential (optional in plot), electron x/v, time,
                absorbed count (for title), axis limits → RGB image (H,W,3)
    """
    fig, ax1 = plt.subplots(figsize=(6.08, 3.04), dpi=100)
    ax1.scatter(ele_x, ele_v, s=0.05)
    ax1.set_xlim(x_nodes[0], x_nodes[-1])
    ax1.set_ylim(*ylim_v)
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Electron velocity')
    ax1.set_title(f'Phase space at t={t:.3f}, absorbed={absorbed_count}')
    fig.tight_layout()
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return img

# =============================================================================
# Main
# =============================================================================

@dataclass
class State:
    ele_x: np.ndarray
    ele_v: np.ndarray
    ele_v_half: np.ndarray
    ele_q: np.ndarray
    sheath_left_q: float
    sheath_right_q: float
    t: float
    step: int


def main() -> None:
    """
        * main
        * Orchestrates the PIC sheath simulation using the Dirichlet BC model
          and leapfrog (default) or explicit Euler (optional) integrators. Electrons
          are absorbed at walls; their charge accumulates into left/right sheath
          reservoirs which source the potential via either Green’s function
          deltas or a near‑wall smear layer.
        * Date Created: 08/12/25
        * Date Last Modified: 09/16/25
        * Author: Beckett Henderson
        * API: None (uses parse_args for CLI). Side effects: writes MP4/PNG.
    """
    args = parse_args()

    # Grid
    L = args.L
    N_grid = args.N_grid
    dx = L / N_grid
    x_nodes = np.linspace(0.0, L, N_grid + 1)

    # Species (ions immobile, optionally included in field)
    rng = np.random.default_rng(args.seed)

    ion_x = np.linspace(0, L, args.N_ion, endpoint=False) + L/(2*args.N_ion) if args.N_ion > 0 else np.zeros(0)
    ion_q = np.full(ion_x.size, args.q_i)

    # Electrons: quasi‑uniform with small jitter, Maxwellian velocities
    base = (np.arange(args.N_electron) + 0.5) / args.N_electron * L
    jitter = (rng.random(args.N_electron) - 0.5) * (0.25 * dx)
    ele_x = np.mod(base + jitter, L)
    ele_v = rng.normal(0.0, args.v_th, args.N_electron)
    ele_q = np.full(args.N_electron, args.q_e)

    # Sheath reservoirs
    sheath_left_q  = 0.0
    sheath_right_q = 0.0

    # Initial field build
    rho_e_0 = cic_deposit_nodes(ele_x, ele_q, dx, N_grid)
    rho_tot_0 = rho_e_0
    if args.include_ions_in_force and ion_x.size:
        rho_ion = cic_deposit_nodes(ion_x, ion_q, dx, N_grid)
        rho_tot_0 = rho_tot_0 + rho_ion

    phi = solve_phi_from_rho(rho_tot_0, dx, args.phi0, args.phib)
    # Add sheath sources
    if args.sheath_mode == 'greens':
        if sheath_left_q != 0.0:
            phi += sheath_left_q  * dirichlet_green_1d(x_nodes, 0.0, L)
        if sheath_right_q != 0.0:
            phi += sheath_right_q * dirichlet_green_1d(x_nodes, L,   L)
    else:  # smear
        phi += solve_phi_from_rho(deposit_sheath_layers(sheath_left_q, sheath_right_q, dx, N_grid, args.smear_cells), dx, args.phi0, args.phib)
    phi[0]  = args.phi0
    phi[-1] = args.phib

    # Field and initial velocity state
    grad_phi = compute_grad(phi, dx)
    E_nodes = grad_phi              # E = +∂φ/∂x
    E_elec  = gather_linear(ele_x, E_nodes, dx, N_grid)

    # Time step selection
    fixed_step_mode = (args.dt is not None and args.steps is not None)
    if fixed_step_mode:
        dt = args.dt
        total_steps = args.steps
        next_frame_step = 0
    else:
        vmax0 = float(np.max(np.abs(ele_v))) if ele_v.size else 0.0
        dt = min(args.dt_max, args.CFL * dx / max(vmax0, args.eps_v))
        next_frame_time = 0.0

    # Integrator state
    if args.euler:
        ele_v_half = None  # not used
    else:
        ele_v_half = ele_v + (ele_q/args.m_e) * E_elec * (0.5*dt)

    # Video writer (optional)
    writer = imageio.get_writer(args.output, fps=args.fps) if args.make_video else None
    frames_written = 0

    t = 0.0
    step = 0
    loss_hist = []
    snap_steps = set()
    if args.snap_t:
        # convert times to target steps for both modes
        if fixed_step_mode:
            snap_steps = set(int(np.round(s/ dt)) for s in args.snap_t)
        else:
            snap_steps = set()  # handled by time comparison

    # First frame
    if writer is not None:
        active0 = (ele_q != 0.0)
        frame0 = render_frame(x_nodes, phi, ele_x[active0], ele_v[active0], t,
                              int(np.count_nonzero(ele_q==0.0)), args.ylim_v, args.ylim_phi)
        writer.append_data(frame0)
        frames_written += 1
        if not fixed_step_mode:
            next_frame_time = t + args.frame_dt
        else:
            next_frame_step = step + (args.save_every if args.save_every else 1)

    pbar = tqdm(total=(args.T_final if not fixed_step_mode else total_steps), desc='Simulating', unit=('s' if not fixed_step_mode else 'step'))

    # Time loop
    while (t < args.T_final) if not fixed_step_mode else (step < total_steps):
        if not fixed_step_mode:
            # Adaptive CFL dt
            active = (ele_q != 0.0)
            vmax = float(np.max(np.abs((ele_v if args.euler else ele_v_half)[active]))) if np.any(active) else 0.0
            dt = min(args.dt_max, args.CFL * dx / max(vmax, args.eps_v))
        # (1) Update positions (DRIFT) and (Euler also updates v after)
        if args.euler:
            # Euler: compute acceleration at n, then update v and x
            ele_x += ele_v * dt
        else:
            # Leapfrog drift
            ele_x += ele_v_half * dt

        # (2) Absorb at walls → accumulate sheath charge
        left_mask  = ele_x < 0.0
        right_mask = ele_x > L
        absorbed = int(np.count_nonzero(left_mask) + np.count_nonzero(right_mask))
        if absorbed:
            if np.any(left_mask):
                q_abs = float(np.sum(ele_q[left_mask]))
                sheath_left_q += q_abs
                ele_q[left_mask] = 0.0
                if not args.euler:
                    ele_v_half[left_mask] = 0.0
                ele_v[left_mask] = 0.0
                ele_x[left_mask] = 0.0
            if np.any(right_mask):
                q_abs = float(np.sum(ele_q[right_mask]))
                sheath_right_q += q_abs
                ele_q[right_mask] = 0.0
                if not args.euler:
                    ele_v_half[right_mask] = 0.0
                ele_v[right_mask] = 0.0
                ele_x[right_mask] = L
        loss_hist.append(absorbed)

        # (3) Deposit charges for field solve
        rho_e   = cic_deposit_nodes(ele_x, ele_q, dx, N_grid)
        rho_tot = rho_e
        if args.include_ions_in_force and ion_x.size:
            rho_ion = cic_deposit_nodes(ion_x, ion_q, dx, N_grid)
            rho_tot = rho_tot + rho_ion

        # (4) Field solve + add sheath sources
        phi = solve_phi_from_rho(rho_tot, dx, args.phi0, args.phib)
        if args.sheath_mode == 'greens':
            if sheath_left_q != 0.0:
                phi += sheath_left_q  * dirichlet_green_1d(x_nodes, 0.0, L)
            if sheath_right_q != 0.0:
                phi += sheath_right_q * dirichlet_green_1d(x_nodes, L,   L)
        else:
            phi += solve_phi_from_rho(deposit_sheath_layers(sheath_left_q, sheath_right_q, dx, N_grid, args.smear_cells), dx, args.phi0, args.phib)
        phi[0]  = args.phi0
        phi[-1] = args.phib

        grad_phi = compute_grad(phi, dx)
        E_nodes = grad_phi          # E = +∂φ/∂x
        E_elec  = gather_linear(ele_x, E_nodes, dx, N_grid)

        # (5) Velocity update
        if args.euler:
            ele_v = ele_v + (ele_q/args.m_e) * E_elec * dt
        else:
            ele_v_half = ele_v_half + (ele_q/args.m_e) * E_elec * dt
            ele_v = ele_v_half - (ele_q/args.m_e) * E_elec * (0.5*dt)

        # (6) Advance time/counters
        if not fixed_step_mode:
            t += dt
            pbar.update(dt)
        else:
            step += 1
            pbar.update(1)

        # (7) Video cadence
        if writer is not None:
            save_frame = False
            if args.save_every is not None and fixed_step_mode:
                if step >= next_frame_step:
                    save_frame = True
                    next_frame_step = step + args.save_every
            else:
                if (not fixed_step_mode and (t >= next_frame_time or t >= args.T_final)) or (fixed_step_mode and step == 1):
                    save_frame = True
                    if not fixed_step_mode:
                        next_frame_time = t + args.frame_dt
            if save_frame:
                active_mask = (ele_q != 0.0)
                frame = render_frame(x_nodes, phi, ele_x[active_mask], ele_v[active_mask], (t if not fixed_step_mode else step*dt),
                                     int(np.count_nonzero(ele_q==0.0)), args.ylim_v, args.ylim_phi)
                writer.append_data(frame)
                frames_written += 1

        # (8) Snapshots of φ(x) at requested times
        if args.snap_t:
            if fixed_step_mode:
                if step in snap_steps:
                    plt.figure(figsize=(6,3))
                    plt.plot(x_nodes, phi)
                    plt.xlabel('x')
                    plt.ylabel('φ')
                    plt.ylim(args.ylim_phi)
                    plt.title(f'Potential at t~{step*dt:.2f}')
                    plt.tight_layout()
                    plt.savefig(f"snapshot_t{step*dt:.2f}.png", dpi=300)
                    plt.close()
            else:
                # find any target times crossed in this step
                for ts in args.snap_t:
                    if (t-dt) < ts <= t:
                        plt.figure(figsize=(6,3))
                        plt.plot(x_nodes, phi)
                        plt.xlabel('x')
                        plt.ylabel('φ')
                        plt.ylim(args.ylim_phi)
                        plt.title(f'Potential at t={ts:.2f}')
                        plt.tight_layout()
                        plt.savefig(f"snapshot_t{ts:.2f}.png", dpi=300)
                        plt.close()

        # increment step in adaptive mode after outputs
        if not fixed_step_mode:
            step += 1

    pbar.close()

    if writer is not None:
        writer.close()

    # Absorption diagnostics plot
    plt.figure(figsize=(6.08, 3.04), dpi=100)
    if len(loss_hist) > 0:
        x_axis = np.linspace(0.0, (t if not fixed_step_mode else step*dt), len(loss_hist))
        plt.plot(x_axis, loss_hist)
    plt.xlabel('Time (sim units)')
    plt.ylabel('Particles absorbed per step')
    plt.title('Absorption Rate (adaptive/fixed dt, e–e + sheath field)')
    plt.tight_layout()
    plt.savefig(args.loss_png, dpi=120)
    plt.close()

    logging.info(f"[export] Completed {(t if not fixed_step_mode else step*dt):.3f} sim‑units in {step} steps.")
    if writer is not None and frames_written > 0:
        logging.info(f"[export] Video saved to {args.output} with {frames_written} frames.")
    else:
        logging.info("[export] Video disabled or zero frames written.")


if __name__ == '__main__':
    main()
