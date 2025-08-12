#!/usr/bin/env python3
# plasma_sheath_pic.py

"""
1D Plasma Sheath Simulation using Particle-in-Cell (PIC)
Author: Beckett Henderson
Date Created: 08/12/25
Date Last Modified: 08/12/25

This script simulates a 1D plasma sheath with immobile ions and mobile electrons
using a standard PIC field solve on a uniform grid:

  • Charge assignment: CIC (Cloud-In-Cell) to grid nodes
  • Field solve: Dirichlet Poisson problem −φ" = ρ via Thomas algorithm
  • Field gather: linear interpolation of E = −∂φ/∂x back to particle positions
  • Boundaries: grounded walls φ(0)=φ(L)=0; absorbed electrons accumulate as
    “sheath” charge reservoirs that are re‑deposited near the walls each step
  • Time integrator: explicit Euler (default) or velocity‑Verlet/leapfrog (flag)

Non‑dimensionalization:
  Reference scales:
    - Length:    L_ref = λ_D (Debye length)
    - Velocity:  v_th
    - Time:      T_ref = L_ref / v_th
    - Potential: φ_0   = m_e v_th^2 / q_0

  Dimensionless variables (primes omitted in code):
    x' = x / L_ref,  v' = v / v_th,  t' = t / T_ref,  φ' = φ / φ_0,
    E' = (q_0 L_ref / m_e v_th^2) E

  Governing equations in code units:
    ∂²φ/∂x² = −ρ,   E = −∂φ/∂x,
    dx/dt = v,
    dv/dt = q E   (with unit mass; sign carried by q)

Time stepping (Euler default):
    v^{n+1} = v^n + (q E^n) Δt,
    x^{n+1} = x^n + v^{n+1} Δt.

Time stepping (leapfrog option):
    v^{n+1/2} = v^{n−1/2} + (q E^n) Δt,
    x^{n+1}   = x^n + v^{n+1/2} Δt.

All core parameters can be set via command‑line flags; defaults match the
reference setup used by the author.
"""

# =============================================================================
# Imports & Logging
# =============================================================================
# pip install numpy matplotlib imageio tqdm
import argparse
import logging
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import trange

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    '''
        * parse_args
        * Parses command line arguments for simulation parameters.
        * Equations:
            * Kinematic Relation (dimensionless):
                * dx'/dt' = v'
            * Dynamic Relation (dimensionless):
                * dv'/dt' = E'
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * None
            * Outputs:
                * args (argparse.Namespace): parsed command-line arguments

    The argument parser lets you override default simulation parameters
    via command-line flags, so you can easily adjust domain size, particle
    counts, time step, resolution, and other settings without editing the code.

    Default Initial Conditions:
        * Domain Length (L)                 = 1.0
        * Number of immobile ions           = 10000
        * Number of electrons               = 10000
        * Time step (dt)                    = 0.01
        * Total iterations (steps)          = 500
        * Grid resolution (N_grid)          = 200
        * Electron thermal velocity (v_th)  = 1.0
        * Random seed for reproducibility   = 0
        * Left/Right wall potentials        = 0.0 / 0.0
        * Output animation filename         = "electron_evolution.mp4"
        * Snapshot times (snap_t)           = [3.0]
    '''
    p = argparse.ArgumentParser(description="1D Plasma Sheath (PIC: CIC + Thomas)")
    p.add_argument("--L", type=float, default=1.0, help="Domain length")
    p.add_argument("--N_ion", type=int, default=10_000, help="Number of immobile ions")
    p.add_argument("--N_electron", type=int, default=10_000, help="Number of electrons")
    p.add_argument("--N_grid", type=int, default=200, help="Grid resolution (cells); nodes = N_grid+1")
    p.add_argument("--dt", type=float, default=0.01, help="Time step")
    p.add_argument("--steps", type=int, default=500, help="Number of time steps")
    p.add_argument("--v_th", type=float, default=1.0, help="Electron thermal velocity for initialization")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    # Boundary & sheath handling
    p.add_argument("--phi0", type=float, default=0.0, help="Left wall potential φ(0)")
    p.add_argument("--phib", type=float, default=0.0, help="Right wall potential φ(L)")
    p.add_argument("--smear_cells", type=int, default=1, help="How many near‑wall cells to smear sheath charge across")

    # Integrator
    p.add_argument("--leapfrog", action="store_true", help="Use leapfrog/velocity‑Verlet instead of Euler")

    # Output
    p.add_argument("--output", type=str, default="electron_evolution.mp4", help="MP4 filename")
    p.add_argument("--fps", type=int, default=10, help="Frames per second for MP4")
    p.add_argument("--save_every", type=int, default=1, help="Save a frame every N steps to the MP4")
    p.add_argument("--snap_t", type=float, nargs='*', default=[3.0], help="Times (in sim units) to save PNG snapshots")
    p.add_argument("--ylim_v", type=float, nargs=2, default=[-10.0, 10.0], help="y‑limits for velocity plot")
    p.add_argument("--ylim_phi", type=float, nargs=2, default=[0.0, 1.2e4], help="y‑limits for potential plot")

    args = p.parse_args()

    if args.N_grid < 2:
        p.error("--N_grid must be ≥ 2")
    if args.steps <= 0:
        p.error("--steps must be positive")
    if args.dt <= 0:
        p.error("--dt must be positive")

    return args

# =============================================================================
# Numerical Kernels (CIC, Thomas, Gather)
# =============================================================================

def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    '''
        * thomas_solve
        * Solves a tridiagonal linear system A x = d using the Thomas algorithm.
        * Equations:
            * A is tridiagonal with sub-diagonal a, main diagonal b, super-diagonal c
            * Forward elimination then back substitution
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * a (np.ndarray): sub-diagonal (length n-1)
                * b (np.ndarray): main diagonal (length n)
                * c (np.ndarray): super-diagonal (length n-1)
                * d (np.ndarray): right-hand side vector (length n)
            * Outputs:
                * x (np.ndarray): solution vector (length n)

    Numerical Notes:
        * Stable for strictly diagonally dominant tridiagonal matrices arising
          from second-order centered differences of the Poisson operator.
        * O(n) memory and time complexity.
    '''
    n = len(b)
    cc = c.astype(np.float64).copy()
    dd = d.astype(np.float64).copy()
    bb = b.astype(np.float64).copy()
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    for i in range(1, n):
        denom = bb[i] - a[i-1] * cc[i-1]
        if i < n - 1:
            cc[i] /= denom
        dd[i] = (dd[i] - a[i-1] * dd[i-1]) / denom
    x = np.empty(n, dtype=np.float64)
    x[-1] = dd[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - cc[i] * x[i + 1]
    return x


def solve_phi_from_rho(rho_nodes: np.ndarray, dx: float, phi0: float, phib: float) -> np.ndarray:
    """
        * solve_phi_from_rho
        * Solves the Dirichlet Poisson problem −φ'' = ρ on [0, L] using a node-based
          finite-difference discretization and the Thomas tridiagonal solver.
        * Equations:
            * Discrete interior: (−φ_{i-1} + 2φ_i − φ_{i+1}) / dx^2 = ρ_i,  i=1..N_grid−1
            * Boundary conditions: φ_0 = φ(0) = phi0, φ_N = φ(L) = phib
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * rho_nodes (np.ndarray): per-length charge density at nodes (size N_grid+1)
                * dx (float): grid spacing
                * phi0 (float): left boundary potential
                * phib (float): right boundary potential
            * Outputs:
                * phi (np.ndarray): potential at nodes (size N_grid+1)

    Numerical Notes:
        * Builds a tridiagonal linear system for interior nodes only.
        * RHS includes boundary contributions: rhs[0]+=phi0/dx^2, rhs[-1]+=phib/dx^2.
    """
    N_grid = rho_nodes.size - 1
    n = N_grid - 1
    inv_dx2 = 1.0 / dx**2
    a = -np.ones(n - 1) * inv_dx2
    b =  2.0 * np.ones(n)   * inv_dx2
    c = -np.ones(n - 1) * inv_dx2
    rhs = rho_nodes[1:N_grid].copy()
    rhs[0]  += phi0 * inv_dx2
    rhs[-1] += phib * inv_dx2
    phi_int = thomas_solve(a, b, c, rhs)
    phi = np.empty(N_grid + 1)
    phi[0] = phi0
    phi[1:N_grid] = phi_int
    phi[N_grid] = phib
    return phi


def compute_E_from_phi(phi: np.ndarray, dx: float) -> np.ndarray:
    """
        * compute_E_from_phi
        * Computes the electric field on nodes as E = −∂φ/∂x with centered differences
          in the interior and one-sided differences at the boundaries.
        * Equations:
            * E_0   = −(φ_1 − φ_0)/dx
            * E_i   = −(φ_{i+1} − φ_{i−1})/(2 dx),  i=1..N−1
            * E_N   = −(φ_N − φ_{N−1})/dx
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * phi (np.ndarray): potential at nodes (size N_grid+1)
                * dx (float): grid spacing
            * Outputs:
                * E (np.ndarray): electric field at nodes (size N_grid+1)

    Numerical Notes:
        * Second-order accurate in the interior; first-order at the boundaries.
    """
    E = np.empty_like(phi)
    E[0]      = -(phi[1]   - phi[0])    / dx
    E[-1]     = -(phi[-1]  - phi[-2])   / dx
    E[1:-1]   = -(phi[2:]  - phi[:-2])  / (2.0 * dx)
    return E


def cic_deposit_nodes(positions: np.ndarray, charges: np.ndarray, dx: float, N_grid: int) -> np.ndarray:
    """
        * cic_deposit_nodes
        * Deposits particle charges to grid nodes using Cloud-In-Cell (CIC) weighting.
        * Equations:
            * Let s = x/dx, j = ⌊s⌋, θ = s − j
            * ρ_j   += (q/dx) (1−θ)
            * ρ_{j+1} += (q/dx) θ
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * positions (np.ndarray): particle positions
                * charges (np.ndarray): particle charges (carry sign)
                * dx (float): grid spacing
                * N_grid (int): number of cells (nodes = N_grid+1)
            * Outputs:
                * rho (np.ndarray): per-length charge density at nodes (size N_grid+1)

    Numerical Notes:
        * Conserves total charge by construction; compatible with linear gather.
    """
    rho = np.zeros(N_grid + 1)
    s = np.clip(positions / dx, 0.0, N_grid - 1 - 1e-12)
    j = np.floor(s).astype(int)
    th = (positions - j * dx) / dx
    qw = charges / dx
    np.add.at(rho, j,     qw * (1.0 - th))
    np.add.at(rho, j + 1, qw * th)
    return rho


def gather_linear(positions: np.ndarray, field_nodes: np.ndarray, dx: float, N_grid: int) -> np.ndarray:
    """
        * gather_linear
        * Linearly interpolates a nodal field back to particle positions (CIC-consistent gather).
        * Equations:
            * With s = x/dx, j = ⌊s⌋, θ = s − j:
              f(x) ≈ (1−θ) f_j + θ f_{j+1}
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * positions (np.ndarray): particle positions
                * field_nodes (np.ndarray): nodal field values
                * dx (float): grid spacing
                * N_grid (int): number of cells
            * Outputs:
                * f_p (np.ndarray): field interpolated at particle positions

    Numerical Notes:
        * Paired with CIC deposit to reduce self-force and aliasing.
    """
    s = np.clip(positions / dx, 0.0, N_grid - 1 - 1e-12)
    j = np.floor(s).astype(int)
    th = (positions - j * dx) / dx
    return (1.0 - th) * field_nodes[j] + th * field_nodes[j + 1]


def deposit_sheath_layers(q_left: float, q_right: float, dx: float, N_grid: int, smear_cells: int) -> np.ndarray:
    """
        * deposit_sheath_layers
        * Converts accumulated wall (sheath) charges into a near-wall nodal ρ array.
        * Equations:
            * Left: place effective charge at x≈0+; Right: at x≈L−; smear optional.
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * q_left (float): accumulated charge at left wall
                * q_right (float): accumulated charge at right wall
                * dx (float): grid spacing
                * N_grid (int): number of cells
                * smear_cells (int): optional depth to smear into domain
            * Outputs:
                * rho (np.ndarray): per-length charge density at nodes (size N_grid+1)

    Numerical Notes:
        * Default behavior deposits a thin layer (delta-like) at each wall for stability.
    """
    rho = np.zeros(N_grid + 1)
    # Left wall: place at x ≈ 0+
    if q_left != 0.0:
        s = 1e-9 / dx
        j = 0
        th = s - j
        q = q_left / dx
        rho[j]     += q * (1.0 - th)
        rho[j + 1] += q * th
        for m in range(1, max(1, smear_cells)):
            w = max(0.0, 1.0 - m / smear_cells)
            rho[min(m, N_grid)] += q * w * 0.0  # keep default delta-like layer
    # Right wall: place at x ≈ L−
    if q_right != 0.0:
        s = N_grid - 1e-9 / dx
        j = int(np.floor(s))
        th = s - j
        q = q_right / dx
        rho[j]     += q * (1.0 - th)
        rho[j + 1] += q * th
        for m in range(1, max(1, smear_cells)):
            w = max(0.0, 1.0 - m / smear_cells)
            idx = max(0, N_grid - m)
            rho[idx] += q * w * 0.0
    return rho

# =============================================================================
# Initialization & Diagnostics
# =============================================================================

def initialize_particles(L: float, N_i: int, N_e: int, v_th: float, rng: np.random.Generator):
    '''
        * initialize_particles
        * Initializes immobile ions (uniform) and electrons (random positions, Maxwellian velocities).
        * Equations:
            * Ion positions: midpoints of uniform sub-intervals on [0,L]
            * Electron velocities: v ~ Normal(0, v_th)
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * L (float): domain length
                * N_i (int): number of ions (immobile)
                * N_e (int): number of electrons (mobile)
                * v_th (float): thermal velocity scale for initialization
                * rng (np.random.Generator): random number generator
            * Outputs:
                * ion_pos (np.ndarray), ion_q (np.ndarray), ele_pos (np.ndarray), ele_q (np.ndarray), ele_v (np.ndarray)

    Numerical Notes:
        * Charge magnitudes default to ±1/10 for stability and visualization.
    '''
    ion_pos = np.linspace(0, L, N_i, endpoint=False) + L / (2 * N_i)
    ion_q   = np.ones(N_i) * (1/10)
    ele_pos = rng.random(N_e) * L
    ele_q   = -np.ones(N_e) * (1/10)
    ele_v   = rng.normal(loc=0.0, scale=v_th, size=N_e)
    return ion_pos, ion_q, ele_pos, ele_q, ele_v


def potential_energy(charges: np.ndarray, phi_at_particles: np.ndarray) -> float:
    '''
        * potential_energy
        * Computes total potential energy U = Σ q_i φ_i for a set of particles.
        * Equations:
            * U = Σ_i q_i φ(x_i)
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * charges (np.ndarray): particle charges
                * phi_at_particles (np.ndarray): potential evaluated at particle positions
            * Outputs:
                * U (float): total potential energy
    '''
    return float(np.sum(charges * phi_at_particles))

# =============================================================================
# Plotting Helpers
# =============================================================================

def render_frame(x_nodes: np.ndarray,
                 phi_tot: np.ndarray,
                 ele_x: np.ndarray,
                 ele_v: np.ndarray,
                 t: float,
                 absorbed_count: int,
                 ylim_v,
                 ylim_phi):
    '''
        * render_frame
        * Renders a combined phase-space scatter (x,v) and potential profile φ(x) for the current step.
        * Equations:
            * Visualization only (no equations advanced here)
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs:
                * x_nodes (np.ndarray): grid node positions
                * phi_tot (np.ndarray): total potential at nodes
                * ele_x (np.ndarray): electron positions
                * ele_v (np.ndarray): electron velocities
                * t (float): current simulation time
                * absorbed_count (int): number of absorbed electrons so far
                * ylim_v (tuple): y-limits for velocity axes
                * ylim_phi (tuple): y-limits for potential axes
            * Outputs:
                * img (np.ndarray): RGB image array for video writer
    '''
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.scatter(ele_x, ele_v, s=0.1, alpha=0.8)
    ax1.set_xlim(-0.1 * x_nodes[-1], 1.1 * x_nodes[-1])
    ax1.set_ylim(*ylim_v)
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Electron velocity')
    ax1.axvline(x_nodes[0], color='grey', linestyle=':', linewidth=1)
    ax1.axvline(x_nodes[-1], color='grey', linestyle=':', linewidth=1)
    ax1.set_title(f'Phase space t={t:.2f}, absorbed: {absorbed_count}')

    ax2 = ax1.twinx()
    ax2.plot(x_nodes, phi_tot, lw=1)
    ax2.set_ylabel('Total potential φ')
    ax2.set_ylim(*ylim_phi)

    fig.tight_layout()
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return img

# =============================================================================
# Main
# =============================================================================

def main():
    '''
        * main
        * Orchestrates the PIC sheath simulation: initialization, time stepping,
          field solve, particle push, boundary absorption, and I/O.
        * Equations:
            * Particle push (Euler default):
                * v^{n+1} = v^n + (q E^n) Δt
                * x^{n+1} = x^n + v^{n+1} Δt
            * Poisson solve each step: −φ'' = ρ (ions + electrons + sheath)
        * Date Created: 08/12/25
        * Date Last Modified: 08/12/25
        * Author: Beckett Henderson
        * API:
            * Inputs: None (uses parse_args for CLI)
            * Outputs: MP4 animation, optional PNG snapshots, and log diagnostics

    Default Initial Conditions:
        * Domain Length (L)                 = 1.0
        * Number of immobile ions           = 10000
        * Number of electrons               = 10000
        * Time step (dt)                    = 0.01
        * Total iterations (steps)          = 500
        * Grid resolution (N_grid)          = 200
        * Electron thermal velocity (v_th)  = 1.0
        * Random seed for reproducibility   = 0
        * Output animation filename         = "electron_evolution.mp4"
        * Snapshot times (snap_t)           = [3.0]
    '''
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Grid
    L = args.L
    N_grid = args.N_grid
    dx = L / N_grid
    x_nodes = np.linspace(0.0, L, N_grid + 1)

    # Particles
    ion_x, ion_q, ele_x, ele_q, ele_v = initialize_particles(L, args.N_ion, args.N_electron, args.v_th, rng)

    # Sheath reservoirs
    sheath_left = 0.0
    sheath_right = 0.0

    # Precompute static ion potential
    rho_ion = cic_deposit_nodes(ion_x, ion_q, dx, N_grid)
    phi_ion = solve_phi_from_rho(rho_ion, dx, args.phi0, args.phib)

    # Initial energy report
    phi_elec_init = solve_phi_from_rho(cic_deposit_nodes(ele_x, ele_q, dx, N_grid), dx, args.phi0, args.phib)
    phi_tot_init = phi_ion + phi_elec_init
    PE_elec_init = potential_energy(ele_q, gather_linear(ele_x, phi_tot_init, dx, N_grid))
    PE_ion_init  = potential_energy(ion_q, gather_linear(ion_x,  phi_tot_init, dx, N_grid))
    logging.info(f"Initial PE → electrons: {PE_elec_init:.6f}, ions: {PE_ion_init:.6f}, total: {PE_elec_init + PE_ion_init:.6f}")

    # Integrator state (for leapfrog)
    if args.leapfrog:
        # Kick by half step using initial field
        E_nodes = compute_E_from_phi(phi_tot_init, dx)
        E_elec  = gather_linear(ele_x, E_nodes, dx, N_grid)
        ele_v  += (ele_q * E_elec) * (0.5 * args.dt)

    writer = imageio.get_writer(args.output, fps=args.fps)

    # Time loop
    snap_times = set(np.round(np.array(args.snap_t) / args.dt).astype(int)) if args.snap_t else set()

    for step in trange(args.steps, desc="Simulating"):
        t = step * args.dt

        # (1) Deposit electron charge + sheath layers + ions
        rho_e = cic_deposit_nodes(ele_x, ele_q, dx, N_grid)
        rho_s = deposit_sheath_layers(sheath_left, sheath_right, dx, N_grid, args.smear_cells)
        rho_tot = rho_ion + rho_e + rho_s

        # (2) Solve Poisson and compute E
        phi_tot = solve_phi_from_rho(rho_tot, dx, args.phi0, args.phib)
        E_nodes = compute_E_from_phi(phi_tot, dx)
        E_elec  = gather_linear(ele_x, E_nodes, dx, N_grid)

        # (3) Push
        if args.leapfrog:
            ele_v += (ele_q * E_elec) * args.dt
            ele_x += ele_v * args.dt
        else:
            ele_v += (ele_q * E_elec) * args.dt
            ele_x += ele_v * args.dt

        # (4) Absorption at walls
        left_mask = ele_x < 0.0
        if np.any(left_mask):
            sheath_left  += float(np.sum(ele_q[left_mask]))
            ele_x[left_mask]  = 0.0
            ele_v[left_mask]  = 0.0
            ele_q[left_mask]  = 0.0
        right_mask = ele_x > L
        if np.any(right_mask):
            sheath_right += float(np.sum(ele_q[right_mask]))
            ele_x[right_mask] = L
            ele_v[right_mask] = 0.0
            ele_q[right_mask] = 0.0

        # (5) Output frame & snapshots
        if step % args.save_every == 0:
            frame = render_frame(x_nodes, phi_tot, ele_x, ele_v, t, int(np.count_nonzero(ele_q == 0)), args.ylim_v, args.ylim_phi)
            writer.append_data(frame)
        if step in snap_times:
            plt.figure(figsize=(6,3))
            plt.plot(x_nodes, phi_tot)
            plt.xlabel('x')
            plt.ylabel('φ')
            plt.title(f'Potential at t={t:.2f}')
            plt.tight_layout()
            plt.savefig(f"snapshot_t{t:.2f}.png", dpi=300)
            plt.close()

    writer.close()

    # Final diagnostics
    phi_e_final = solve_phi_from_rho(cic_deposit_nodes(ele_x, ele_q, dx, N_grid), dx, args.phi0, args.phib)
    phi_s_final = solve_phi_from_rho(deposit_sheath_layers(sheath_left, sheath_right, dx, N_grid, args.smear_cells), dx, args.phi0, args.phib)
    phi_tot_final = phi_ion + phi_e_final + phi_s_final

    PE_elec = potential_energy(ele_q, gather_linear(ele_x, phi_tot_final, dx, N_grid))
    PE_ion  = potential_energy(ion_q, gather_linear(ion_x,  phi_tot_final, dx, N_grid))
    logging.info(f"Final  PE → electrons: {PE_elec:.6f}, ions: {PE_ion:.6f}, total: {PE_elec + PE_ion:.6f}")

    logging.info(f"Saved animation to {args.output}")

if __name__ == "__main__":
    main()
