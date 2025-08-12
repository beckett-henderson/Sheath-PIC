import numpy as np
import matplotlib.pyplot as plt
import imageio

# ================================================
# 1D Sheath Simulation (PIC version)
# ----------------------------------
# - Particles: immobile ion background + mobile electrons
# - Field: Poisson solve on a uniform grid with Dirichlet BCs (phi=0 at walls)
# - Charge assignment: CIC (Cloud-In-Cell) deposit to nodes
# - Field solve: Thomas algorithm on tridiagonal system for -phi'' = rho
# - Force gather: linear interpolation of E = -grad(phi) back to particles
# - Sheath accumulation: electrons hitting walls are absorbed and their charge
#   is added to a boundary reservoir, which is smoothed and deposited as a near-wall
#   charge layer on the grid each step (mimicking wall charge buildup).
# - Visualization: phase space + potential each step, MP4 output, and final diagnostics
# ================================================

# --------------------
# Parameters
# --------------------
L = 1.0                # Domain length
N_i = 10000            # Number of immobile ions
N_e = 10000            # Number of electrons
N_grid = 200           # Grid resolution (nodes = N_grid + 1)
dt = 0.01              # Time step
total_steps = 500      # Number of iterations
v_th = 1.0             # Electron thermal velocity (for init only)
np.random.seed(0)

# --------------------
# Initialization
# --------------------
# Ion background: uniform immobile charges (centered in sub-intervals)
ion_positions = np.linspace(0, L, N_i, endpoint=False) + L/(2*N_i)
ion_charges   = np.ones(N_i) * (1/10)

# Electrons: random positions, Maxwellian velocities
electron_positions  = np.random.rand(N_e) * L
electron_charges    = -np.ones(N_e) * (1/10)
electron_velocities = np.random.normal(loc=0.0, scale=v_th, size=N_e)

# Sheath reservoirs (accumulated absorbed electron charge)
sheath_left_charge  = 0.0
sheath_right_charge = 0.0

# Grid
dx = L / N_grid
x_nodes = np.linspace(0.0, L, N_grid + 1)

# --------------------
# Utilities
# --------------------

def thomas_solve(a, b, c, d):
    """Solve tridiagonal system Ax=d with diagonals a(sub), b(main), c(super)."""
    n = len(b)
    cc = c.astype(np.float64).copy()
    dd = d.astype(np.float64).copy()
    bb = b.astype(np.float64).copy()
    # Forward sweep
    cc[0] /= bb[0]
    dd[0] /= bb[0]
    for i in range(1, n):
        denom = bb[i] - a[i-1] * cc[i-1]
        if i < n - 1:
            cc[i] /= denom
        dd[i] = (dd[i] - a[i-1] * dd[i-1]) / denom
    # Back substitution
    x = np.empty(n, dtype=np.float64)
    x[-1] = dd[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - cc[i] * x[i + 1]
    return x


def solve_phi_from_rho(rho_nodes, phi0=0.0, phib=0.0):
    """
    Solve -phi'' = rho on [0, L] with Dirichlet BCs: phi(0)=phi0, phi(L)=phib.
    rho_nodes is defined on grid nodes (length N_grid+1) as per-length charge density.
    """
    # Interior unknowns: nodes 1..N_grid-1 (count n = N_grid-1)
    n = N_grid - 1
    if n <= 0:
        raise ValueError("N_grid must be >= 2")

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


def compute_E_from_phi(phi):
    """Centered differences for E = -dphi/dx at nodes; one-sided at boundaries."""
    E = np.empty_like(phi)
    E[0]      = -(phi[1]   - phi[0])      / dx
    E[-1]     = -(phi[-1]  - phi[-2])     / dx
    E[1:-1]   = -(phi[2:]  - phi[:-2])    / (2.0 * dx)
    return E


def cic_deposit_nodes(positions, charges, per_length=True):
    """
    CIC deposit to grid nodes. Returns rho at nodes (size N_grid+1).
    If per_length=True, divide by dx to produce per-length density (consistent with -phi''=rho).
    """
    rho = np.zeros(N_grid + 1)
    s = np.clip(positions / dx, 0.0, N_grid - 1 - 1e-12)
    j = np.floor(s).astype(int)         # left node index
    th = (positions - j * dx) / dx      # weight to right node
    w0 = (1.0 - th)
    w1 = th
    q = charges.copy()
    if per_length:
        q = q / dx
    np.add.at(rho, j,     q * w0)
    np.add.at(rho, j + 1, q * w1)
    return rho


def gather_linear(positions, field_nodes):
    """Linear (CIC-consistent) interpolation of nodal field back to particle positions."""
    s = np.clip(positions / dx, 0.0, N_grid - 1 - 1e-12)
    j = np.floor(s).astype(int)
    th = (positions - j * dx) / dx
    return (1.0 - th) * field_nodes[j] + th * field_nodes[j + 1]


def deposit_sheath_layers(q_left, q_right, smear_cells=1):
    """
    Convert accumulated boundary charges into a near-wall nodal rho array
    by smearing them over the first/last `smear_cells` cells using CIC.
    This approximates a thin charge layer on the wall.
    """
    rho = np.zeros(N_grid + 1)
    if q_left != 0.0:
        # Represent as a point just inside the domain at x = 0+ (epsilon)
        xL = 0.0 + 1e-9
        s = xL / dx
        j = 0
        th = s - j
        q = q_left / dx
        rho[j]     += q * (1.0 - th)
        rho[j + 1] += q * th
        # Optional: smear deeper (commented out by default)
        # for m in range(1, smear_cells):
        #     w = max(0.0, 1.0 - m / smear_cells)
        #     rho[m] += q * w
    if q_right != 0.0:
        xR = L - 1e-9
        s = xR / dx
        j = int(np.floor(s))
        th = s - j
        q = q_right / dx
        rho[j]     += q * (1.0 - th)
        rho[j + 1] += q * th
        # Optional smear as above
    return rho

# --------------------
# Precompute ion potential (static)
# --------------------
rho_ion_nodes = cic_deposit_nodes(ion_positions, ion_charges, per_length=True)
phi_ion = solve_phi_from_rho(rho_ion_nodes, phi0=0.0, phib=0.0)

# Initial energy (using ion field only + initial electrons)
E_ion_nodes = compute_E_from_phi(phi_ion)
phi_at_elec_init = gather_linear(electron_positions, phi_ion)
phi_at_ions_init = gather_linear(ion_positions,      phi_ion)
PE_elec_init = np.sum(electron_charges * phi_at_elec_init)
PE_ion_init  = np.sum(ion_charges      * phi_at_ions_init)
initial_PE   = PE_elec_init + PE_ion_init
print(f"Initial PE → electrons: {PE_elec_init:.6f}, ions: {PE_ion_init:.6f}, total: {initial_PE:.6f}")

# --------------------
# MP4 writer
# --------------------
writer = imageio.get_writer('electron_evolution.mp4', fps=10)
margin = 0.1 * L

# --------------------
# Time evolution
# --------------------
for t in range(total_steps):
    # (1) Deposit electron charge to nodes (CIC)
    rho_e_nodes = cic_deposit_nodes(electron_positions, electron_charges, per_length=True)

    # (2) Convert sheath reservoirs into near-wall rho and add to electrons
    rho_sheath_nodes = deposit_sheath_layers(sheath_left_charge, sheath_right_charge, smear_cells=1)

    # (3) Total rho = ions (static) + electrons (mobile) + sheath layers
    rho_tot_nodes = rho_ion_nodes + rho_e_nodes + rho_sheath_nodes

    # (4) Solve Poisson for total potential with Dirichlet BCs (walls grounded)
    phi_tot = solve_phi_from_rho(rho_tot_nodes, phi0=0.0, phib=0.0)

    # (5) Compute E field on nodes and gather to electron positions
    E_nodes = compute_E_from_phi(phi_tot)
    E_elec  = gather_linear(electron_positions, E_nodes)

    # (6) Advance electrons (explicit Euler; can switch to leapfrog if desired)
    #     F = q * E; here charges are negative
    accel = (electron_charges * E_elec)  # unit mass assumed
    electron_velocities += accel * dt
    electron_positions  += electron_velocities * dt

    # (7) Absorption at walls → accumulate sheath charge, zero out absorbed particles
    left_mask = electron_positions < 0.0
    if np.any(left_mask):
        sheath_left_charge  += np.sum(electron_charges[left_mask])
        electron_positions[left_mask]  = 0.0
        electron_velocities[left_mask] = 0.0
        electron_charges[left_mask]    = 0.0

    right_mask = electron_positions > L
    if np.any(right_mask):
        sheath_right_charge += np.sum(electron_charges[right_mask])
        electron_positions[right_mask]  = L
        electron_velocities[right_mask] = 0.0
        electron_charges[right_mask]    = 0.0

    # (8) Visualization: phase space + potential
    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax1.scatter(electron_positions, electron_velocities, s=0.1, alpha=0.8)
    ax1.set_ylim(-10, 10)
    ax1.set_xlim(-margin, L + margin)
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Electron velocity')
    absorbed = np.count_nonzero(electron_charges == 0)
    ax1.set_title(f'Phase space t={t*dt:.2f}s, absorbed: {absorbed}')
    ax1.axvline(0, color='grey', linestyle=':', linewidth=1)
    ax1.axvline(L, color='grey', linestyle=':', linewidth=1)

    ax2 = ax1.twinx()
    ax2.plot(x_nodes, phi_tot, lw=1)
    ax2.set_ylabel('Total potential φ')
    ax2.set_ylim(0, 12000)

    fig.tight_layout()
    if np.isclose(t * dt, 3.0, atol=0.5 * dt):
        fig.savefig("snapshot_3s.png", dpi=300)
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    writer.append_data(img)
    plt.close(fig)

writer.close()
print('Saved electron_evolution.mp4')

# --------------------
# Final diagnostics
# --------------------
# Rebuild components for diagnostics
rho_e_final     = cic_deposit_nodes(electron_positions, electron_charges, per_length=True)
rho_sheath_final= deposit_sheath_layers(sheath_left_charge, sheath_right_charge, smear_cells=1)

phi_elec  = solve_phi_from_rho(rho_e_final,      phi0=0.0, phib=0.0)
phi_sheath= solve_phi_from_rho(rho_sheath_final, phi0=0.0, phib=0.0)
phi_ion_f = phi_ion.copy()  # static

phi_tot_final = phi_ion_f + phi_elec + phi_sheath

# Potential energies
phi_at_elec = gather_linear(electron_positions, phi_tot_final)
phi_at_ions = gather_linear(ion_positions,      phi_tot_final)
PE_elec     = np.sum(electron_charges * phi_at_elec)
PE_ion      = np.sum(ion_charges      * phi_at_ions)
final_PE    = PE_elec + PE_ion
print(f"Final  PE → electrons: {PE_elec:.6f}, ions: {PE_ion:.6f}, total: {final_PE:.6f}")

# Component plot
plt.figure(figsize=(6,3))
plt.plot(x_nodes, phi_ion_f,     label='Ion')
plt.plot(x_nodes, phi_elec,      label='Electron')
plt.plot(x_nodes, phi_sheath,    label='Sheath')
plt.plot(x_nodes, phi_tot_final, lw=2, label='Total')
plt.xlabel('x')
plt.ylabel('φ')
plt.title('Final Electrostatic Potential (PIC)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()