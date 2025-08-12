# 1D Plasma Sheath Simulation (PIC Method)

Author: **Beckett Henderson**  
Date Created: **August 12, 2025**  
Last Modified: **August 12, 2025**

---

## 🌄 Overview

This project simulates the **1D plasma sheath** that forms when a plasma with mobile electrons and **immobile background ions** contacts grounded walls.  
It uses a **Particle-in-Cell (PIC)** method with **Cloud-In-Cell (CIC)** charge deposition and a **Thomas algorithm solver** for the electrostatic potential.  
The simulation applies **Dirichlet boundary conditions** (fixed wall potentials) and models **electron absorption** into sheath charge reservoirs that are re-deposited near the walls each time step.  

Integration can be done via **explicit Euler** or optional **velocity-Verlet/leapfrog** for improved stability.

---

## 📚 Non-Dimensionalization

The simulation is fully non-dimensionalized using the following reference scales:

- **Length scale:**  L_ref   = λ_D (Debye length)  
- **Thermal velocity:** v_th (electron thermal velocity)  
- **Time scale:**    T_ref  = L_ref / v_th  
- **Potential scale:** φ₀    = mₑ v_th² / q₀

With dimensionless variables:
- $x' = \\frac{x}{L_\\text{ref}}$  
- $v' = \\frac{v}{v_{th}}$  
- $t' = \\frac{t}{T_\\text{ref}}$  
- $\\phi' = \\frac{\\phi}{\\phi_0}$  
- $E' = \\left(\\frac{q_0 L_\\text{ref}}{m_e v_{th}^2}\\right) E$

---

## 🧪 Governing Equations

- **Kinematics:**  dx'/dt' = v'  
- **Dynamics:**  dv'/dt' = q'·E'  
- **Poisson Equation:**  ∂²φ'/∂x'² = -ρ'  
- **Euler Updates:**
  - v'ₙ₊₁ = v'ₙ + q'·E'(x') · Δt'  
  - x'ₙ₊₁ = x'ₙ + v'ₙ₊₁ · Δt'  
- **Leapfrog Updates:**
  - v'ₙ₊½ = v'ₙ₋½ + q'·E'(x') · Δt'  
  - x'ₙ₊₁ = x'ₙ + v'ₙ₊½ · Δt'  
- **Boundary Conditions:** Dirichlet φ(0) = φ₀, φ(L) = φ_b; absorbing electrons at both walls.

---

## ⚙️ Features

- **CIC charge deposition** from particles to grid nodes.  
- **Thomas algorithm** for solving tridiagonal Poisson systems with Dirichlet BCs.  
- **Electron absorption** at walls with charge stored and re-deposited as a thin sheath layer.  
- Switchable **Euler** or **leapfrog** integration.  
- Configurable **wall potentials**, **smearing depth** of sheath layers, and **particle counts**.  
- **MP4 animation output** with optional **PNG snapshots** at selected times.  
- Fully **CLI-configurable** for reproducible runs.

---

## 🚀 Getting Started

### 🧰 Install Dependencies
```bash
pip install numpy matplotlib imageio argparse tqdm
```

### ⏯️ Run the Simulation
```bash
python plasma_sheath_pic.py
```
Example with custom parameters:
```bash
python plasma_sheath_pic.py --N_ion 5000 --N_electron 5000 --steps 2000 --output sheath.mp4 --snap_t 5 10
```

---

## 📊 Example Output

- **MP4 Animation:** Shows electron phase space and potential evolution as the sheath forms.  
- **Snapshot PNG:** Potential profile at specified simulation times.

<img src="snapshot_t5.00.png" width="400">

---

## 📔 Default Parameters

| Parameter       | Default   | Description                                     |
| --------------- | --------- | ----------------------------------------------- |
| `L`             | 1.0       | Domain length                                   |
| `N_ion`         | 10,000    | Number of immobile ions                         |
| `N_electron`    | 10,000    | Number of mobile electrons                      |
| `N_grid`        | 200       | Number of cells (nodes = N_grid + 1)            |
| `dt`            | 0.01      | Time step                                       |
| `steps`         | 500       | Number of time steps                            |
| `v_th`          | 1.0       | Electron thermal velocity for initialization    |
| `phi0`          | 0.0       | Left wall potential                             |
| `phib`          | 0.0       | Right wall potential                            |
| `smear_cells`   | 1         | Number of cells to smear sheath charge across   |
| `output`        | *.mp4     | Animation filename                              |
| `fps`           | 10        | Frames per second for MP4                       |
| `save_every`    | 1         | Save a frame every N steps                      |
| `snap_t`        | [3.0]     | Times to save PNG snapshots                     |
| `ylim_v`        | [-10, 10] | Velocity plot y-limits                          |
| `ylim_phi`      | [0, 1.2e4]| Potential plot y-limits                         |

---

## 📁 Files Included

- **plasma_sheath_pic.py** — main simulation driver  
- **README.md** — project overview and usage instructions  
- **\*.png** — snapshot outputs (generated at runtime)  
- **\*.mp4** — animation of sheath evolution  

---

## 📚 References

- **Christlieb, A.J., Krasny, R., & Verboncouer, J.P. (2004)**
  *Efficient particle Simulation of a virtual cathode using a grid-free treecode Poisson solver*
  *IEEE transactions on Plasma Science,* Vol. 32, No. 2
  🔗 ieeexplore.ieee.org/document/1308480
- **Christlieb, A.J., Krasny, R., Verboncouer, J.P., Emhoff, J.W., Boyd, I.D. (2006)**
  *Grid-free plasma Simulation techniques*
  *IEEE Transactions on Plasma Science,* Vol. 34, No. 2
  🔗 ieeexplore.ieee.org/document/1621283
- **Causley, M., Christlieb, A., Güçlü, Y., Wolf, E. (2013)**
  *Method of lines transpose: A fast implicit wave propagator*
- **Christlieb, Andrew J., Sands, William A., White, Stephen R. (2025)**
  *A Particle-in-Cell Method for Plasmas with a Generalized Momentum Formulation, Part I: Model   Formulation*
  *Journal of Scientific Computing,* Vol. 103, No. 15
  🔗 link.springer.com/article/10.1007/s10915-025-02824-1
- **White, Stephen R. (2025)**
  *An Involution Satisfying Particle-In-Cell Method*
  *PhD Dissertation, Michigan State University*
  🔗 www.proquest.com/dissertations-theses/involution-satisfying-particle-cell-method/docview/32043919

---

## 💡 Future Additions

- **Diagnostics logging** to CSV.  
- **Periodic boundary option** for testing.  
- **Parallelization** for larger particle counts.  

---

## 🌐 License

This project is available for academic and educational use.  
If you use this code in your research or coursework, please credit the author.
