# CUDA Fluid Simulator — GPU-Accelerated 2D/3D Navier-Stokes Solver

![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)
![Language: C++17](https://img.shields.io/badge/Language-C++17-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg?logo=nvidia)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3-5586A4.svg?logo=opengl)
![Real-Time](https://img.shields.io/badge/Real--Time-Interactive-ff69b4.svg)

A **real-time 2D/3D fluid dynamics simulator** solving the incompressible Navier-Stokes equations from scratch in C++/CUDA. No physics engine, no simulation library — every PDE solver, advection scheme, and pressure projection is hand-coded with explicit math. GPU-accelerated with custom CUDA kernels, rendered interactively via OpenGL.

**2D mode:** Click and drag to inject colorful dye and force into the fluid. Watch vortices form, swirl, and dissipate in real time on a fullscreen textured quad.

**3D mode:** Volumetric fluid simulation on a cubic grid with real-time volume raymarching. Orbit the camera to view swirling dye clouds from any angle.

<br>

## 🚀 Key Features

- **2D/3D Navier-Stokes Solver** — Full incompressible fluid simulation: $\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$
- **Semi-Lagrangian Advection** — Unconditionally stable backtracing with bilinear (2D) / trilinear (3D) interpolation
- **Pressure Projection** — Poisson equation $\nabla^2 p = \nabla \cdot \mathbf{u}$ solved via Jacobi iteration (4-neighbor 2D / 6-neighbor 3D)
- **Vorticity Confinement** — 2D scalar curl or 3D vector curl $\boldsymbol{\omega} = \nabla \times \mathbf{u}$ with confinement force $\mathbf{f}_{conf} = \varepsilon \cdot (\hat{\eta} \times \boldsymbol{\omega})$
- **RGB Dye System** — Three independent density channels for colorful fluid mixing
- **CUDA GPU Acceleration** — All solver steps offloaded to custom CUDA kernels (2D and 3D variants)
- **Volume Raymarching (3D)** — Real-time GLSL raymarcher with front-to-back compositing through a 3D density texture
- **Orbiting Camera (3D)** — Right-click drag to orbit, scroll to zoom, look-at camera with perspective projection
- **Real-Time OpenGL Rendering** — 2D: textured fullscreen quad | 3D: raymarched volume
- **Mouse Interaction** — Click and drag to inject force + dye with Gaussian falloff
- **Multiple Visualizations** — Dye (RGB), velocity magnitude, pressure field, vorticity (curl)
- **JSON Configuration** — All simulation parameters tunable via config file
- **CLI Flags** — `--3d` to force 3D mode, `--sim` for hands-free demo, `--config <path>` for custom config
- **Demo Mode (`--sim`)** — Automated showcase: 15s of 2D fluid (opposing jets, vortex rings, von Kármán street), then transitions to 3D (helical jets, breathing emitters, orbiting swirl) with auto-orbiting camera

## 📐 Mathematical Foundation

### Governing Equations

The incompressible Navier-Stokes equations (2D and 3D):

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

where $\mathbf{u} = (u, v)$ in 2D or $\mathbf{u} = (u, v, w)$ in 3D.

### Operator Splitting (Stam's Stable Fluids)

Solved via operator splitting — each physical effect handled independently:

| Step | Operation | Method | Equation |
|------|-----------|--------|----------|
| 1 | **Vorticity Confinement** | Curl + normalized gradient | $\omega = \nabla \times \mathbf{u}$, $\mathbf{f} = \varepsilon(\hat{\eta} \times \omega)$ |
| 2 | **Diffusion** | Implicit Jacobi iteration | $(I - \nu \Delta t \nabla^2)\mathbf{u}^{n+1} = \mathbf{u}^n$ |
| 3 | **Projection** | Poisson + gradient subtraction | $\nabla^2 p = \nabla \cdot \mathbf{u}$, $\mathbf{u} = \mathbf{u} - \nabla p$ |
| 4 | **Advection** | Semi-Lagrangian backtrace | $\mathbf{u}(\mathbf{x}, t+\Delta t) = \mathbf{u}(\mathbf{x} - \Delta t \cdot \mathbf{u}, t)$ |
| 5 | **Dye Transport** | Semi-Lagrangian (per RGB channel) | $\rho(\mathbf{x}, t+\Delta t) = \rho(\mathbf{x} - \Delta t \cdot \mathbf{u}, t)$ |

### Numerical Methods

| Component | 2D | 3D | File |
|-----------|----|----|------|
| **Advection** | Bilinear backtrace: $\mathbf{x}_{src} = \mathbf{x} - \Delta t \cdot \mathbf{u}$ | Trilinear backtrace (8 corners) | `FluidSolver.cpp` |
| **Diffusion** | 4-neighbor Jacobi: $\frac{x^0 + a \sum x_{nb}}{1 + 4a}$ | 6-neighbor Jacobi: $\frac{x^0 + a \sum x_{nb}}{1 + 6a}$ | `FluidSolver.cpp` |
| **Divergence** | $\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$ | $+ \frac{\partial w}{\partial z}$ | `FluidSolver.cpp` |
| **Pressure** | 4-neighbor: $\frac{p_{nb} + \text{div}}{4}$ | 6-neighbor: $\frac{p_{nb} + \text{div}}{6}$ | `FluidSolver.cpp` |
| **Vorticity** | Scalar: $\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$ | Vector: $\boldsymbol{\omega} = \nabla \times \mathbf{u}$ (3 components) | `FluidSolver.cpp` |
| **Force** | 2D Gaussian splat | 3D Gaussian splat with oscillating $f_z$ | `FluidSolver.cpp` |

## 📋 Requirements

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.18+
- **OpenGL 3.3+** capable GPU
- **GLFW 3.4** (auto-downloaded via CMake FetchContent)
- **GLAD** (vendored, auto-generated)
- **CUDA Toolkit 12.x** (optional, for GPU acceleration)
- **OpenMP** (optional, for parallel CPU fallback)

## 📥 Building

### Windows (MSVC + CUDA — Ninja)
```powershell
.\build.bat
```

### Linux / macOS
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Linux (with CUDA)
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make -j$(nproc)
```

## 📖 Usage

### Running
```bash
# 2D mode (default — 512×512 grid)
./fluid_sim

# 3D mode (128³ cubic volume with volume raymarching)
./fluid_sim --3d

# demo mode (hands-free: 2D showcase 15s → 3D showcase, ESC to skip/quit)
./fluid_sim --sim

# custom config file
./fluid_sim --config path/to/config.json

# 3D with custom config
./fluid_sim --3d --config config/sim.json
```

### Controls

| Input | 2D Mode | 3D Mode |
|-------|---------|--------|
| **Left-click + drag** | Inject dye + force | Inject dye + force (at mid-depth) |
| **Right-click + drag** | — | Orbit camera |
| **Scroll wheel** | Change brush radius | Zoom camera |
| **SPACE** | Pause / Resume | Pause / Resume |
| **R** | Reset simulation | Reset simulation |
| **V** | Cycle visualization | Cycle visualization |
| **ESC** | Quit | Quit |

> **Demo mode (`--sim`):** Left-click is disabled. The simulation auto-injects scripted patterns. In 3D, the camera auto-orbits. Right-click drag and scroll still work for manual camera control. ESC skips to the next phase or quits.

### Configuration (`config/sim.json`)
```json
{
    "gridWidth": 512,
    "gridHeight": 512,
    "gridDepth": 1,
    "windowWidth": 900,
    "windowHeight": 900,
    "viscosity": 0.00001,
    "diffusion": 0.00001,
    "dt": 0.016,
    "jacobiIterations": 40,
    "vorticityStrength": 0.35,
    "dyeRadius": 15.0,
    "forceStrength": 5000.0,
    "dyeDissipation": 0.997,
    "velocityDissipation": 0.999
}
```

> **Tip:** Set `"gridDepth": 128` to permanently enable 3D mode via config (no `--3d` flag needed).

## 🛠️ Technical Details

### Architecture

```
src/
├── main.cpp              Entry point, CLI parsing, CUDA init
├── Simulation.cpp        Main loop, GLFW window, mouse input
├── FluidSolver.cpp       Navier-Stokes solver (2D + 3D, CPU + GPU paths)
├── Grid.cpp              Memory management, CPU↔GPU transfers
├── Renderer.cpp          OpenGL rendering (2D quad + 3D volume raymarcher)
└── cuda_kernels.cu       16 CUDA kernels (8 × 2D + 8 × 3D)

include/
├── FluidSolver.hpp       SimConfig struct, solver class
├── Grid.hpp              2D/3D grid with GPU mirror pointers
├── Renderer.hpp          2D texture + 3D volume renderer
├── Simulation.hpp        Main loop orchestration
└── cuda_kernels.cuh      CUDA kernel declarations
```

### CUDA Kernels

| Kernel | 2D | 3D | Description |
|--------|----|----|-------------|
| `k_advect` | 16×16 blocks | 8×8×4 blocks | Semi-Lagrangian backtrace |
| `k_diffuse` | 16×16 | 8×8×4 | Jacobi iteration (4/6-neighbor) |
| `k_divergence` | 16×16 | 8×8×4 | Central difference ∇·u |
| `k_pressure_jacobi` | 16×16 | 8×8×4 | Poisson solver with ping-pong |
| `k_gradient_subtract` | 16×16 | 8×8×4 | u = u - ∇p |
| `k_vorticity` | 16×16 | 8×8×4 | Scalar curl / vector curl |
| `k_vorticity_confinement` | 16×16 | 8×8×4 | N × ω confinement force |
| `k_add_force` | 16×16 | 8×8×4 | Gaussian force + dye splat |

### Visualization Modes

| Mode | Key | Description |
|------|-----|-------------|
| **Dye (RGB)** | V | Direct RGB density → color |
| **Velocity** | V | Velocity magnitude → heat map (blue→cyan→green→yellow→red) |
| **Pressure** | V | Pressure field → diverging color map |
| **Vorticity** | V | Curl of velocity → heat map |

### 3D Volume Rendering

The 3D renderer uses **front-to-back raymarching** through a `GL_TEXTURE_3D` density volume:

1. Camera orbits on a sphere (spherical coordinates θ, φ)
2. For each pixel, reconstruct ray via inverse(VP) matrix
3. Ray-AABB intersection with unit cube [0,1]³
4. March along ray in fixed steps, sampling 3D texture
5. Front-to-back alpha compositing: $C_{out} = C_{out} + (1 - \alpha_{out}) \cdot \alpha_s \cdot C_s$

### Key Design Decisions

1. **Collocated Grid** — All fields stored at cell centers (simpler than staggered MAC grid, works well with Chorin's projection method)
2. **Operator Splitting** — Each physical effect (advection, diffusion, projection) solved independently per timestep. Unconditionally stable.
3. **Neumann Boundary** — Zero-gradient at walls (∂u/∂n = 0). Fluid slides along boundaries.
4. **Ping-Pong Buffers** — Jacobi iteration alternates between two buffers to avoid read-after-write hazards on GPU.
5. **Float Precision** — Single-precision throughout (2× CUDA performance vs double, sufficient for real-time visualization).
6. **Dual-mode Architecture** — Same solver class handles both 2D and 3D with separate kernel dispatch paths, keeping code DRY.

## 📜 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Jos Stam** — *"Stable Fluids"* (SIGGRAPH 1999) — the foundational algorithm
- **"Fluid Simulation for Computer Graphics"** — Robert Bridson
- **GLFW** — Cross-platform windowing library
- **GLAD** — OpenGL function loader

---

© 2026 Adithyanraj✨
