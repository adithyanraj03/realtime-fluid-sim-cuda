#include "FluidSolver.hpp"
#include "json.hpp"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

using json = nlohmann::json;

FluidSolver::FluidSolver(Grid *grid, const SimConfig &config)
    : grid(grid), config(config), useGpu(false) {
#ifdef USE_CUDA
    this->useGpu = cuda_is_available();
    if (this->useGpu) {
        this->grid->allocGpu();
        this->grid->toGpu();
        cout << "[Solver] Using GPU acceleration" << endl;
    }
#endif
    if (!this->useGpu)
        cout << "[Solver] Using CPU path" << endl;
}

FluidSolver::~FluidSolver() {}

// ======================== main step ======================== //

void FluidSolver::step() {
    bool is3d = this->config.is3D;
    float dt  = this->config.dt;

    if (this->useGpu) {
#ifdef USE_CUDA
        Grid *g = this->grid;
        int W = g->width, H = g->height, D = g->depth;

        if (!is3d) {
            // ---- GPU 2D path ---- //

            //1. vorticity confinement
            cuda_vorticity_2d(g->d_vorticity, g->d_vx, g->d_vy, W, H);
            cuda_vorticity_confinement_2d(g->d_vx, g->d_vy, g->d_vorticity,
                                          W, H, this->config.vorticityStrength, dt);

            //2. diffuse velocity (implicit Jacobi)
            //swap so d_vx0/d_vy0 hold current velocity as Jacobi RHS
            cuda_swap(&g->d_vx, &g->d_vx0);
            cuda_swap(&g->d_vy, &g->d_vy0);
            cuda_diffuse_2d(g->d_vx, g->d_vx0, W, H, this->config.viscosity, dt, this->config.jacobiIterations);
            cuda_diffuse_2d(g->d_vy, g->d_vy0, W, H, this->config.viscosity, dt, this->config.jacobiIterations);

            //3. project (enforce incompressibility ∇·u=0)
            cuda_divergence_2d(g->d_divergence, g->d_vx, g->d_vy, W, H);
            cuda_fill(g->d_pressure, g->size, 0.0f);
            cuda_pressure_jacobi_2d(g->d_pressure, g->d_vx0, g->d_divergence,
                                    W, H, this->config.jacobiIterations);
            cuda_gradient_subtract_2d(g->d_vx, g->d_vy, g->d_pressure, W, H);

            //4. advect velocity (semi-Lagrangian backtrace)
            cuda_advect_2d(g->d_vx0, g->d_vx, g->d_vx, g->d_vy, W, H, dt, this->config.velocityDissipation);
            cuda_advect_2d(g->d_vy0, g->d_vy, g->d_vx, g->d_vy, W, H, dt, this->config.velocityDissipation);
            cuda_swap(&g->d_vx, &g->d_vx0);
            cuda_swap(&g->d_vy, &g->d_vy0);

            //5. project again
            cuda_divergence_2d(g->d_divergence, g->d_vx, g->d_vy, W, H);
            cuda_fill(g->d_pressure, g->size, 0.0f);
            cuda_pressure_jacobi_2d(g->d_pressure, g->d_vx0, g->d_divergence,
                                    W, H, this->config.jacobiIterations);
            cuda_gradient_subtract_2d(g->d_vx, g->d_vy, g->d_pressure, W, H);

            //6. advect + diffuse density
            cuda_advect_2d(g->d_densityR0, g->d_densityR, g->d_vx, g->d_vy, W, H, dt, this->config.dyeDissipation);
            cuda_advect_2d(g->d_densityG0, g->d_densityG, g->d_vx, g->d_vy, W, H, dt, this->config.dyeDissipation);
            cuda_advect_2d(g->d_densityB0, g->d_densityB, g->d_vx, g->d_vy, W, H, dt, this->config.dyeDissipation);
            //d_density*0 now hold advected result — use directly as Jacobi RHS
            cuda_diffuse_2d(g->d_densityR, g->d_densityR0, W, H, this->config.diffusion, dt, this->config.jacobiIterations);
            cuda_diffuse_2d(g->d_densityG, g->d_densityG0, W, H, this->config.diffusion, dt, this->config.jacobiIterations);
            cuda_diffuse_2d(g->d_densityB, g->d_densityB0, W, H, this->config.diffusion, dt, this->config.jacobiIterations);

            //download density to CPU for rendering
            g->densityToHost();
            g->velocityToHost();
            g->pressureToHost();
            cuda_copy_to_host(g->vorticity, g->d_vorticity, g->size);
        } else {
            // ---- GPU 3D path ---- //

            //1. vorticity confinement (3D vector vorticity ω = ∇×u)
            cuda_vorticity_3d(g->d_vorticity, g->d_vorticityY, g->d_vorticityZ,
                              g->d_vx, g->d_vy, g->d_vz, W, H, D);
            cuda_vorticity_confinement_3d(g->d_vx, g->d_vy, g->d_vz,
                                          g->d_vorticity, g->d_vorticityY, g->d_vorticityZ,
                                          W, H, D, this->config.vorticityStrength, dt);

            //2. diffuse velocity
            //swap so d_v*0 hold current velocity as Jacobi RHS
            cuda_swap(&g->d_vx, &g->d_vx0);
            cuda_swap(&g->d_vy, &g->d_vy0);
            cuda_swap(&g->d_vz, &g->d_vz0);
            cuda_diffuse_3d(g->d_vx, g->d_vx0, W, H, D, this->config.viscosity, dt, this->config.jacobiIterations);
            cuda_diffuse_3d(g->d_vy, g->d_vy0, W, H, D, this->config.viscosity, dt, this->config.jacobiIterations);
            cuda_diffuse_3d(g->d_vz, g->d_vz0, W, H, D, this->config.viscosity, dt, this->config.jacobiIterations);

            //3. project
            cuda_divergence_3d(g->d_divergence, g->d_vx, g->d_vy, g->d_vz, W, H, D);
            cuda_fill(g->d_pressure, g->size, 0.0f);
            cuda_pressure_jacobi_3d(g->d_pressure, g->d_vx0, g->d_divergence, W, H, D, this->config.jacobiIterations);
            cuda_gradient_subtract_3d(g->d_vx, g->d_vy, g->d_vz, g->d_pressure, W, H, D);

            //4. advect velocity
            cuda_advect_3d(g->d_vx0, g->d_vx, g->d_vx, g->d_vy, g->d_vz, W, H, D, dt, this->config.velocityDissipation);
            cuda_advect_3d(g->d_vy0, g->d_vy, g->d_vx, g->d_vy, g->d_vz, W, H, D, dt, this->config.velocityDissipation);
            cuda_advect_3d(g->d_vz0, g->d_vz, g->d_vx, g->d_vy, g->d_vz, W, H, D, dt, this->config.velocityDissipation);
            cuda_swap(&g->d_vx, &g->d_vx0);
            cuda_swap(&g->d_vy, &g->d_vy0);
            cuda_swap(&g->d_vz, &g->d_vz0);

            //5. project again
            cuda_divergence_3d(g->d_divergence, g->d_vx, g->d_vy, g->d_vz, W, H, D);
            cuda_fill(g->d_pressure, g->size, 0.0f);
            cuda_pressure_jacobi_3d(g->d_pressure, g->d_vx0, g->d_divergence, W, H, D, this->config.jacobiIterations);
            cuda_gradient_subtract_3d(g->d_vx, g->d_vy, g->d_vz, g->d_pressure, W, H, D);

            //6. advect + diffuse density
            cuda_advect_3d(g->d_densityR0, g->d_densityR, g->d_vx, g->d_vy, g->d_vz, W, H, D, dt, this->config.dyeDissipation);
            cuda_advect_3d(g->d_densityG0, g->d_densityG, g->d_vx, g->d_vy, g->d_vz, W, H, D, dt, this->config.dyeDissipation);
            cuda_advect_3d(g->d_densityB0, g->d_densityB, g->d_vx, g->d_vy, g->d_vz, W, H, D, dt, this->config.dyeDissipation);
            //d_density*0 now hold advected result — use directly as Jacobi RHS
            cuda_diffuse_3d(g->d_densityR, g->d_densityR0, W, H, D, this->config.diffusion, dt, this->config.jacobiIterations);
            cuda_diffuse_3d(g->d_densityG, g->d_densityG0, W, H, D, this->config.diffusion, dt, this->config.jacobiIterations);
            cuda_diffuse_3d(g->d_densityB, g->d_densityB0, W, H, D, this->config.diffusion, dt, this->config.jacobiIterations);

            //download to CPU for rendering
            g->densityToHost();
            g->velocityToHost();
            g->pressureToHost();
        }
#endif
    } else {
        // ---- CPU path ---- //
        if (!is3d) {
            this->vorticityConfinement2D();

            this->grid->swapVelocity();
            this->diffuse2D(this->grid->vx, this->grid->vx0, this->config.viscosity);
            this->diffuse2D(this->grid->vy, this->grid->vy0, this->config.viscosity);

            this->project2D();

            this->grid->swapVelocity();
            this->advect2D(this->grid->vx, this->grid->vx0, this->grid->vx0, this->grid->vy0, this->config.velocityDissipation);
            this->advect2D(this->grid->vy, this->grid->vy0, this->grid->vx0, this->grid->vy0, this->config.velocityDissipation);

            this->project2D();

            this->grid->swapDensity();
            this->advect2D(this->grid->densityR, this->grid->densityR0, this->grid->vx, this->grid->vy, this->config.dyeDissipation);
            this->advect2D(this->grid->densityG, this->grid->densityG0, this->grid->vx, this->grid->vy, this->config.dyeDissipation);
            this->advect2D(this->grid->densityB, this->grid->densityB0, this->grid->vx, this->grid->vy, this->config.dyeDissipation);

            this->grid->swapDensity();
            this->diffuse2D(this->grid->densityR, this->grid->densityR0, this->config.diffusion);
            this->diffuse2D(this->grid->densityG, this->grid->densityG0, this->config.diffusion);
            this->diffuse2D(this->grid->densityB, this->grid->densityB0, this->config.diffusion);
        } else {
            this->vorticityConfinement3D();

            this->grid->swapVelocity();
            this->diffuse3D(this->grid->vx, this->grid->vx0, this->config.viscosity);
            this->diffuse3D(this->grid->vy, this->grid->vy0, this->config.viscosity);
            this->diffuse3D(this->grid->vz, this->grid->vz0, this->config.viscosity);

            this->project3D();

            this->grid->swapVelocity();
            this->advect3D(this->grid->vx, this->grid->vx0, this->grid->vx0, this->grid->vy0, this->grid->vz0, this->config.velocityDissipation);
            this->advect3D(this->grid->vy, this->grid->vy0, this->grid->vx0, this->grid->vy0, this->grid->vz0, this->config.velocityDissipation);
            this->advect3D(this->grid->vz, this->grid->vz0, this->grid->vx0, this->grid->vy0, this->grid->vz0, this->config.velocityDissipation);

            this->project3D();

            this->grid->swapDensity();
            this->advect3D(this->grid->densityR, this->grid->densityR0, this->grid->vx, this->grid->vy, this->grid->vz, this->config.dyeDissipation);
            this->advect3D(this->grid->densityG, this->grid->densityG0, this->grid->vx, this->grid->vy, this->grid->vz, this->config.dyeDissipation);
            this->advect3D(this->grid->densityB, this->grid->densityB0, this->grid->vx, this->grid->vy, this->grid->vz, this->config.dyeDissipation);

            this->grid->swapDensity();
            this->diffuse3D(this->grid->densityR, this->grid->densityR0, this->config.diffusion);
            this->diffuse3D(this->grid->densityG, this->grid->densityG0, this->config.diffusion);
            this->diffuse3D(this->grid->densityB, this->grid->densityB0, this->config.diffusion);
        }
    }
}

// ======================== force injection ======================== //

void FluidSolver::addForce(float x, float y, float z, float fx, float fy, float fz,
                           float dyeR, float dyeG, float dyeB, float radius) {
    if (this->useGpu) {
#ifdef USE_CUDA
        Grid *g = this->grid;
        if (!this->config.is3D) {
            cuda_add_force_2d(g->d_vx, g->d_vy,
                              g->d_densityR, g->d_densityG, g->d_densityB,
                              g->width, g->height,
                              x, y, fx, fy, dyeR, dyeG, dyeB, radius);
        } else {
            cuda_add_force_3d(g->d_vx, g->d_vy, g->d_vz,
                              g->d_densityR, g->d_densityG, g->d_densityB,
                              g->width, g->height, g->depth,
                              x, y, z, fx, fy, fz,
                              dyeR, dyeG, dyeB, radius);
        }
#endif
    } else {
        if (!this->config.is3D)
            this->addForce2D(x, y, fx, fy, dyeR, dyeG, dyeB, radius);
        else
            this->addForce3D(x, y, z, fx, fy, fz, dyeR, dyeG, dyeB, radius);
    }
}

// ======================== 2D CPU solvers ======================== //

//bilinear interpolation at fractional grid position (x,y)
float FluidSolver::bilinearSample(float *field, float x, float y) {
    int W = this->grid->width, H = this->grid->height;

    //clamp to [0.5, W-1.5] to stay inside cell centers
    x = fmaxf(0.5f, fminf((float)(W - 1) - 0.5f, x));
    y = fmaxf(0.5f, fminf((float)(H - 1) - 0.5f, y));

    int x0 = (int)x, y0 = (int)y;
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float sx = x - x0, sy = y - y0;

    return (1 - sx) * (1 - sy) * field[y0 * W + x0] +
           sx       * (1 - sy) * field[y0 * W + x1] +
           (1 - sx) * sy       * field[y1 * W + x0] +
           sx       * sy       * field[y1 * W + x1];
}

//semi-Lagrangian advection: trace particle backwards, sample
//∂q/∂t + u·∇q = 0  →  q(x,t+dt) = q(x - u*dt, t)
void FluidSolver::advect2D(float *field, float *field0, float *vx, float *vy, float dissipation) {
    int W = this->grid->width, H = this->grid->height;
    float dt = this->config.dt;

    #pragma omp parallel for
    for (int j = 1; j < H - 1; j++) {
        for (int i = 1; i < W - 1; i++) {
            int idx = j * W + i;
            //backtrace
            float px = (float)i - dt * vx[idx];
            float py = (float)j - dt * vy[idx];
            field[idx] = dissipation * this->bilinearSample(field0, px, py);
        }
    }
}

//implicit diffusion via Jacobi iteration
//solves (I - ν·dt·∇²)u = u₀
void FluidSolver::diffuse2D(float *field, float *field0, float diff) {
    int W = this->grid->width, H = this->grid->height;
    float a = diff * this->config.dt;
    float denom = 1.0f + 4.0f * a;

    for (int k = 0; k < this->config.jacobiIterations; k++) {
        #pragma omp parallel for
        for (int j = 1; j < H - 1; j++) {
            for (int i = 1; i < W - 1; i++) {
                int idx = j * W + i;
                field[idx] = (field0[idx] + a * (
                    field[idx - 1] + field[idx + 1] +
                    field[idx - W] + field[idx + W]
                )) / denom;
            }
        }
    }
}

//Helmholtz-Hodge pressure projection
//∇²p = ∇·u  then  u = u - ∇p
void FluidSolver::project2D() {
    int W = this->grid->width, H = this->grid->height;
    float *p   = this->grid->pressure;
    float *div = this->grid->divergence;

    //compute divergence ∇·u = ∂vx/∂x + ∂vy/∂y
    #pragma omp parallel for
    for (int j = 1; j < H - 1; j++) {
        for (int i = 1; i < W - 1; i++) {
            int idx = j * W + i;
            div[idx] = -0.5f * (
                this->grid->vx[idx + 1] - this->grid->vx[idx - 1] +
                this->grid->vy[idx + W] - this->grid->vy[idx - W]
            );
        }
    }

    //zero pressure
    memset(p, 0, this->grid->size * sizeof(float));

    //Jacobi iteration for ∇²p = div
    for (int k = 0; k < this->config.jacobiIterations; k++) {
        #pragma omp parallel for
        for (int j = 1; j < H - 1; j++) {
            for (int i = 1; i < W - 1; i++) {
                int idx = j * W + i;
                p[idx] = (div[idx] + p[idx - 1] + p[idx + 1] + p[idx - W] + p[idx + W]) / 4.0f;
            }
        }
    }

    //subtract gradient  u = u - ∇p
    #pragma omp parallel for
    for (int j = 1; j < H - 1; j++) {
        for (int i = 1; i < W - 1; i++) {
            int idx = j * W + i;
            this->grid->vx[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]);
            this->grid->vy[idx] -= 0.5f * (p[idx + W] - p[idx - W]);
        }
    }
}

//vorticity confinement for 2D
//ω = ∂vy/∂x - ∂vx/∂y  (scalar curl in 2D)
//N = ∇|ω| / |∇|ω||    (direction toward vortex core)
//f = ε * (N × ω) * h   (confinement force)
void FluidSolver::vorticityConfinement2D() {
    int W = this->grid->width, H = this->grid->height;
    float *vort = this->grid->vorticity;

    //compute vorticity
    #pragma omp parallel for
    for (int j = 1; j < H - 1; j++) {
        for (int i = 1; i < W - 1; i++) {
            int idx = j * W + i;
            float dvydx = (this->grid->vy[idx + 1] - this->grid->vy[idx - 1]) * 0.5f;
            float dvxdy = (this->grid->vx[idx + W] - this->grid->vx[idx - W]) * 0.5f;
            vort[idx] = dvydx - dvxdy;
        }
    }

    //apply confinement force
    float eps = this->config.vorticityStrength;
    float dt  = this->config.dt;

    #pragma omp parallel for
    for (int j = 2; j < H - 2; j++) {
        for (int i = 2; i < W - 2; i++) {
            int idx = j * W + i;

            //gradient of |ω|
            float dwdx = (fabsf(vort[idx + 1]) - fabsf(vort[idx - 1])) * 0.5f;
            float dwdy = (fabsf(vort[idx + W]) - fabsf(vort[idx - W])) * 0.5f;

            float len = sqrtf(dwdx * dwdx + dwdy * dwdy) + 1e-5f;
            dwdx /= len;
            dwdy /= len;

            //cross product N × ω (in 2D: N × ω·ẑ gives (Ny·ω, -Nx·ω))
            this->grid->vx[idx] += eps * dt * (dwdy * vort[idx]);
            this->grid->vy[idx] -= eps * dt * (dwdx * vort[idx]);
        }
    }
}

//add Gaussian force + dye splat at (cx, cy)
void FluidSolver::addForce2D(float x, float y, float fx, float fy,
                              float dyeR, float dyeG, float dyeB, float radius) {
    int W = this->grid->width, H = this->grid->height;
    float r2 = radius * radius;

    int x0 = max(1, (int)(x - radius - 1));
    int x1 = min(W - 2, (int)(x + radius + 1));
    int y0 = max(1, (int)(y - radius - 1));
    int y1 = min(H - 2, (int)(y + radius + 1));

    for (int j = y0; j <= y1; j++) {
        for (int i = x0; i <= x1; i++) {
            float dx = (float)i - x;
            float dy = (float)j - y;
            float d2 = dx * dx + dy * dy;
            if (d2 < r2) {
                float w = expf(-d2 / (2.0f * r2)); //Gaussian weight
                int idx = j * W + i;
                this->grid->vx[idx] += fx * w;
                this->grid->vy[idx] += fy * w;
                this->grid->densityR[idx] += dyeR * w;
                this->grid->densityG[idx] += dyeG * w;
                this->grid->densityB[idx] += dyeB * w;
            }
        }
    }

    if (this->useGpu) {
#ifdef USE_CUDA
        //re-upload modified region
        this->grid->velocityToGpu();
        this->grid->densityToGpu();
#endif
    }
}

// ======================== 3D CPU solvers ======================== //

//trilinear interpolation at fractional grid position (x,y,z)
float FluidSolver::trilinearSample(float *field, float x, float y, float z) {
    int W = this->grid->width, H = this->grid->height, D = this->grid->depth;

    x = fmaxf(0.5f, fminf((float)(W - 1) - 0.5f, x));
    y = fmaxf(0.5f, fminf((float)(H - 1) - 0.5f, y));
    z = fmaxf(0.5f, fminf((float)(D - 1) - 0.5f, z));

    int x0 = (int)x, y0 = (int)y, z0 = (int)z;
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1), z1 = min(z0 + 1, D - 1);
    float sx = x - x0, sy = y - y0, sz = z - z0;

    //sample 8 corners of the voxel
    float c000 = field[z0 * W * H + y0 * W + x0];
    float c100 = field[z0 * W * H + y0 * W + x1];
    float c010 = field[z0 * W * H + y1 * W + x0];
    float c110 = field[z0 * W * H + y1 * W + x1];
    float c001 = field[z1 * W * H + y0 * W + x0];
    float c101 = field[z1 * W * H + y0 * W + x1];
    float c011 = field[z1 * W * H + y1 * W + x0];
    float c111 = field[z1 * W * H + y1 * W + x1];

    //interpolate along x, then y, then z
    float c00 = c000 * (1 - sx) + c100 * sx;
    float c10 = c010 * (1 - sx) + c110 * sx;
    float c01 = c001 * (1 - sx) + c101 * sx;
    float c11 = c011 * (1 - sx) + c111 * sx;

    float c0 = c00 * (1 - sy) + c10 * sy;
    float c1 = c01 * (1 - sy) + c11 * sy;

    return c0 * (1 - sz) + c1 * sz;
}

//3D semi-Lagrangian advection
void FluidSolver::advect3D(float *field, float *field0, float *vx, float *vy, float *vz, float dissipation) {
    int W = this->grid->width, H = this->grid->height, D = this->grid->depth;
    float dt = this->config.dt;

    #pragma omp parallel for
    for (int k = 1; k < D - 1; k++) {
        for (int j = 1; j < H - 1; j++) {
            for (int i = 1; i < W - 1; i++) {
                int idx = k * W * H + j * W + i;
                float px = (float)i - dt * vx[idx];
                float py = (float)j - dt * vy[idx];
                float pz = (float)k - dt * vz[idx];
                field[idx] = dissipation * this->trilinearSample(field0, px, py, pz);
            }
        }
    }
}

//3D implicit diffusion (6-neighbor Jacobi)
//solves (I - ν·dt·∇²)u = u₀  with 6-neighbor stencil
void FluidSolver::diffuse3D(float *field, float *field0, float diff) {
    int W = this->grid->width, H = this->grid->height, D = this->grid->depth;
    float a = diff * this->config.dt;
    float denom = 1.0f + 6.0f * a;
    int slab = W * H;

    for (int iter = 0; iter < this->config.jacobiIterations; iter++) {
        #pragma omp parallel for
        for (int k = 1; k < D - 1; k++) {
            for (int j = 1; j < H - 1; j++) {
                for (int i = 1; i < W - 1; i++) {
                    int idx = k * slab + j * W + i;
                    field[idx] = (field0[idx] + a * (
                        field[idx - 1]    + field[idx + 1] +
                        field[idx - W]    + field[idx + W] +
                        field[idx - slab] + field[idx + slab]
                    )) / denom;
                }
            }
        }
    }
}

//3D Helmholtz-Hodge projection
//∇²p = ∇·u  then  u = u - ∇p
void FluidSolver::project3D() {
    int W = this->grid->width, H = this->grid->height, D = this->grid->depth;
    int slab = W * H;
    float *p   = this->grid->pressure;
    float *div = this->grid->divergence;

    //compute divergence ∇·u = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
    #pragma omp parallel for
    for (int k = 1; k < D - 1; k++) {
        for (int j = 1; j < H - 1; j++) {
            for (int i = 1; i < W - 1; i++) {
                int idx = k * slab + j * W + i;
                div[idx] = -0.5f * (
                    this->grid->vx[idx + 1]    - this->grid->vx[idx - 1] +
                    this->grid->vy[idx + W]    - this->grid->vy[idx - W] +
                    this->grid->vz[idx + slab] - this->grid->vz[idx - slab]
                );
            }
        }
    }

    //zero pressure
    memset(p, 0, this->grid->size * sizeof(float));

    //6-neighbor Jacobi for ∇²p = div
    for (int iter = 0; iter < this->config.jacobiIterations; iter++) {
        #pragma omp parallel for
        for (int k = 1; k < D - 1; k++) {
            for (int j = 1; j < H - 1; j++) {
                for (int i = 1; i < W - 1; i++) {
                    int idx = k * slab + j * W + i;
                    p[idx] = (div[idx] +
                        p[idx - 1] + p[idx + 1] +
                        p[idx - W] + p[idx + W] +
                        p[idx - slab] + p[idx + slab]
                    ) / 6.0f;
                }
            }
        }
    }

    //subtract gradient
    #pragma omp parallel for
    for (int k = 1; k < D - 1; k++) {
        for (int j = 1; j < H - 1; j++) {
            for (int i = 1; i < W - 1; i++) {
                int idx = k * slab + j * W + i;
                this->grid->vx[idx] -= 0.5f * (p[idx + 1]    - p[idx - 1]);
                this->grid->vy[idx] -= 0.5f * (p[idx + W]    - p[idx - W]);
                this->grid->vz[idx] -= 0.5f * (p[idx + slab] - p[idx - slab]);
            }
        }
    }
}

//3D vorticity confinement
//ω = ∇×u (full 3D vector curl)
//  ωx = ∂uz/∂y - ∂uy/∂z
//  ωy = ∂ux/∂z - ∂uz/∂x
//  ωz = ∂uy/∂x - ∂ux/∂y
//N = ∇|ω| / |∇|ω||
//f = ε * (N × ω) * dt
void FluidSolver::vorticityConfinement3D() {
    int W = this->grid->width, H = this->grid->height, D = this->grid->depth;
    int slab = W * H;
    float *wx = this->grid->vorticity;
    float *wy = this->grid->vorticityY;
    float *wz = this->grid->vorticityZ;

    //compute 3D vector vorticity ω = ∇×u
    #pragma omp parallel for
    for (int k = 1; k < D - 1; k++) {
        for (int j = 1; j < H - 1; j++) {
            for (int i = 1; i < W - 1; i++) {
                int idx = k * slab + j * W + i;

                //central differences
                float duzdx = (this->grid->vz[idx + 1]    - this->grid->vz[idx - 1]) * 0.5f;
                float duzdy = (this->grid->vz[idx + W]    - this->grid->vz[idx - W]) * 0.5f;
                float duzdz = (this->grid->vz[idx + slab] - this->grid->vz[idx - slab]) * 0.5f;
                float duxdy = (this->grid->vx[idx + W]    - this->grid->vx[idx - W]) * 0.5f;
                float duxdz = (this->grid->vx[idx + slab] - this->grid->vx[idx - slab]) * 0.5f;
                float duydx = (this->grid->vy[idx + 1]    - this->grid->vy[idx - 1]) * 0.5f;
                float duydz = (this->grid->vy[idx + slab] - this->grid->vy[idx - slab]) * 0.5f;

                wx[idx] = duzdy - duydz;   //∂uz/∂y - ∂uy/∂z
                wy[idx] = duxdz - duzdx;   //∂ux/∂z - ∂uz/∂x
                wz[idx] = duydx - duxdy;   //∂uy/∂x - ∂ux/∂y
            }
        }
    }

    //apply confinement force
    float eps = this->config.vorticityStrength;
    float dt  = this->config.dt;

    #pragma omp parallel for
    for (int k = 2; k < D - 2; k++) {
        for (int j = 2; j < H - 2; j++) {
            for (int i = 2; i < W - 2; i++) {
                int idx = k * slab + j * W + i;

                //|ω| magnitude
                float mag = sqrtf(wx[idx]*wx[idx] + wy[idx]*wy[idx] + wz[idx]*wz[idx]);

                //gradient of |ω|
                auto magAt = [&](int ii) {
                    return sqrtf(wx[ii]*wx[ii] + wy[ii]*wy[ii] + wz[ii]*wz[ii]);
                };

                float dmdx = (magAt(idx+1) - magAt(idx-1)) * 0.5f;
                float dmdy = (magAt(idx+W) - magAt(idx-W)) * 0.5f;
                float dmdz = (magAt(idx+slab) - magAt(idx-slab)) * 0.5f;

                float len = sqrtf(dmdx*dmdx + dmdy*dmdy + dmdz*dmdz) + 1e-5f;
                float nx = dmdx / len;
                float ny = dmdy / len;
                float nz = dmdz / len;

                //N × ω (cross product)
                float fx = ny * wz[idx] - nz * wy[idx];
                float fy = nz * wx[idx] - nx * wz[idx];
                float fz = nx * wy[idx] - ny * wx[idx];

                this->grid->vx[idx] += eps * dt * fx;
                this->grid->vy[idx] += eps * dt * fy;
                this->grid->vz[idx] += eps * dt * fz;
            }
        }
    }
}

//add Gaussian force + dye splat in 3D at (cx,cy,cz)
void FluidSolver::addForce3D(float x, float y, float z, float fx, float fy, float fz,
                              float dyeR, float dyeG, float dyeB, float radius) {
    int W = this->grid->width, H = this->grid->height, D = this->grid->depth;
    int slab = W * H;
    float r2 = radius * radius;

    int x0 = max(1, (int)(x - radius - 1));
    int x1 = min(W - 2, (int)(x + radius + 1));
    int y0 = max(1, (int)(y - radius - 1));
    int y1 = min(H - 2, (int)(y + radius + 1));
    int z0 = max(1, (int)(z - radius - 1));
    int z1 = min(D - 2, (int)(z + radius + 1));

    for (int k = z0; k <= z1; k++) {
        for (int j = y0; j <= y1; j++) {
            for (int i = x0; i <= x1; i++) {
                float ddx = (float)i - x;
                float ddy = (float)j - y;
                float ddz = (float)k - z;
                float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                if (d2 < r2) {
                    float w = expf(-d2 / (2.0f * r2));
                    int idx = k * slab + j * W + i;
                    this->grid->vx[idx] += fx * w;
                    this->grid->vy[idx] += fy * w;
                    this->grid->vz[idx] += fz * w;
                    this->grid->densityR[idx] += dyeR * w;
                    this->grid->densityG[idx] += dyeG * w;
                    this->grid->densityB[idx] += dyeB * w;
                }
            }
        }
    }

    if (this->useGpu) {
#ifdef USE_CUDA
        this->grid->velocityToGpu();
        this->grid->densityToGpu();
#endif
    }
}

// ======================== config loader ======================== //

SimConfig FluidSolver::loadConfig(const string &path) {
    SimConfig cfg = {};
    cfg.gridWidth         = 256;
    cfg.gridHeight        = 256;
    cfg.gridDepth         = 1;
    cfg.windowWidth       = 800;
    cfg.windowHeight      = 800;
    cfg.viscosity         = 0.0001f;
    cfg.diffusion         = 0.0001f;
    cfg.dt                = 0.016f;
    cfg.jacobiIterations  = 40;
    cfg.vorticityStrength = 0.35f;
    cfg.dyeRadius         = 12.0f;
    cfg.forceStrength     = 300.0f;
    cfg.dyeDissipation    = 0.995f;
    cfg.velocityDissipation = 0.99f;
    cfg.is3D              = false;

    try {
        ifstream f(path);
        if (f.is_open()) {
            json j = json::parse(f);
            if (j.contains("gridWidth"))          cfg.gridWidth         = j["gridWidth"];
            if (j.contains("gridHeight"))         cfg.gridHeight        = j["gridHeight"];
            if (j.contains("gridDepth"))          cfg.gridDepth         = j["gridDepth"];
            if (j.contains("windowWidth"))        cfg.windowWidth       = j["windowWidth"];
            if (j.contains("windowHeight"))       cfg.windowHeight      = j["windowHeight"];
            if (j.contains("viscosity"))          cfg.viscosity         = j["viscosity"];
            if (j.contains("diffusion"))          cfg.diffusion         = j["diffusion"];
            if (j.contains("dt"))                 cfg.dt                = j["dt"];
            if (j.contains("jacobiIterations"))   cfg.jacobiIterations  = j["jacobiIterations"];
            if (j.contains("vorticityStrength"))  cfg.vorticityStrength = j["vorticityStrength"];
            if (j.contains("dyeRadius"))          cfg.dyeRadius         = j["dyeRadius"];
            if (j.contains("forceStrength"))      cfg.forceStrength     = j["forceStrength"];
            if (j.contains("dyeDissipation"))     cfg.dyeDissipation    = j["dyeDissipation"];
            if (j.contains("velocityDissipation"))cfg.velocityDissipation = j["velocityDissipation"];

            cfg.is3D = (cfg.gridDepth > 1);
            cout << "[Config] Loaded from " << path << endl;
        } else {
            cout << "[Config] File not found, using defaults" << endl;
        }
    } catch (const exception &e) {
        cout << "[Config] Parse error: " << e.what() << ", using defaults" << endl;
    }

    cout << "[Config] Grid: " << cfg.gridWidth << "x" << cfg.gridHeight;
    if (cfg.is3D) cout << "x" << cfg.gridDepth << " (3D)";
    else cout << " (2D)";
    cout << endl;

    return cfg;
}
