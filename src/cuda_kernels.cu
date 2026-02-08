#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

using namespace std;

// ======================== device init ======================== //

static bool g_cudaReady = false;

int cuda_init() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        cout << "[CUDA] No CUDA devices found" << endl;
        return -1;
    }
    cudaSetDevice(0);
    g_cudaReady = true;
    return 0;
}

bool cuda_is_available() { return g_cudaReady; }

void cuda_shutdown() {
    if (g_cudaReady) {
        cudaDeviceReset();
        g_cudaReady = false;
    }
}

void cuda_print_stats() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "[CUDA] Device: " << prop.name << endl;
    cout << "[CUDA] SMs: " << prop.multiProcessorCount
         << "  VRAM: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << endl;
}

// ======================== helpers ======================== //

void cuda_alloc(float **d_ptr, int count) {
    cudaMalloc((void**)d_ptr, count * sizeof(float));
}

void cuda_free(float *d_ptr) {
    if (d_ptr) cudaFree(d_ptr);
}

void cuda_copy_to_device(float *d_dst, const float *h_src, int count) {
    cudaMemcpy(d_dst, h_src, count * sizeof(float), cudaMemcpyHostToDevice);
}

void cuda_copy_to_host(float *h_dst, const float *d_src, int count) {
    cudaMemcpy(h_dst, d_src, count * sizeof(float), cudaMemcpyDeviceToHost);
}

void cuda_swap(float **a, float **b) {
    float *tmp = *a; *a = *b; *b = tmp;
}

// ======================== fill kernel ======================== //

__global__ void k_fill(float *field, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) field[idx] = value;
}

void cuda_fill(float *d_field, int size, float value) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_fill<<<blocks, threads>>>(d_field, size, value);
}

// ============================================================ //
//                         2D KERNELS                           //
// ============================================================ //

// ---- advect 2D ---- //
__global__ void k_advect_2d(float *out, const float *field, const float *vx, const float *vy,
                            int W, int H, float dt, float dissipation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1) return;

    int idx = j * W + i;
    float px = (float)i - dt * vx[idx];
    float py = (float)j - dt * vy[idx];

    //clamp
    px = fmaxf(0.5f, fminf((float)(W-1) - 0.5f, px));
    py = fmaxf(0.5f, fminf((float)(H-1) - 0.5f, py));

    int x0 = (int)px, y0 = (int)py;
    int x1 = min(x0+1, W-1), y1 = min(y0+1, H-1);
    float sx = px - x0, sy = py - y0;

    out[idx] = dissipation * (
        (1-sx)*(1-sy)*field[y0*W+x0] + sx*(1-sy)*field[y0*W+x1] +
        (1-sx)*sy*field[y1*W+x0]     + sx*sy*field[y1*W+x1]
    );
}

void cuda_advect_2d(float *d_out, const float *d_field, const float *d_vx, const float *d_vy,
                    int W, int H, float dt, float dissipation) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    k_advect_2d<<<grid, block>>>(d_out, d_field, d_vx, d_vy, W, H, dt, dissipation);
}

// ---- diffuse 2D ---- //
__global__ void k_diffuse_2d(float *field, const float *field0, int W, int H, float a, float denom) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1) return;

    int idx = j * W + i;
    field[idx] = (field0[idx] + a * (
        field[idx-1] + field[idx+1] + field[idx-W] + field[idx+W]
    )) / denom;
}

void cuda_diffuse_2d(float *d_field, float *d_field0, int W, int H, float diff, float dt, int iters) {
    float a = diff * dt;
    float denom = 1.0f + 4.0f * a;
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    for (int k = 0; k < iters; k++) {
        k_diffuse_2d<<<grid, block>>>(d_field, d_field0, W, H, a, denom);
    }
}

// ---- divergence 2D ---- //
__global__ void k_divergence_2d(float *div, const float *vx, const float *vy, int W, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1) return;

    int idx = j * W + i;
    div[idx] = -0.5f * (vx[idx+1] - vx[idx-1] + vy[idx+W] - vy[idx-W]);
}

void cuda_divergence_2d(float *d_div, const float *d_vx, const float *d_vy, int W, int H) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    k_divergence_2d<<<grid, block>>>(d_div, d_vx, d_vy, W, H);
}

// ---- pressure Jacobi 2D ---- //
__global__ void k_pressure_jacobi_2d(float *p_out, const float *p_in, const float *div, int W, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1) return;

    int idx = j * W + i;
    p_out[idx] = (div[idx] + p_in[idx-1] + p_in[idx+1] + p_in[idx-W] + p_in[idx+W]) / 4.0f;
}

void cuda_pressure_jacobi_2d(float *d_p, float *d_p_temp, const float *d_div, int W, int H, int iters) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    for (int k = 0; k < iters; k++) {
        k_pressure_jacobi_2d<<<grid, block>>>(d_p_temp, d_p, d_div, W, H);
        cuda_swap(&d_p, &d_p_temp);
    }
}

// ---- gradient subtract 2D ---- //
__global__ void k_gradient_subtract_2d(float *vx, float *vy, const float *p, int W, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1) return;

    int idx = j * W + i;
    vx[idx] -= 0.5f * (p[idx+1] - p[idx-1]);
    vy[idx] -= 0.5f * (p[idx+W] - p[idx-W]);
}

void cuda_gradient_subtract_2d(float *d_vx, float *d_vy, const float *d_p, int W, int H) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    k_gradient_subtract_2d<<<grid, block>>>(d_vx, d_vy, d_p, W, H);
}

// ---- vorticity 2D ---- //
__global__ void k_vorticity_2d(float *vort, const float *vx, const float *vy, int W, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1) return;

    int idx = j * W + i;
    float dvydx = (vy[idx+1] - vy[idx-1]) * 0.5f;
    float dvxdy = (vx[idx+W] - vx[idx-W]) * 0.5f;
    vort[idx] = dvydx - dvxdy;
}

void cuda_vorticity_2d(float *d_vort, const float *d_vx, const float *d_vy, int W, int H) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    k_vorticity_2d<<<grid, block>>>(d_vort, d_vx, d_vy, W, H);
}

// ---- vorticity confinement 2D ---- //
__global__ void k_vorticity_confinement_2d(float *vx, float *vy, const float *vort,
                                           int W, int H, float eps, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || i >= W-2 || j < 2 || j >= H-2) return;

    int idx = j * W + i;
    float dwdx = (fabsf(vort[idx+1]) - fabsf(vort[idx-1])) * 0.5f;
    float dwdy = (fabsf(vort[idx+W]) - fabsf(vort[idx-W])) * 0.5f;
    float len = sqrtf(dwdx*dwdx + dwdy*dwdy) + 1e-5f;
    dwdx /= len;
    dwdy /= len;

    vx[idx] += eps * dt * (dwdy * vort[idx]);
    vy[idx] -= eps * dt * (dwdx * vort[idx]);
}

void cuda_vorticity_confinement_2d(float *d_vx, float *d_vy, const float *d_vort,
                                   int W, int H, float epsilon, float dt) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    k_vorticity_confinement_2d<<<grid, block>>>(d_vx, d_vy, d_vort, W, H, epsilon, dt);
}

// ---- add force 2D ---- //
__global__ void k_add_force_2d(float *vx, float *vy,
                               float *densR, float *densG, float *densB,
                               int W, int H,
                               float cx, float cy, float fx, float fy,
                               float dyeR, float dyeG, float dyeB, float radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1) return;

    float dx = (float)i - cx;
    float dy = (float)j - cy;
    float d2 = dx*dx + dy*dy;
    float r2 = radius * radius;

    if (d2 < r2) {
        float w = expf(-d2 / (2.0f * r2));
        int idx = j * W + i;
        vx[idx] += fx * w;
        vy[idx] += fy * w;
        densR[idx] += dyeR * w;
        densG[idx] += dyeG * w;
        densB[idx] += dyeB * w;
    }
}

void cuda_add_force_2d(float *d_vx, float *d_vy,
                       float *d_densR, float *d_densG, float *d_densB,
                       int W, int H,
                       float cx, float cy, float fx, float fy,
                       float dyeR, float dyeG, float dyeB, float radius) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    k_add_force_2d<<<grid, block>>>(d_vx, d_vy, d_densR, d_densG, d_densB,
                                     W, H, cx, cy, fx, fy, dyeR, dyeG, dyeB, radius);
}

// ============================================================ //
//                         3D KERNELS                           //
// ============================================================ //

// ---- advect 3D (trilinear interpolation) ---- //
__global__ void k_advect_3d(float *out, const float *field,
                            const float *vx, const float *vy, const float *vz,
                            int W, int H, int D, float dt, float dissipation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1 || k < 1 || k >= D-1) return;

    int slab = W * H;
    int idx = k * slab + j * W + i;

    //backtrace
    float px = (float)i - dt * vx[idx];
    float py = (float)j - dt * vy[idx];
    float pz = (float)k - dt * vz[idx];

    //clamp
    px = fmaxf(0.5f, fminf((float)(W-1) - 0.5f, px));
    py = fmaxf(0.5f, fminf((float)(H-1) - 0.5f, py));
    pz = fmaxf(0.5f, fminf((float)(D-1) - 0.5f, pz));

    int x0 = (int)px, y0 = (int)py, z0 = (int)pz;
    int x1 = min(x0+1, W-1), y1 = min(y0+1, H-1), z1 = min(z0+1, D-1);
    float sx = px - x0, sy = py - y0, sz = pz - z0;

    //trilinear: sample 8 corners
    float c000 = field[z0*slab + y0*W + x0];
    float c100 = field[z0*slab + y0*W + x1];
    float c010 = field[z0*slab + y1*W + x0];
    float c110 = field[z0*slab + y1*W + x1];
    float c001 = field[z1*slab + y0*W + x0];
    float c101 = field[z1*slab + y0*W + x1];
    float c011 = field[z1*slab + y1*W + x0];
    float c111 = field[z1*slab + y1*W + x1];

    float c00 = c000*(1-sx) + c100*sx;
    float c10 = c010*(1-sx) + c110*sx;
    float c01 = c001*(1-sx) + c101*sx;
    float c11 = c011*(1-sx) + c111*sx;
    float c0  = c00*(1-sy)  + c10*sy;
    float c1  = c01*(1-sy)  + c11*sy;

    out[idx] = dissipation * (c0*(1-sz) + c1*sz);
}

void cuda_advect_3d(float *d_out, const float *d_field,
                    const float *d_vx, const float *d_vy, const float *d_vz,
                    int W, int H, int D, float dt, float dissipation) {
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    k_advect_3d<<<grid, block>>>(d_out, d_field, d_vx, d_vy, d_vz, W, H, D, dt, dissipation);
}

// ---- diffuse 3D (6-neighbor Jacobi) ---- //
__global__ void k_diffuse_3d(float *field, const float *field0, int W, int H, int D, float a, float denom) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1 || k < 1 || k >= D-1) return;

    int slab = W * H;
    int idx = k * slab + j * W + i;
    field[idx] = (field0[idx] + a * (
        field[idx-1] + field[idx+1] +
        field[idx-W] + field[idx+W] +
        field[idx-slab] + field[idx+slab]
    )) / denom;
}

void cuda_diffuse_3d(float *d_field, float *d_field0, int W, int H, int D, float diff, float dt, int iters) {
    float a = diff * dt;
    float denom = 1.0f + 6.0f * a;
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    for (int k = 0; k < iters; k++) {
        k_diffuse_3d<<<grid, block>>>(d_field, d_field0, W, H, D, a, denom);
    }
}

// ---- divergence 3D ---- //
__global__ void k_divergence_3d(float *div, const float *vx, const float *vy, const float *vz,
                                int W, int H, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1 || k < 1 || k >= D-1) return;

    int slab = W * H;
    int idx = k * slab + j * W + i;
    div[idx] = -0.5f * (
        vx[idx+1] - vx[idx-1] +
        vy[idx+W] - vy[idx-W] +
        vz[idx+slab] - vz[idx-slab]
    );
}

void cuda_divergence_3d(float *d_div, const float *d_vx, const float *d_vy, const float *d_vz,
                        int W, int H, int D) {
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    k_divergence_3d<<<grid, block>>>(d_div, d_vx, d_vy, d_vz, W, H, D);
}

// ---- pressure Jacobi 3D (6-neighbor) ---- //
__global__ void k_pressure_jacobi_3d(float *p_out, const float *p_in, const float *div,
                                     int W, int H, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1 || k < 1 || k >= D-1) return;

    int slab = W * H;
    int idx = k * slab + j * W + i;
    p_out[idx] = (div[idx] +
        p_in[idx-1] + p_in[idx+1] +
        p_in[idx-W] + p_in[idx+W] +
        p_in[idx-slab] + p_in[idx+slab]
    ) / 6.0f;
}

void cuda_pressure_jacobi_3d(float *d_p, float *d_p_temp, const float *d_div,
                             int W, int H, int D, int iters) {
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    for (int k = 0; k < iters; k++) {
        k_pressure_jacobi_3d<<<grid, block>>>(d_p_temp, d_p, d_div, W, H, D);
        cuda_swap(&d_p, &d_p_temp);
    }
}

// ---- gradient subtract 3D ---- //
__global__ void k_gradient_subtract_3d(float *vx, float *vy, float *vz, const float *p,
                                       int W, int H, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1 || k < 1 || k >= D-1) return;

    int slab = W * H;
    int idx = k * slab + j * W + i;
    vx[idx] -= 0.5f * (p[idx+1] - p[idx-1]);
    vy[idx] -= 0.5f * (p[idx+W] - p[idx-W]);
    vz[idx] -= 0.5f * (p[idx+slab] - p[idx-slab]);
}

void cuda_gradient_subtract_3d(float *d_vx, float *d_vy, float *d_vz,
                               const float *d_p, int W, int H, int D) {
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    k_gradient_subtract_3d<<<grid, block>>>(d_vx, d_vy, d_vz, d_p, W, H, D);
}

// ---- vorticity 3D (vector curl ω = ∇×u) ---- //
__global__ void k_vorticity_3d(float *wx, float *wy, float *wz,
                               const float *vx, const float *vy, const float *vz,
                               int W, int H, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1 || k < 1 || k >= D-1) return;

    int slab = W * H;
    int idx = k * slab + j * W + i;

    //central differences
    float duzdy = (vz[idx+W]    - vz[idx-W])    * 0.5f;
    float duydz = (vy[idx+slab] - vy[idx-slab])  * 0.5f;
    float duxdz = (vx[idx+slab] - vx[idx-slab])  * 0.5f;
    float duzdx = (vz[idx+1]    - vz[idx-1])     * 0.5f;
    float duydx = (vy[idx+1]    - vy[idx-1])     * 0.5f;
    float duxdy = (vx[idx+W]    - vx[idx-W])     * 0.5f;

    //ω = ∇×u
    wx[idx] = duzdy - duydz;   //∂uz/∂y - ∂uy/∂z
    wy[idx] = duxdz - duzdx;   //∂ux/∂z - ∂uz/∂x
    wz[idx] = duydx - duxdy;   //∂uy/∂x - ∂ux/∂y
}

void cuda_vorticity_3d(float *d_wx, float *d_wy, float *d_wz,
                       const float *d_vx, const float *d_vy, const float *d_vz,
                       int W, int H, int D) {
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    k_vorticity_3d<<<grid, block>>>(d_wx, d_wy, d_wz, d_vx, d_vy, d_vz, W, H, D);
}

// ---- vorticity confinement 3D ---- //
__global__ void k_vorticity_confinement_3d(float *vx, float *vy, float *vz,
                                           const float *wx, const float *wy, const float *wz,
                                           int W, int H, int D, float eps, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 2 || i >= W-2 || j < 2 || j >= H-2 || k < 2 || k >= D-2) return;

    int slab = W * H;
    int idx = k * slab + j * W + i;

    //|ω| at current cell
    float mag = sqrtf(wx[idx]*wx[idx] + wy[idx]*wy[idx] + wz[idx]*wz[idx]);

    //helper: |ω| at neighboring cell
    auto magAt = [&](int ii) -> float {
        return sqrtf(wx[ii]*wx[ii] + wy[ii]*wy[ii] + wz[ii]*wz[ii]);
    };

    //gradient of |ω|
    float dmdx = (magAt(idx+1)    - magAt(idx-1))    * 0.5f;
    float dmdy = (magAt(idx+W)    - magAt(idx-W))    * 0.5f;
    float dmdz = (magAt(idx+slab) - magAt(idx-slab)) * 0.5f;

    float len = sqrtf(dmdx*dmdx + dmdy*dmdy + dmdz*dmdz) + 1e-5f;
    float nx = dmdx / len;
    float ny = dmdy / len;
    float nz = dmdz / len;

    //N × ω (cross product)
    float fx = ny * wz[idx] - nz * wy[idx];
    float fy = nz * wx[idx] - nx * wz[idx];
    float fz2 = nx * wy[idx] - ny * wx[idx];

    vx[idx] += eps * dt * fx;
    vy[idx] += eps * dt * fy;
    vz[idx] += eps * dt * fz2;
}

void cuda_vorticity_confinement_3d(float *d_vx, float *d_vy, float *d_vz,
                                   const float *d_wx, const float *d_wy, const float *d_wz,
                                   int W, int H, int D, float epsilon, float dt) {
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    k_vorticity_confinement_3d<<<grid, block>>>(d_vx, d_vy, d_vz, d_wx, d_wy, d_wz,
                                                 W, H, D, epsilon, dt);
}

// ---- add force 3D ---- //
__global__ void k_add_force_3d(float *vx, float *vy, float *vz,
                               float *densR, float *densG, float *densB,
                               int W, int H, int D,
                               float cx, float cy, float cz,
                               float fx, float fy, float fz,
                               float dyeR, float dyeG, float dyeB, float radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || i >= W-1 || j < 1 || j >= H-1 || k < 1 || k >= D-1) return;

    float dx = (float)i - cx;
    float dy = (float)j - cy;
    float dz = (float)k - cz;
    float d2 = dx*dx + dy*dy + dz*dz;
    float r2 = radius * radius;

    if (d2 < r2) {
        float w = expf(-d2 / (2.0f * r2));
        int slab = W * H;
        int idx = k * slab + j * W + i;
        vx[idx] += fx * w;
        vy[idx] += fy * w;
        vz[idx] += fz * w;
        densR[idx] += dyeR * w;
        densG[idx] += dyeG * w;
        densB[idx] += dyeB * w;
    }
}

void cuda_add_force_3d(float *d_vx, float *d_vy, float *d_vz,
                       float *d_densR, float *d_densG, float *d_densB,
                       int W, int H, int D,
                       float cx, float cy, float cz,
                       float fx, float fy, float fz,
                       float dyeR, float dyeG, float dyeB, float radius) {
    dim3 block(8, 8, 4);
    dim3 grid((W+7)/8, (H+7)/8, (D+3)/4);
    k_add_force_3d<<<grid, block>>>(d_vx, d_vy, d_vz, d_densR, d_densG, d_densB,
                                     W, H, D, cx, cy, cz, fx, fy, fz, dyeR, dyeG, dyeB, radius);
}

#endif // USE_CUDA
