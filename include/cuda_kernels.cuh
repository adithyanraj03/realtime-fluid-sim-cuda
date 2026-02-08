#ifndef _CUDA_KERNELS_CUH_
#define _CUDA_KERNELS_CUH_

#ifdef USE_CUDA

//device init
int cuda_init();
bool cuda_is_available();
void cuda_shutdown();
void cuda_print_stats();

// ======================== 2D kernels ======================== //

void cuda_advect_2d(float *d_out, const float *d_field, const float *d_vx, const float *d_vy,
                    int W, int H, float dt, float dissipation);

void cuda_diffuse_2d(float *d_field, float *d_field0,
                     int W, int H, float diff, float dt, int iters);

void cuda_divergence_2d(float *d_div, const float *d_vx, const float *d_vy, int W, int H);

void cuda_pressure_jacobi_2d(float *d_p, float *d_p_temp, const float *d_div,
                             int W, int H, int iters);

void cuda_gradient_subtract_2d(float *d_vx, float *d_vy, const float *d_p, int W, int H);

void cuda_vorticity_2d(float *d_vort, const float *d_vx, const float *d_vy, int W, int H);

void cuda_vorticity_confinement_2d(float *d_vx, float *d_vy, const float *d_vort,
                                   int W, int H, float epsilon, float dt);

void cuda_add_force_2d(float *d_vx, float *d_vy,
                       float *d_densR, float *d_densG, float *d_densB,
                       int W, int H,
                       float cx, float cy, float fx, float fy,
                       float dyeR, float dyeG, float dyeB, float radius);

// ======================== 3D kernels ======================== //

void cuda_advect_3d(float *d_out, const float *d_field,
                    const float *d_vx, const float *d_vy, const float *d_vz,
                    int W, int H, int D, float dt, float dissipation);

void cuda_diffuse_3d(float *d_field, float *d_field0,
                     int W, int H, int D, float diff, float dt, int iters);

void cuda_divergence_3d(float *d_div,
                        const float *d_vx, const float *d_vy, const float *d_vz,
                        int W, int H, int D);

void cuda_pressure_jacobi_3d(float *d_p, float *d_p_temp, const float *d_div,
                             int W, int H, int D, int iters);

void cuda_gradient_subtract_3d(float *d_vx, float *d_vy, float *d_vz,
                               const float *d_p, int W, int H, int D);

void cuda_vorticity_3d(float *d_wx, float *d_wy, float *d_wz,
                       const float *d_vx, const float *d_vy, const float *d_vz,
                       int W, int H, int D);

void cuda_vorticity_confinement_3d(float *d_vx, float *d_vy, float *d_vz,
                                   const float *d_wx, const float *d_wy, const float *d_wz,
                                   int W, int H, int D, float epsilon, float dt);

void cuda_add_force_3d(float *d_vx, float *d_vy, float *d_vz,
                       float *d_densR, float *d_densG, float *d_densB,
                       int W, int H, int D,
                       float cx, float cy, float cz,
                       float fx, float fy, float fz,
                       float dyeR, float dyeG, float dyeB, float radius);

// ======================== common ======================== //

void cuda_fill(float *d_field, int size, float value);
void cuda_swap(float **a, float **b);
void cuda_alloc(float **d_ptr, int count);
void cuda_free(float *d_ptr);
void cuda_copy_to_device(float *d_dst, const float *h_src, int count);
void cuda_copy_to_host(float *h_dst, const float *d_src, int count);

#endif
#endif
