#include "Grid.hpp"
#include <cstdlib>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

//constructor ;allocates all flat field arrays for 2D or 3D grid
Grid::Grid(int w, int h, int d)
    : width(w), height(h), depth(d),
      vz(nullptr), vz0(nullptr),
      vorticityY(nullptr), vorticityZ(nullptr),
      d_vx(nullptr), d_vy(nullptr), d_vz(nullptr),
      d_vx0(nullptr), d_vy0(nullptr), d_vz0(nullptr),
      d_densityR(nullptr), d_densityG(nullptr), d_densityB(nullptr),
      d_densityR0(nullptr), d_densityG0(nullptr), d_densityB0(nullptr),
      d_pressure(nullptr), d_divergence(nullptr),
      d_vorticity(nullptr), d_vorticityY(nullptr), d_vorticityZ(nullptr),
      gpuAllocated(false) {

    this->is3D = (d > 1);
    this->size = w * h * d;

    //velocity
    this->vx  = new float[this->size]();
    this->vy  = new float[this->size]();
    this->vx0 = new float[this->size]();
    this->vy0 = new float[this->size]();

    //z-velocity (3D only)
    if (this->is3D) {
        this->vz  = new float[this->size]();
        this->vz0 = new float[this->size]();
    }

    //dye density RGB
    this->densityR  = new float[this->size]();
    this->densityG  = new float[this->size]();
    this->densityB  = new float[this->size]();
    this->densityR0 = new float[this->size]();
    this->densityG0 = new float[this->size]();
    this->densityB0 = new float[this->size]();

    //pressure and divergence
    this->pressure   = new float[this->size]();
    this->divergence = new float[this->size]();

    //vorticity ;scalar for 2D, vector for 3D
    this->vorticity = new float[this->size]();
    if (this->is3D) {
        this->vorticityY = new float[this->size]();
        this->vorticityZ = new float[this->size]();
    }

    cout << "[Grid] Allocated " << (this->is3D ? "3D" : "2D") << " grid: "
         << w << "x" << h;
    if (this->is3D) cout << "x" << d;
    cout << " (" << this->size << " cells)" << endl;
}

//destructor ;releases CPU and GPU memory
Grid::~Grid() {
    this->freeGpu();

    delete[] this->vx;
    delete[] this->vy;
    delete[] this->vx0;
    delete[] this->vy0;

    if (this->vz)  { delete[] this->vz;  delete[] this->vz0; }

    delete[] this->densityR;
    delete[] this->densityG;
    delete[] this->densityB;
    delete[] this->densityR0;
    delete[] this->densityG0;
    delete[] this->densityB0;

    delete[] this->pressure;
    delete[] this->divergence;
    delete[] this->vorticity;

    if (this->vorticityY) delete[] this->vorticityY;
    if (this->vorticityZ) delete[] this->vorticityZ;

    cout << "[Grid] Freed all memory" << endl;
}

//swap velocity for ping-pong
void Grid::swapVelocity() {
    swap(this->vx, this->vx0);
    swap(this->vy, this->vy0);
    if (this->is3D) swap(this->vz, this->vz0);
}

//swap density for ping-pong
void Grid::swapDensity() {
    swap(this->densityR, this->densityR0);
    swap(this->densityG, this->densityG0);
    swap(this->densityB, this->densityB0);
}

//zero all fields
void Grid::clear() {
    memset(this->vx, 0, this->size * sizeof(float));
    memset(this->vy, 0, this->size * sizeof(float));
    memset(this->vx0, 0, this->size * sizeof(float));
    memset(this->vy0, 0, this->size * sizeof(float));

    if (this->is3D) {
        memset(this->vz, 0, this->size * sizeof(float));
        memset(this->vz0, 0, this->size * sizeof(float));
    }

    memset(this->densityR, 0, this->size * sizeof(float));
    memset(this->densityG, 0, this->size * sizeof(float));
    memset(this->densityB, 0, this->size * sizeof(float));
    memset(this->densityR0, 0, this->size * sizeof(float));
    memset(this->densityG0, 0, this->size * sizeof(float));
    memset(this->densityB0, 0, this->size * sizeof(float));

    memset(this->pressure, 0, this->size * sizeof(float));
    memset(this->divergence, 0, this->size * sizeof(float));
    memset(this->vorticity, 0, this->size * sizeof(float));

    if (this->is3D) {
        memset(this->vorticityY, 0, this->size * sizeof(float));
        memset(this->vorticityZ, 0, this->size * sizeof(float));
    }
}

// ======================== GPU memory ======================== //

void Grid::allocGpu() {
#ifdef USE_CUDA
    if (this->gpuAllocated) return;

    int n = this->size;
    cuda_alloc(&this->d_vx, n);
    cuda_alloc(&this->d_vy, n);
    cuda_alloc(&this->d_vx0, n);
    cuda_alloc(&this->d_vy0, n);

    if (this->is3D) {
        cuda_alloc(&this->d_vz, n);
        cuda_alloc(&this->d_vz0, n);
    }

    cuda_alloc(&this->d_densityR, n);
    cuda_alloc(&this->d_densityG, n);
    cuda_alloc(&this->d_densityB, n);
    cuda_alloc(&this->d_densityR0, n);
    cuda_alloc(&this->d_densityG0, n);
    cuda_alloc(&this->d_densityB0, n);

    cuda_alloc(&this->d_pressure, n);
    cuda_alloc(&this->d_divergence, n);
    cuda_alloc(&this->d_vorticity, n);

    if (this->is3D) {
        cuda_alloc(&this->d_vorticityY, n);
        cuda_alloc(&this->d_vorticityZ, n);
    }

    this->gpuAllocated = true;

    int mb = (int)(n * sizeof(float) * (this->is3D ? 20 : 15) / (1024 * 1024));
    cout << "[Grid] GPU allocated ~" << mb << " MB" << endl;
#endif
}

void Grid::freeGpu() {
#ifdef USE_CUDA
    if (!this->gpuAllocated) return;

    cuda_free(this->d_vx);    cuda_free(this->d_vy);
    cuda_free(this->d_vx0);   cuda_free(this->d_vy0);

    if (this->d_vz)  { cuda_free(this->d_vz);  cuda_free(this->d_vz0); }

    cuda_free(this->d_densityR);  cuda_free(this->d_densityG);  cuda_free(this->d_densityB);
    cuda_free(this->d_densityR0); cuda_free(this->d_densityG0); cuda_free(this->d_densityB0);

    cuda_free(this->d_pressure);  cuda_free(this->d_divergence);
    cuda_free(this->d_vorticity);

    if (this->d_vorticityY) cuda_free(this->d_vorticityY);
    if (this->d_vorticityZ) cuda_free(this->d_vorticityZ);

    this->gpuAllocated = false;
    cout << "[Grid] GPU memory freed" << endl;
#endif
}

void Grid::toGpu() {
    this->velocityToGpu();
    this->densityToGpu();
    this->pressureToGpu();
}

void Grid::toHost() {
    this->velocityToHost();
    this->densityToHost();
    this->pressureToHost();
}

void Grid::velocityToGpu() {
#ifdef USE_CUDA
    int n = this->size;
    cuda_copy_to_device(this->d_vx, this->vx, n);
    cuda_copy_to_device(this->d_vy, this->vy, n);
    cuda_copy_to_device(this->d_vx0, this->vx0, n);
    cuda_copy_to_device(this->d_vy0, this->vy0, n);
    if (this->is3D) {
        cuda_copy_to_device(this->d_vz, this->vz, n);
        cuda_copy_to_device(this->d_vz0, this->vz0, n);
    }
#endif
}

void Grid::velocityToHost() {
#ifdef USE_CUDA
    int n = this->size;
    cuda_copy_to_host(this->vx, this->d_vx, n);
    cuda_copy_to_host(this->vy, this->d_vy, n);
    if (this->is3D) {
        cuda_copy_to_host(this->vz, this->d_vz, n);
    }
#endif
}

void Grid::densityToGpu() {
#ifdef USE_CUDA
    int n = this->size;
    cuda_copy_to_device(this->d_densityR, this->densityR, n);
    cuda_copy_to_device(this->d_densityG, this->densityG, n);
    cuda_copy_to_device(this->d_densityB, this->densityB, n);
    cuda_copy_to_device(this->d_densityR0, this->densityR0, n);
    cuda_copy_to_device(this->d_densityG0, this->densityG0, n);
    cuda_copy_to_device(this->d_densityB0, this->densityB0, n);
#endif
}

void Grid::densityToHost() {
#ifdef USE_CUDA
    int n = this->size;
    cuda_copy_to_host(this->densityR, this->d_densityR, n);
    cuda_copy_to_host(this->densityG, this->d_densityG, n);
    cuda_copy_to_host(this->densityB, this->d_densityB, n);
#endif
}

void Grid::pressureToGpu() {
#ifdef USE_CUDA
    int n = this->size;
    cuda_copy_to_device(this->d_pressure, this->pressure, n);
    cuda_copy_to_device(this->d_divergence, this->divergence, n);
#endif
}

void Grid::pressureToHost() {
#ifdef USE_CUDA
    int n = this->size;
    cuda_copy_to_host(this->pressure, this->d_pressure, n);
#endif
}
