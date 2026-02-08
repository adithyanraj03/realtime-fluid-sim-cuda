#ifndef _GRID_HPP_
#define _GRID_HPP_

#include <vector>
#include <cstring>
#include <iostream>

using namespace std;

//2D/3D fluid grid ;stores velocity, density, pressure fields on a collocated grid
//all fields are flat arrays:
//  2D: index = row * width + col
//  3D: index = z * width * height + y * width + x
//supports optional GPU mirror for CUDA acceleration
class Grid {
public:
    int                 width;          //grid cells in x
    int                 height;         //grid cells in y
    int                 depth;          //grid cells in z (1 for 2D mode)
    int                 size;           //width * height * depth ;total cells
    bool                is3D;           //true if depth > 1

    //velocity field (m/s)
    float*              vx;
    float*              vy;
    float*              vz;             //z-component (3D only, nullptr in 2D)
    float*              vx0;            //swap buffers
    float*              vy0;
    float*              vz0;

    //dye density (RGB)
    float*              densityR;
    float*              densityG;
    float*              densityB;
    float*              densityR0;
    float*              densityG0;
    float*              densityB0;

    //pressure + divergence
    float*              pressure;
    float*              divergence;     //∇·u

    //vorticity
    //2D: scalar ω = ∂vy/∂x - ∂vx/∂y
    //3D: vector ω = ∇×u  (three components)
    float*              vorticity;      //2D scalar OR 3D ωx
    float*              vorticityY;     //3D only: ωy
    float*              vorticityZ;     //3D only: ωz

    //GPU mirrors
    float*              d_vx;
    float*              d_vy;
    float*              d_vz;
    float*              d_vx0;
    float*              d_vy0;
    float*              d_vz0;
    float*              d_densityR;
    float*              d_densityG;
    float*              d_densityB;
    float*              d_densityR0;
    float*              d_densityG0;
    float*              d_densityB0;
    float*              d_pressure;
    float*              d_divergence;
    float*              d_vorticity;
    float*              d_vorticityY;
    float*              d_vorticityZ;
    bool                gpuAllocated;

    Grid(int w, int h, int d = 1);
    ~Grid();

    //index helpers
    inline int idx(int row, int col) const { return row * this->width + col; }
    inline int idx3(int x, int y, int z) const {
        return z * this->width * this->height + y * this->width + x;
    }

    void swapVelocity();
    void swapDensity();
    void clear();

    //GPU memory management
    void allocGpu();
    void freeGpu();
    void toGpu();
    void toHost();
    void velocityToGpu();
    void velocityToHost();
    void densityToGpu();
    void densityToHost();
    void pressureToGpu();
    void pressureToHost();
};

#endif
