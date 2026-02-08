#ifndef _FLUID_SOLVER_HPP_
#define _FLUID_SOLVER_HPP_

#include "Grid.hpp"
#include <cmath>
#include <string>

using namespace std;

//simulation configuration ;loaded from JSON
struct SimConfig {
    int         gridWidth;
    int         gridHeight;
    int         gridDepth;              //1 = 2D, >1 = 3D
    int         windowWidth;
    int         windowHeight;
    float       viscosity;              //kinematic viscosity ν (m²/s)
    float       diffusion;              //dye diffusion coefficient
    float       dt;                     //timestep (seconds)
    int         jacobiIterations;       //pressure solver iterations
    float       vorticityStrength;      //ε for vorticity confinement
    float       dyeRadius;              //brush radius for dye injection
    float       forceStrength;          //mouse force multiplier
    float       dyeDissipation;         //dye fade per step [0..1]
    float       velocityDissipation;    //velocity damping per step [0..1]
    bool        is3D;                   //true when gridDepth > 1
};

//2D/3D incompressible Navier-Stokes solver ;operator splitting approach
//
//governing equations:
//  ∂u/∂t + (u·∇)u = -1/ρ ∇p + ν∇²u + f    (momentum)
//  ∇·u = 0                                   (incompressibility)
//
//solver steps (Stam's Stable Fluids):
//  1. vorticity confinement   amplify existing vortices
//  2. diffuse velocity        ∂u/∂t = ν∇²u             (implicit Jacobi)
//  3. project                 ∇·u = 0 via ∇²p = ∇·u
//  4. advect velocity         semi-Lagrangian backtrace
//  5. project again
//  6. advect + diffuse density
class FluidSolver {
public:
    Grid*           grid;
    SimConfig       config;
    bool            useGpu;

    FluidSolver(Grid *grid, const SimConfig &config);
    ~FluidSolver();

    //main simulation step ;advances one dt
    void step();

    //add impulse ;force + dye injection
    //2D: (x,y) grid coords
    //3D: (x,y,z) grid coords
    void addForce(float x, float y, float z, float fx, float fy, float fz,
                  float dyeR, float dyeG, float dyeB, float radius);

    static SimConfig loadConfig(const string &path);

private:
    // ---- 2D CPU solvers ---- //
    void advect2D(float *field, float *field0, float *vx, float *vy, float dissipation);
    void diffuse2D(float *field, float *field0, float diff);
    void project2D();
    void vorticityConfinement2D();
    void addForce2D(float x, float y, float fx, float fy,
                    float dyeR, float dyeG, float dyeB, float radius);

    // ---- 3D CPU solvers ---- //
    void advect3D(float *field, float *field0, float *vx, float *vy, float *vz, float dissipation);
    void diffuse3D(float *field, float *field0, float diff);
    void project3D();
    void vorticityConfinement3D();
    void addForce3D(float x, float y, float z, float fx, float fy, float fz,
                    float dyeR, float dyeG, float dyeB, float radius);

    //interpolation
    float bilinearSample(float *field, float x, float y);
    float trilinearSample(float *field, float x, float y, float z);
};

#endif
