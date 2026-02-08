#include <iostream>
#include <string>
#include "Simulation.hpp"
#include "FluidSolver.hpp"

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

using namespace std;

int main(int argc, char *argv[]) {
    cout << "========================================" << endl;
    cout << "  Navier-Stokes Fluid Simulator" << endl;
    cout << "  2D / 3D Real-Time Simulation" << endl;
    cout << "========================================" << endl;

    //parse CLI args
    bool force3D = false;
    bool demoMode = false;
    string configPath = "config/sim.json";

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--3d")      force3D = true;
        if (arg == "--sim")     demoMode = true;
        if (arg == "--config" && i + 1 < argc) configPath = argv[++i];
    }

    //load config
    SimConfig config = FluidSolver::loadConfig(configPath);

    //CLI override: --3d forces 3D mode with defaults
    if (force3D && !config.is3D) {
        config.gridWidth  = 128;
        config.gridHeight = 128;
        config.gridDepth  = 128;
        config.is3D = true;
        config.dyeRadius    = 8.0f;
        config.forceStrength = 800.0f;
        cout << "[Main] --3d flag: forcing 3D mode (" << config.gridWidth
             << "x" << config.gridHeight << "x" << config.gridDepth << ")" << endl;
    }

    //init CUDA
#ifdef USE_CUDA
    if (cuda_init() == 0) {
        cuda_print_stats();
    }
#endif

    if (demoMode) {
        // ---- demo mode: 2D phase then 3D phase ---- //
        cout << "\n[Demo] === DEMO MODE ===" << endl;
        cout << "[Demo] Phase 1: 2D Fluid (15 seconds)" << endl;
        cout << "[Demo] Phase 2: 3D Volume (until ESC)" << endl;
        cout << "[Demo] Press ESC to skip / quit at any time\n" << endl;

        // -- Phase 1: 2D -- //
        SimConfig cfg2d = config;
        cfg2d.gridWidth  = 512;
        cfg2d.gridHeight = 512;
        cfg2d.gridDepth  = 1;
        cfg2d.is3D       = false;
        cfg2d.dyeRadius  = 15.0f;
        cfg2d.forceStrength = 5000.0f;

        {
            Simulation sim2d(cfg2d, true, 15.0);
            sim2d.run();
        }

        // -- Phase 2: 3D -- //
        cout << "\n[Demo] Transitioning to 3D..." << endl;

        SimConfig cfg3d = config;
        cfg3d.gridWidth  = 128;
        cfg3d.gridHeight = 128;
        cfg3d.gridDepth  = 128;
        cfg3d.is3D       = true;
        cfg3d.dyeRadius  = 8.0f;
        cfg3d.forceStrength = 800.0f;

        {
            Simulation sim3d(cfg3d, true, 0.0);
            sim3d.run();
        }
    } else {
        //normal interactive mode
        Simulation sim(config);
        sim.run();
    }

    //cleanup
#ifdef USE_CUDA
    cuda_shutdown();
#endif

    cout << "[Main] Goodbye!" << endl;
    return 0;
}
