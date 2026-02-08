#ifndef _SIMULATION_HPP_
#define _SIMULATION_HPP_

#include "Grid.hpp"
#include "FluidSolver.hpp"
#include "Renderer.hpp"

struct GLFWwindow;

//main simulation loop ;handles window, input, timing, orchestration
class Simulation {
public:
    Grid*           grid;
    FluidSolver*    solver;
    Renderer*       renderer;
    GLFWwindow*     window;
    SimConfig       config;

    //mouse state
    double          mouseX, mouseY;
    double          prevMouseX, prevMouseY;
    bool            mouseDown;          //left button
    bool            rightMouseDown;     //right button
    float           dyeHue;            //rotating rainbow hue

    //simulation state
    bool            paused;
    bool            running;
    int             frameCount;
    double          lastFpsTime;
    int             fpsCount;

    //demo mode state
    bool            demoMode;           //auto-inject patterns, no mouse interaction
    double          demoTimeLimit;      //seconds before auto-exit (0 = no limit)
    double          demoStartTime;      //glfwGetTime() at start

    Simulation(const SimConfig &config, bool demo = false, double demoLimit = 0.0);
    ~Simulation();

    void run();

private:
    void processInput();
    void demoInject();              //auto-inject scripted force/dye patterns

    //GLFW callbacks (static, forwarded to instance)
    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);

    void hsvToRgb(float h, float s, float v, float &r, float &g, float &b);
};

#endif
