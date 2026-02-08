#include "Simulation.hpp"
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <sstream>

using namespace std;

Simulation::Simulation(const SimConfig &config, bool demo, double demoLimit) : config(config) {
    this->mouseX = 0;
    this->mouseY = 0;
    this->prevMouseX = 0;
    this->prevMouseY = 0;
    this->mouseDown = false;
    this->rightMouseDown = false;
    this->dyeHue = 0.0f;
    this->paused = false;
    this->running = true;
    this->frameCount = 0;
    this->lastFpsTime = 0;
    this->fpsCount = 0;
    this->demoMode = demo;
    this->demoTimeLimit = demoLimit;
    this->demoStartTime = 0;

    //init GLFW
    if (!glfwInit()) {
        cerr << "[GLFW] Init failed!" << endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    string title = config.is3D ? "Navier-Stokes 3D Fluid Simulator" : "Navier-Stokes 2D Fluid Simulator";
    if (this->demoMode) title += " [DEMO]";
    this->window = glfwCreateWindow(config.windowWidth, config.windowHeight, title.c_str(), nullptr, nullptr);
    if (!this->window) {
        cerr << "[GLFW] Window creation failed!" << endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(this->window);
    glfwSwapInterval(1);  //vsync

    //load OpenGL via GLAD
    int version = gladLoadGL(glfwGetProcAddress);
    if (!version) {
        cerr << "[GLAD] Failed to load OpenGL!" << endl;
        return;
    }
    cout << "[OpenGL] Version: " << glGetString(GL_VERSION) << endl;

    glViewport(0, 0, config.windowWidth, config.windowHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    //set callbacks
    glfwSetWindowUserPointer(this->window, this);
    glfwSetKeyCallback(this->window, Simulation::keyCallback);
    glfwSetMouseButtonCallback(this->window, Simulation::mouseButtonCallback);
    glfwSetScrollCallback(this->window, Simulation::scrollCallback);

    //create grid + solver + renderer
    this->grid = new Grid(config.gridWidth, config.gridHeight, config.gridDepth);
    this->solver = new FluidSolver(this->grid, config);
    this->renderer = new Renderer(config.windowWidth, config.windowHeight,
                                   config.gridWidth, config.gridHeight, config.gridDepth);
    this->renderer->init();

    this->lastFpsTime = glfwGetTime();
    this->demoStartTime = glfwGetTime();

    cout << "[Sim] Ready (" << (config.is3D ? "3D" : "2D") << " mode";
    if (this->demoMode) {
        cout << ", DEMO";
        if (this->demoTimeLimit > 0) cout << " " << (int)this->demoTimeLimit << "s";
    }
    cout << ")" << endl;

    if (!this->demoMode) {
        cout << "[Sim] Controls:" << endl;
        cout << "  Left-click + drag : inject dye + force" << endl;
        if (config.is3D) {
            cout << "  Right-click + drag: orbit camera" << endl;
        }
        cout << "  Scroll            : " << (config.is3D ? "zoom" : "brush radius") << endl;
        cout << "  SPACE             : pause/resume" << endl;
        cout << "  R                 : reset fluid" << endl;
        cout << "  V                 : cycle visualization" << endl;
        cout << "  ESC               : quit" << endl;
    } else {
        cout << "[Sim] Controls: ESC = skip/quit";
        if (config.is3D) cout << ", Right-click = orbit, Scroll = zoom";
        cout << endl;
    }
}

Simulation::~Simulation() {
    delete this->renderer;
    delete this->solver;
    delete this->grid;
    if (this->window) {
        glfwDestroyWindow(this->window);
    }
    glfwTerminate();
    cout << "[Sim] Shutdown complete" << endl;
}

// ======================== main loop ======================== //

void Simulation::run() {
    while (!glfwWindowShouldClose(this->window) && this->running) {
        glfwPollEvents();
        this->processInput();

        //demo mode: auto-inject patterns + check time limit
        if (this->demoMode) {
            this->demoInject();

            if (this->demoTimeLimit > 0) {
                double elapsed = glfwGetTime() - this->demoStartTime;
                if (elapsed >= this->demoTimeLimit) {
                    cout << "[Demo] Phase complete (" << (int)elapsed << "s)" << endl;
                    break;
                }
            }

            //slow camera auto-orbit in 3D demo
            if (this->config.is3D) {
                this->renderer->rotateCamera(0.003f, 0.0f);
            }
        }

        if (!this->paused) {
            this->solver->step();
        }

        this->renderer->render(this->grid);
        glfwSwapBuffers(this->window);

        //FPS counter
        this->frameCount++;
        this->fpsCount++;
        double now = glfwGetTime();
        if (now - this->lastFpsTime >= 1.0) {
            double fps = this->fpsCount / (now - this->lastFpsTime);
            ostringstream ss;
            ss << (this->config.is3D ? "NS 3D" : "NS 2D");
            if (this->demoMode) ss << " DEMO";
            ss << " | " << this->config.gridWidth << "x" << this->config.gridHeight;
            if (this->config.is3D) ss << "x" << this->config.gridDepth;
            ss << " | " << this->renderer->getVisModeName();
            ss << " | " << (int)fps << " FPS";
            if (this->demoMode && this->demoTimeLimit > 0) {
                int remaining = (int)(this->demoTimeLimit - (glfwGetTime() - this->demoStartTime));
                if (remaining > 0) ss << " | " << remaining << "s left";
            }
            if (this->paused) ss << " [PAUSED]";
            glfwSetWindowTitle(this->window, ss.str().c_str());
            this->lastFpsTime = now;
            this->fpsCount = 0;
        }
    }
}

// ======================== input processing ======================== //

void Simulation::processInput() {
    double mx, my;
    glfwGetCursorPos(this->window, &mx, &my);

    //camera orbit (3D, right-click drag)
    if (this->config.is3D && this->rightMouseDown) {
        double dx = mx - this->prevMouseX;
        double dy = my - this->prevMouseY;
        this->renderer->rotateCamera((float)dx * 0.005f, (float)-dy * 0.005f);
    }

    //dye injection (left-click drag) — disabled in demo mode
    if (this->mouseDown && !this->demoMode) {
        double dx = mx - this->prevMouseX;
        double dy = my - this->prevMouseY;

        //map mouse coords to grid coords
        float gx, gy, gz;

        if (!this->config.is3D) {
            //2D: direct mapping from window to grid
            gx = (float)mx / (float)this->config.windowWidth * (float)this->config.gridWidth;
            gy = (float)(this->config.windowHeight - my) / (float)this->config.windowHeight * (float)this->config.gridHeight;
            gz = 0;
        } else {
            //3D: inject at center of volume, mouse controls x/y
            gx = (float)mx / (float)this->config.windowWidth * (float)this->config.gridWidth;
            gy = (float)(this->config.windowHeight - my) / (float)this->config.windowHeight * (float)this->config.gridHeight;
            gz = (float)this->config.gridDepth * 0.5f;  //inject at mid-depth
        }

        //force from mouse velocity
        float fx = (float)dx * this->config.forceStrength;
        float fy = (float)-dy * this->config.forceStrength;
        float fz = 0.0f;

        //3D: generate z-force from mouse speed to create volumetric swirling
        //oscillates with time so dye spreads across depth, not flat sheets
        if (this->config.is3D) {
            float speed = sqrtf(fx * fx + fy * fy);
            fz = sinf((float)this->frameCount * 0.15f) * speed * 0.6f;
        }

        //rainbow dye (rotating hue)
        float r, g, b;
        this->hsvToRgb(this->dyeHue, 1.0f, 1.0f, r, g, b);
        this->dyeHue += 0.003f;
        if (this->dyeHue > 1.0f) this->dyeHue -= 1.0f;

        this->solver->addForce(gx, gy, gz, fx, fy, fz,
                               r * 0.8f, g * 0.8f, b * 0.8f,
                               this->config.dyeRadius);
    }

    this->prevMouseX = mx;
    this->prevMouseY = my;
}

// ======================== demo auto-injection ======================== //

void Simulation::demoInject() {
    double t = glfwGetTime() - this->demoStartTime;
    int W = this->config.gridWidth;
    int H = this->config.gridHeight;
    int D = this->config.gridDepth;
    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;
    float cz = (float)D * 0.5f;
    float radius = this->config.dyeRadius;
    float strength = this->config.forceStrength * 0.012f;  //scaled for per-frame

    //rotating hue for rainbow dye
    float r, g, b;
    this->hsvToRgb(fmodf((float)t * 0.08f, 1.0f), 1.0f, 1.0f, r, g, b);

    float r2, g2, b2;
    this->hsvToRgb(fmodf((float)t * 0.08f + 0.5f, 1.0f), 1.0f, 1.0f, r2, g2, b2);

    float r3, g3, b3;
    this->hsvToRgb(fmodf((float)t * 0.08f + 0.33f, 1.0f), 1.0f, 1.0f, r3, g3, b3);

    if (!this->config.is3D) {
        // ---- 2D demo patterns ---- //

        //pattern A: opposing corner jets (0-5s)
        //two diagonal jets that collide in the center, creating turbulence
        if (t < 5.0) {
            float s = (float)t / 5.0f;
            float jet = strength * (1.0f + s * 2.0f);

            //bottom-left → top-right
            this->solver->addForce(W * 0.15f, H * 0.15f, 0,
                                   jet, jet, 0,
                                   r * 0.6f, g * 0.6f, b * 0.6f, radius * 1.2f);

            //top-right → bottom-left
            this->solver->addForce(W * 0.85f, H * 0.85f, 0,
                                   -jet, -jet, 0,
                                   r2 * 0.6f, g2 * 0.6f, b2 * 0.6f, radius * 1.2f);
        }

        //pattern B: rotating emitter ring (3-10s)
        //multiple emitters orbit the center, pointing inward → vortex
        if (t >= 3.0 && t < 10.0) {
            int nJets = 4;
            for (int i = 0; i < nJets; i++) {
                float angle = (float)t * 0.8f + (float)i * 6.2832f / (float)nJets;
                float orbitR = (float)W * 0.32f;
                float px = cx + cosf(angle) * orbitR;
                float py = cy + sinf(angle) * orbitR;

                //tangential force (perpendicular to radius → creates spin)
                float tx = -sinf(angle) * strength * 2.0f;
                float ty =  cosf(angle) * strength * 2.0f;

                float ri, gi, bi;
                this->hsvToRgb(fmodf((float)i / (float)nJets + (float)t * 0.05f, 1.0f), 1.0f, 1.0f, ri, gi, bi);

                this->solver->addForce(px, py, 0,
                                       tx, ty, 0,
                                       ri * 0.5f, gi * 0.5f, bi * 0.5f, radius);
            }
        }

        //pattern C: von Kármán vortex street (8-15s)
        //steady horizontal flow past an obstacle-like region
        if (t >= 8.0) {
            float phase = (float)t * 2.5f;

            //left-side inflow (steady horizontal jet with slight oscillation)
            for (int j = 0; j < 3; j++) {
                float yy = H * (0.35f + 0.15f * (float)j);
                float wobble = sinf(phase + (float)j * 1.5f) * strength * 0.5f;

                this->solver->addForce(W * 0.05f, yy, 0,
                                       strength * 3.0f, wobble, 0,
                                       r3 * 0.4f, g3 * 0.4f, b3 * 0.4f, radius * 0.8f);
            }

            //periodic vortex pair shedding from center
            if (fmodf((float)t, 0.4f) < 0.05f) {
                float shed_y = cy + sinf(phase * 0.7f) * H * 0.1f;
                this->solver->addForce(cx, shed_y, 0,
                                       0, strength * 4.0f * sinf(phase), 0,
                                       r * 0.7f, g * 0.7f, b * 0.7f, radius * 0.6f);
            }
        }

    } else {
        // ---- 3D demo patterns ---- //

        //pattern A: helical jets from opposing faces (0-8s)
        if (t < 8.0) {
            float angle = (float)t * 1.2f;
            float helix = sinf(angle) * strength * 1.5f;
            float heliy = cosf(angle) * strength * 1.5f;

            //jet from front face → back
            this->solver->addForce(cx + cosf(angle) * W * 0.2f,
                                   cy + sinf(angle) * H * 0.2f,
                                   D * 0.1f,
                                   helix * 0.3f, heliy * 0.3f, strength * 3.0f,
                                   r * 0.5f, g * 0.5f, b * 0.5f, radius * 1.5f);

            //jet from back face → front
            this->solver->addForce(cx - cosf(angle) * W * 0.2f,
                                   cy - sinf(angle) * H * 0.2f,
                                   D * 0.9f,
                                   -helix * 0.3f, -heliy * 0.3f, -strength * 3.0f,
                                   r2 * 0.5f, g2 * 0.5f, b2 * 0.5f, radius * 1.5f);
        }

        //pattern B: 6-face breathing emitters (5s+)
        if (t >= 5.0) {
            float pulse = (sinf((float)t * 1.5f) + 1.0f) * 0.5f;
            float s = strength * 2.0f * pulse;

            //+x face
            this->solver->addForce(W * 0.9f, cy, cz, -s, 0, 0,
                                   r * 0.3f * pulse, g * 0.3f * pulse, b * 0.3f * pulse, radius);
            //-x face
            this->solver->addForce(W * 0.1f, cy, cz, s, 0, 0,
                                   r2 * 0.3f * pulse, g2 * 0.3f * pulse, b2 * 0.3f * pulse, radius);
            //+y face
            this->solver->addForce(cx, H * 0.9f, cz, 0, -s, 0,
                                   r3 * 0.3f * pulse, g3 * 0.3f * pulse, b3 * 0.3f * pulse, radius);
            //-y face
            this->solver->addForce(cx, H * 0.1f, cz, 0, s, 0,
                                   r * 0.3f * pulse, 0, b2 * 0.3f * pulse, radius);
            //+z face
            float za = sinf((float)t * 0.9f) * strength;
            this->solver->addForce(cx, cy, D * 0.9f, za, 0, -s,
                                   0, g3 * 0.3f * pulse, b3 * 0.3f * pulse, radius);
            //-z face
            this->solver->addForce(cx, cy, D * 0.1f, -za, 0, s,
                                   r3 * 0.3f * pulse, g * 0.3f * pulse, 0, radius);
        }

        //pattern C: orbiting point emitter (8s+)
        if (t >= 8.0) {
            float a1 = (float)t * 0.6f;
            float a2 = (float)t * 0.4f;
            float orb = (float)min(W, min(H, D)) * 0.3f;

            float px = cx + cosf(a1) * orb;
            float py = cy + sinf(a1) * cosf(a2) * orb;
            float pz = cz + sinf(a2) * orb;

            //tangential force → swirl
            float fx = -sinf(a1) * strength * 2.0f;
            float fy = cosf(a1) * strength * 2.0f;
            float fz = cosf(a2) * strength * 1.5f;

            float ro, go, bo;
            this->hsvToRgb(fmodf((float)t * 0.12f, 1.0f), 0.9f, 1.0f, ro, go, bo);

            this->solver->addForce(px, py, pz, fx, fy, fz,
                                   ro * 0.6f, go * 0.6f, bo * 0.6f, radius * 1.2f);
        }
    }
}

// ======================== GLFW callbacks ======================== //

void Simulation::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;
    Simulation *sim = (Simulation*)glfwGetWindowUserPointer(window);

    switch (key) {
        case GLFW_KEY_ESCAPE:
            sim->running = false;
            break;
        case GLFW_KEY_SPACE:
            sim->paused = !sim->paused;
            cout << (sim->paused ? "[Sim] Paused" : "[Sim] Resumed") << endl;
            break;
        case GLFW_KEY_R:
            sim->grid->clear();
            if (sim->solver->useGpu) {
                sim->grid->toGpu();
            }
            cout << "[Sim] Reset" << endl;
            break;
        case GLFW_KEY_V:
            sim->renderer->nextVisMode();
            cout << "[Sim] Vis: " << sim->renderer->getVisModeName() << endl;
            break;
    }
}

void Simulation::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    Simulation *sim = (Simulation*)glfwGetWindowUserPointer(window);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        sim->mouseDown = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        sim->rightMouseDown = (action == GLFW_PRESS);
    }
}

void Simulation::scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    Simulation *sim = (Simulation*)glfwGetWindowUserPointer(window);
    if (sim->config.is3D) {
        //3D: scroll to zoom camera
        sim->renderer->zoomCamera((float)-yoffset * 0.2f);
    } else {
        //2D: scroll to change brush radius
        sim->config.dyeRadius += (float)yoffset * 2.0f;
        sim->config.dyeRadius = fmaxf(2.0f, fminf(100.0f, sim->config.dyeRadius));
        cout << "[Sim] Brush radius: " << sim->config.dyeRadius << endl;
    }
}

// ======================== HSV → RGB ======================== //

void Simulation::hsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
    int i = (int)(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (i % 6) {
        case 0: r=v; g=t; b=p; break;
        case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;
        case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;
        case 5: r=v; g=p; b=q; break;
    }
}
