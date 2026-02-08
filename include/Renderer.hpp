#ifndef _RENDERER_HPP_
#define _RENDERER_HPP_

#include "Grid.hpp"
#include <string>

using namespace std;

enum class VisMode {
    DYE,                //RGB dye density
    VELOCITY,           //velocity magnitude heat map
    PRESSURE,           //pressure field heat map
    VORTICITY           //vorticity (curl) heat map
};

//OpenGL renderer
//2D mode: textured fullscreen quad
//3D mode: volume raymarching through 3D texture with orbiting camera
class Renderer {
public:
    int             windowWidth;
    int             windowHeight;
    int             gridWidth;
    int             gridHeight;
    int             gridDepth;
    bool            is3D;

    VisMode         visMode;

    //2D resources
    unsigned int    texture2D;          //2D texture
    unsigned int    vao2D;
    unsigned int    vbo2D;
    unsigned int    shader2D;
    unsigned char*  pixels;             //2D pixel buffer (RGBA)

    //3D resources
    unsigned int    texture3D;          //3D texture (density volume)
    unsigned int    vao3D;
    unsigned int    vbo3D;
    unsigned int    shader3D;           //volume raymarching shader

    //camera (3D mode)
    float           camTheta;           //azimuth angle (radians)
    float           camPhi;             //elevation angle (radians)
    float           camDist;            //distance from center
    float           camFov;             //field of view (degrees)

    Renderer(int winW, int winH, int gridW, int gridH, int gridD);
    ~Renderer();

    void init();
    void render(Grid *grid);
    void nextVisMode();
    string getVisModeName();

    //camera controls (3D)
    void rotateCamera(float dTheta, float dPhi);
    void zoomCamera(float delta);

private:
    //shader compilation
    unsigned int compileShader(const char *vertSrc, const char *fragSrc);

    //2D rendering
    void render2D(Grid *grid);
    void updateTexture2D(Grid *grid);
    void dyeToPixels(Grid *grid);
    void velocityToPixels(Grid *grid);
    void pressureToPixels(Grid *grid);
    void vorticityToPixels(Grid *grid);

    //3D rendering
    void render3D(Grid *grid);
    void updateTexture3D(Grid *grid);

    //color mapping
    void heatMap(float t, unsigned char &r, unsigned char &g, unsigned char &b);
};

#endif
