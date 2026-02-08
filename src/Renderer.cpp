#include "Renderer.hpp"
#include <glad/gl.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

// ======================== embedded GLSL shaders ======================== //

// ---- 2D shaders: fullscreen quad + texture ---- //
static const char *vertSrc2D = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
)";

static const char *fragSrc2D = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = texture(uTex, vUV);
}
)";

// ---- 3D shaders: volume raymarching ---- //
static const char *vertSrc3D = R"(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aPos * 0.5 + 0.5;
}
)";

//volume raymarching fragment shader
//shoots ray from camera through each pixel into a unit cube [0,1]³
//accumulates color + opacity using front-to-back compositing
static const char *fragSrc3D = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform sampler3D uVolume;      //3D density texture (RGB)
uniform vec3 uCamPos;           //camera position (world)
uniform mat4 uInvVP;            //inverse(projection * view)
uniform float uStepSize;        //ray step size
uniform int uMaxSteps;          //max ray steps

//ray-AABB intersection for unit cube [0,1]³
vec2 intersectBox(vec3 ro, vec3 rd) {
    vec3 invRd = 1.0 / rd;
    vec3 t0 = (vec3(0.0) - ro) * invRd;
    vec3 t1 = (vec3(1.0) - ro) * invRd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tNear, tFar);
}

void main() {
    //reconstruct ray direction from screen coords via inverse VP
    vec4 ndc = vec4(vUV * 2.0 - 1.0, -1.0, 1.0);
    vec4 worldTarget = uInvVP * ndc;
    worldTarget /= worldTarget.w;
    vec3 rayDir = normalize(worldTarget.xyz - uCamPos);

    //intersect with unit cube
    vec2 tHit = intersectBox(uCamPos, rayDir);
    if (tHit.x > tHit.y) {
        fragColor = vec4(0.02, 0.02, 0.05, 1.0);
        return;
    }

    tHit.x = max(tHit.x, 0.0);

    //front-to-back raymarching
    vec3 color = vec3(0.0);
    float alpha = 0.0;
    float t = tHit.x;

    for (int i = 0; i < uMaxSteps && t < tHit.y && alpha < 0.99; i++) {
        vec3 pos = uCamPos + rayDir * t;
        vec3 sample_c = texture(uVolume, pos).rgb;

        //density = luminance of dye
        float density = dot(sample_c, vec3(0.299, 0.587, 0.114));
        density = clamp(density * 8.0, 0.0, 1.0);

        //emissive color from dye, modulated by density
        vec3 emission = sample_c * density * 3.0;

        //front-to-back compositing
        float a = density * uStepSize * 15.0;
        color += (1.0 - alpha) * a * emission;
        alpha += (1.0 - alpha) * a;

        t += uStepSize;
    }

    //dark background
    vec3 bg = vec3(0.02, 0.02, 0.05);
    color = color + (1.0 - alpha) * bg;

    fragColor = vec4(color, 1.0);
}
)";

// ======================== constructor / destructor ======================== //

Renderer::Renderer(int winW, int winH, int gridW, int gridH, int gridD)
    : windowWidth(winW), windowHeight(winH),
      gridWidth(gridW), gridHeight(gridH), gridDepth(gridD),
      visMode(VisMode::DYE),
      texture2D(0), vao2D(0), vbo2D(0), shader2D(0), pixels(nullptr),
      texture3D(0), vao3D(0), vbo3D(0), shader3D(0),
      camTheta(0.7f), camPhi(0.4f), camDist(2.5f), camFov(45.0f) {
    this->is3D = (gridD > 1);
}

Renderer::~Renderer() {
    if (this->pixels) delete[] this->pixels;
    if (this->texture2D) glDeleteTextures(1, &this->texture2D);
    if (this->texture3D) glDeleteTextures(1, &this->texture3D);
    if (this->vao2D) glDeleteVertexArrays(1, &this->vao2D);
    if (this->vbo2D) glDeleteBuffers(1, &this->vbo2D);
    if (this->vao3D) glDeleteVertexArrays(1, &this->vao3D);
    if (this->vbo3D) glDeleteBuffers(1, &this->vbo3D);
    if (this->shader2D) glDeleteProgram(this->shader2D);
    if (this->shader3D) glDeleteProgram(this->shader3D);
}

// ======================== init ======================== //

void Renderer::init() {
    //compile shaders
    this->shader2D = this->compileShader(vertSrc2D, fragSrc2D);

    //2D fullscreen quad (pos + uv)
    float quadVerts[] = {
        -1, -1,  0, 0,
         1, -1,  1, 0,
        -1,  1,  0, 1,
         1, -1,  1, 0,
         1,  1,  1, 1,
        -1,  1,  0, 1
    };

    glGenVertexArrays(1, &this->vao2D);
    glGenBuffers(1, &this->vbo2D);
    glBindVertexArray(this->vao2D);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo2D);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    //2D texture
    glGenTextures(1, &this->texture2D);
    glBindTexture(GL_TEXTURE_2D, this->texture2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    //pixel buffer for 2D
    this->pixels = new unsigned char[this->gridWidth * this->gridHeight * 4]();

    if (this->is3D) {
        //3D shader
        this->shader3D = this->compileShader(vertSrc3D, fragSrc3D);

        //3D fullscreen quad (just positions, UV computed in shader)
        float quadVerts3D[] = {
            -1, -1,  1, -1,  -1, 1,
             1, -1,  1,  1,  -1, 1
        };

        glGenVertexArrays(1, &this->vao3D);
        glGenBuffers(1, &this->vbo3D);
        glBindVertexArray(this->vao3D);
        glBindBuffer(GL_ARRAY_BUFFER, this->vbo3D);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts3D), quadVerts3D, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        //3D texture
        glGenTextures(1, &this->texture3D);
        glBindTexture(GL_TEXTURE_3D, this->texture3D);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        cout << "[Renderer] 3D volume renderer initialized" << endl;
    }

    cout << "[Renderer] OpenGL initialized (shader2D=" << this->shader2D << ")" << endl;
}

// ======================== render ======================== //

void Renderer::render(Grid *grid) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (this->is3D) {
        this->render3D(grid);
    } else {
        this->render2D(grid);
    }
}

// ======================== 2D rendering ======================== //

void Renderer::render2D(Grid *grid) {
    this->updateTexture2D(grid);

    glUseProgram(this->shader2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->texture2D);
    glUniform1i(glGetUniformLocation(this->shader2D, "uTex"), 0);

    glBindVertexArray(this->vao2D);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Renderer::updateTexture2D(Grid *grid) {
    switch (this->visMode) {
        case VisMode::DYE:       this->dyeToPixels(grid);       break;
        case VisMode::VELOCITY:  this->velocityToPixels(grid);  break;
        case VisMode::PRESSURE:  this->pressureToPixels(grid);  break;
        case VisMode::VORTICITY: this->vorticityToPixels(grid); break;
    }

    glBindTexture(GL_TEXTURE_2D, this->texture2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->gridWidth, this->gridHeight,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, this->pixels);
}

void Renderer::dyeToPixels(Grid *grid) {
    int W = this->gridWidth, H = this->gridHeight;
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            int gIdx = j * W + i;
            int pIdx = ((H - 1 - j) * W + i) * 4;  //flip Y for OpenGL

            this->pixels[pIdx + 0] = (unsigned char)min(255.0f, max(0.0f, grid->densityR[gIdx] * 255.0f));
            this->pixels[pIdx + 1] = (unsigned char)min(255.0f, max(0.0f, grid->densityG[gIdx] * 255.0f));
            this->pixels[pIdx + 2] = (unsigned char)min(255.0f, max(0.0f, grid->densityB[gIdx] * 255.0f));
            this->pixels[pIdx + 3] = 255;
        }
    }
}

void Renderer::velocityToPixels(Grid *grid) {
    int W = this->gridWidth, H = this->gridHeight;
    float maxV = 0.01f;
    for (int i = 0; i < grid->size; i++) {
        float v = sqrtf(grid->vx[i] * grid->vx[i] + grid->vy[i] * grid->vy[i]);
        if (v > maxV) maxV = v;
    }

    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            int gIdx = j * W + i;
            int pIdx = ((H - 1 - j) * W + i) * 4;
            float v = sqrtf(grid->vx[gIdx] * grid->vx[gIdx] + grid->vy[gIdx] * grid->vy[gIdx]);
            float t = v / maxV;
            this->heatMap(t, this->pixels[pIdx], this->pixels[pIdx + 1], this->pixels[pIdx + 2]);
            this->pixels[pIdx + 3] = 255;
        }
    }
}

void Renderer::pressureToPixels(Grid *grid) {
    int W = this->gridWidth, H = this->gridHeight;
    float maxP = 0.01f;
    for (int i = 0; i < grid->size; i++) {
        float p = fabsf(grid->pressure[i]);
        if (p > maxP) maxP = p;
    }

    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            int gIdx = j * W + i;
            int pIdx = ((H - 1 - j) * W + i) * 4;
            float t = grid->pressure[gIdx] / maxP * 0.5f + 0.5f;
            this->heatMap(t, this->pixels[pIdx], this->pixels[pIdx + 1], this->pixels[pIdx + 2]);
            this->pixels[pIdx + 3] = 255;
        }
    }
}

void Renderer::vorticityToPixels(Grid *grid) {
    int W = this->gridWidth, H = this->gridHeight;
    float maxW = 0.01f;
    for (int i = 0; i < grid->size; i++) {
        float w = fabsf(grid->vorticity[i]);
        if (w > maxW) maxW = w;
    }

    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            int gIdx = j * W + i;
            int pIdx = ((H - 1 - j) * W + i) * 4;
            float t = grid->vorticity[gIdx] / maxW * 0.5f + 0.5f;
            this->heatMap(t, this->pixels[pIdx], this->pixels[pIdx + 1], this->pixels[pIdx + 2]);
            this->pixels[pIdx + 3] = 255;
        }
    }
}

// ======================== 3D rendering ======================== //

void Renderer::render3D(Grid *grid) {
    this->updateTexture3D(grid);

    //compute camera position on sphere
    float cx = this->camDist * cosf(this->camPhi) * sinf(this->camTheta) + 0.5f;
    float cy = this->camDist * sinf(this->camPhi) + 0.5f;
    float cz = this->camDist * cosf(this->camPhi) * cosf(this->camTheta) + 0.5f;

    //view matrix (lookAt)
    float target[3] = { 0.5f, 0.5f, 0.5f };  //center of unit cube
    float forward[3] = { target[0]-cx, target[1]-cy, target[2]-cz };
    float flen = sqrtf(forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]);
    forward[0] /= flen; forward[1] /= flen; forward[2] /= flen;

    float up[3] = { 0, 1, 0 };
    //right = forward × up
    float right[3] = {
        forward[1]*up[2] - forward[2]*up[1],
        forward[2]*up[0] - forward[0]*up[2],
        forward[0]*up[1] - forward[1]*up[0]
    };
    float rlen = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    right[0] /= rlen; right[1] /= rlen; right[2] /= rlen;

    //recalc up = right × forward
    up[0] = right[1]*forward[2] - right[2]*forward[1];
    up[1] = right[2]*forward[0] - right[0]*forward[2];
    up[2] = right[0]*forward[1] - right[1]*forward[0];

    //view matrix (column-major for OpenGL)
    float view[16] = {
        right[0],   up[0],   -forward[0],  0,
        right[1],   up[1],   -forward[1],  0,
        right[2],   up[2],   -forward[2],  0,
        -(right[0]*cx + right[1]*cy + right[2]*cz),
        -(up[0]*cx + up[1]*cy + up[2]*cz),
        (forward[0]*cx + forward[1]*cy + forward[2]*cz),
        1
    };

    //projection matrix (perspective)
    float aspect = (float)this->windowWidth / (float)this->windowHeight;
    float fovRad = this->camFov * 3.14159265f / 180.0f;
    float tanHalf = tanf(fovRad / 2.0f);
    float near = 0.1f, far = 100.0f;

    float proj[16] = { 0 };
    proj[0]  = 1.0f / (aspect * tanHalf);
    proj[5]  = 1.0f / tanHalf;
    proj[10] = -(far + near) / (far - near);
    proj[11] = -1.0f;
    proj[14] = -(2.0f * far * near) / (far - near);

    //VP = proj * view
    float vp[16] = { 0 };
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            for (int k = 0; k < 4; k++)
                vp[c * 4 + r] += proj[k * 4 + r] * view[c * 4 + k];

    //invert VP (for ray reconstruction)
    //simple 4x4 inverse using cofactors
    float inv[16];
    {
        float m[16];
        memcpy(m, vp, 16 * sizeof(float));

        float s[6], c2[6];
        s[0] = m[0]*m[5] - m[4]*m[1];
        s[1] = m[0]*m[6] - m[4]*m[2];
        s[2] = m[0]*m[7] - m[4]*m[3];
        s[3] = m[1]*m[6] - m[5]*m[2];
        s[4] = m[1]*m[7] - m[5]*m[3];
        s[5] = m[2]*m[7] - m[6]*m[3];

        c2[0] = m[8]*m[13]  - m[12]*m[9];
        c2[1] = m[8]*m[14]  - m[12]*m[10];
        c2[2] = m[8]*m[15]  - m[12]*m[11];
        c2[3] = m[9]*m[14]  - m[13]*m[10];
        c2[4] = m[9]*m[15]  - m[13]*m[11];
        c2[5] = m[10]*m[15] - m[14]*m[11];

        float det = s[0]*c2[5] - s[1]*c2[4] + s[2]*c2[3] + s[3]*c2[2] - s[4]*c2[1] + s[5]*c2[0];
        if (fabsf(det) < 1e-10f) det = 1e-10f;
        float idet = 1.0f / det;

        inv[0]  = ( m[5]*c2[5] - m[6]*c2[4] + m[7]*c2[3]) * idet;
        inv[1]  = (-m[1]*c2[5] + m[2]*c2[4] - m[3]*c2[3]) * idet;
        inv[2]  = ( m[13]*s[5] - m[14]*s[4] + m[15]*s[3]) * idet;
        inv[3]  = (-m[9]*s[5]  + m[10]*s[4] - m[11]*s[3]) * idet;

        inv[4]  = (-m[4]*c2[5] + m[6]*c2[2] - m[7]*c2[1]) * idet;
        inv[5]  = ( m[0]*c2[5] - m[2]*c2[2] + m[3]*c2[1]) * idet;
        inv[6]  = (-m[12]*s[5] + m[14]*s[2] - m[15]*s[1]) * idet;
        inv[7]  = ( m[8]*s[5]  - m[10]*s[2] + m[11]*s[1]) * idet;

        inv[8]  = ( m[4]*c2[4] - m[5]*c2[2] + m[7]*c2[0]) * idet;
        inv[9]  = (-m[0]*c2[4] + m[1]*c2[2] - m[3]*c2[0]) * idet;
        inv[10] = ( m[12]*s[4] - m[13]*s[2] + m[15]*s[0]) * idet;
        inv[11] = (-m[8]*s[4]  + m[9]*s[2]  - m[11]*s[0]) * idet;

        inv[12] = (-m[4]*c2[3] + m[5]*c2[1] - m[6]*c2[0]) * idet;
        inv[13] = ( m[0]*c2[3] - m[1]*c2[1] + m[2]*c2[0]) * idet;
        inv[14] = (-m[12]*s[3] + m[13]*s[1] - m[14]*s[0]) * idet;
        inv[15] = ( m[8]*s[3]  - m[9]*s[1]  + m[10]*s[0]) * idet;
    }

    glUseProgram(this->shader3D);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, this->texture3D);
    glUniform1i(glGetUniformLocation(this->shader3D, "uVolume"), 0);
    glUniform3f(glGetUniformLocation(this->shader3D, "uCamPos"), cx, cy, cz);
    glUniformMatrix4fv(glGetUniformLocation(this->shader3D, "uInvVP"), 1, GL_FALSE, inv);

    float stepSize = 1.0f / (float)max(max(this->gridWidth, this->gridHeight), this->gridDepth);
    glUniform1f(glGetUniformLocation(this->shader3D, "uStepSize"), stepSize);
    glUniform1i(glGetUniformLocation(this->shader3D, "uMaxSteps"), 512);

    glBindVertexArray(this->vao3D);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Renderer::updateTexture3D(Grid *grid) {
    int W = this->gridWidth, H = this->gridHeight, D = this->gridDepth;
    int size = W * H * D;

    //pack RGB density into float RGBA for 3D texture
    //use GL_RGB + GL_FLOAT directly
    float *volumeData = new float[size * 3];

    for (int i = 0; i < size; i++) {
        volumeData[i * 3 + 0] = max(0.0f, min(1.0f, grid->densityR[i]));
        volumeData[i * 3 + 1] = max(0.0f, min(1.0f, grid->densityG[i]));
        volumeData[i * 3 + 2] = max(0.0f, min(1.0f, grid->densityB[i]));
    }

    glBindTexture(GL_TEXTURE_3D, this->texture3D);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F, W, H, D, 0, GL_RGB, GL_FLOAT, volumeData);

    delete[] volumeData;
}

// ======================== utilities ======================== //

void Renderer::nextVisMode() {
    int m = ((int)this->visMode + 1) % 4;
    this->visMode = (VisMode)m;
}

string Renderer::getVisModeName() {
    switch (this->visMode) {
        case VisMode::DYE:       return "Dye Density";
        case VisMode::VELOCITY:  return "Velocity";
        case VisMode::PRESSURE:  return "Pressure";
        case VisMode::VORTICITY: return "Vorticity";
    }
    return "Unknown";
}

void Renderer::rotateCamera(float dTheta, float dPhi) {
    this->camTheta += dTheta;
    this->camPhi   += dPhi;
    //clamp phi to avoid gimbal lock
    this->camPhi = fmaxf(-1.4f, fminf(1.4f, this->camPhi));
}

void Renderer::zoomCamera(float delta) {
    this->camDist += delta;
    this->camDist = fmaxf(0.5f, fminf(10.0f, this->camDist));
}

//heat map: blue → cyan → green → yellow → red
void Renderer::heatMap(float t, unsigned char &r, unsigned char &g, unsigned char &b) {
    t = fmaxf(0.0f, fminf(1.0f, t));
    if (t < 0.25f) {
        float s = t / 0.25f;
        r = 0; g = (unsigned char)(s * 255); b = 255;
    } else if (t < 0.5f) {
        float s = (t - 0.25f) / 0.25f;
        r = 0; g = 255; b = (unsigned char)((1 - s) * 255);
    } else if (t < 0.75f) {
        float s = (t - 0.5f) / 0.25f;
        r = (unsigned char)(s * 255); g = 255; b = 0;
    } else {
        float s = (t - 0.75f) / 0.25f;
        r = 255; g = (unsigned char)((1 - s) * 255); b = 0;
    }
}

// ======================== shader compilation ======================== //

unsigned int Renderer::compileShader(const char *vertSrc, const char *fragSrc) {
    unsigned int vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vertSrc, nullptr);
    glCompileShader(vert);
    int ok;
    glGetShaderiv(vert, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(vert, 512, nullptr, log);
        cerr << "[Shader] Vertex error: " << log << endl;
    }

    unsigned int frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fragSrc, nullptr);
    glCompileShader(frag);
    glGetShaderiv(frag, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(frag, 512, nullptr, log);
        cerr << "[Shader] Fragment error: " << log << endl;
    }

    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        cerr << "[Shader] Link error: " << log << endl;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}
