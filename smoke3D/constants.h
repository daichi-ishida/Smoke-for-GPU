#pragma once

constexpr int DIM = 128;
constexpr int xRes = DIM;
constexpr int yRes = DIM * 2;
constexpr int zRes = DIM;
constexpr float Scale = 1.0f;

constexpr int xThreadDim = 8;
constexpr int yThreadDim = 8;
constexpr int zThreadDim = 8;

constexpr int xBlockMaxDim = (int)((xRes + xThreadDim - 1) / xThreadDim);
constexpr int yBlockMaxDim = (int)((yRes + yThreadDim - 1) / yThreadDim);
constexpr int zBlockMaxDim = (int)((zRes + zThreadDim - 1) / zThreadDim);

constexpr int blockArraySize = xBlockMaxDim * yBlockMaxDim * zBlockMaxDim;

constexpr float FPS = 60.0f;
constexpr float CFL = 2.0f;
constexpr float DX = Scale / (float)DIM;
constexpr float RHO = 1.29f;

constexpr float DT = 0.004f;

// ALPHA: gravity
// BETA:  buoyancy
constexpr float ALPHA = 9.8f;
constexpr float BETA = 0.004f * ALPHA;

// projection
constexpr float TOLERANCE = 1.0e-3f;
constexpr int MAX_ITER = 5000;

constexpr float INIT_DENSITY = 0.1f;
constexpr float INIT_VELOCITY = 10.0f * DX;
constexpr float INIT_TEMPERATURE = 30.0f;
constexpr float INFLOW = 2.0f * DX;

constexpr int SOURCE_R = (int)(DIM / 8);
constexpr int SOURCE_X = (int)(DIM / 2);
constexpr int SOURCE_Y = (int)(DIM / 5);
constexpr int SOURCE_Z = (int)(DIM / 2);

constexpr float COLLISION_RADIUS = (float)DIM / 6.0f;
constexpr float COLLISION_CENTER_X = (float)xRes / 2.0f;
constexpr float COLLISION_CENTER_Y = (float)yRes / 2.0f;
constexpr float COLLISION_CENTER_Z = (float)zRes / 2.0f;
constexpr float R2 = COLLISION_RADIUS * COLLISION_RADIUS;

// Output Setting
constexpr bool OFFSCREEN_MODE = false;
constexpr bool SAVE_IMAGE = true;
constexpr bool SAVE_VDB = true;
constexpr int END_FRAME = 900;

constexpr int WIN_WIDTH = 1024;
constexpr int WIN_HEIGHT = 1024;
static const char* WIN_TITLE = "Smoke with GPU 3D";

#ifdef _OPENMP
#include <omp.h>
#define OPENMP_FOR __pragma("omp parallel for")
#define OPENMP_FOR_COLLAPSE __pragma("omp parallel for collapse(3)")
#else
#define OPENMP_FOR
#define OPENMP_FOR_COLLAPSE
#endif
