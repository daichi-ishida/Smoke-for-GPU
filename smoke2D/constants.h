#pragma once

constexpr int DIM = 512;
constexpr int xRes = DIM;
constexpr int yRes = DIM;
constexpr float Scale = 1.0f;

constexpr int xThreadDim = 16;
constexpr int yThreadDim = 16;

constexpr int xBlockMaxDim = (int)((xRes + xThreadDim - 1) / xThreadDim);
constexpr int yBlockMaxDim = (int)((yRes + yThreadDim - 1) / yThreadDim);
constexpr int blockArraySize = xBlockMaxDim * yBlockMaxDim;

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
constexpr float TOLERANCE = 1.0e-4f;
constexpr int MAX_ITER = 5000;

constexpr float INIT_DENSITY = 0.1f;
constexpr float INIT_VELOCITY = 10.0f * DX;
constexpr float INIT_TEMPERATURE = 30.0f;
constexpr float INFLOW = 2.0f * DX;

constexpr int SOURCE_SIZE_X = (int)(DIM / 3);
constexpr int SOURCE_SIZE_Y = (int)(DIM / 20);
constexpr int SOURCE_Y_MERGIN = (int)(DIM / 20);

constexpr float COLLISION_RADIUS = (float)DIM / 6.0f;
constexpr float COLLISION_CENTER_X = 2.0f * (float)xRes / 3.0f;
constexpr float COLLISION_CENTER_Y = (float)yRes / 2.0f;
constexpr float R2 = COLLISION_RADIUS * COLLISION_RADIUS;

constexpr int END_FRAME = 720;

constexpr int WIN_WIDTH = 1024;
constexpr int WIN_HEIGHT = 1024;
static const char *WIN_TITLE = "Smoke with GPU 2D";
constexpr bool OFFSCREEN_MODE = false;
constexpr bool SAVE_IMAGE = true;

#ifdef _OPENMP
#include <omp.h>
#define OPENMP_FOR __pragma("omp parallel for")
#define OPENMP_FOR_COLLAPSE __pragma("omp parallel for collapse(2)")
#else
#define OPENMP_FOR
#define OPENMP_FOR_COLLAPSE
#endif
