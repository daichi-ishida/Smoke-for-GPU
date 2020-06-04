#pragma once

constexpr int DIM = 256;
constexpr int xRes = DIM*2;
constexpr int yRes = DIM;
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
constexpr float TOLERANCE = 1.0e-2f;
constexpr int MAX_ITER = 5000;

constexpr float INIT_DENSITY = 0.1f;
constexpr float INIT_VELOCITY = ((float)DIM) * DX / 5.0f;
constexpr float INFLOW = ((float)DIM) * DX / 8.0f;
constexpr float INIT_TEMPERATURE = 50.0f;
constexpr float INIT_COLD = -3.0f;

constexpr int SOURCE_SIZE_X = 2*DIM/25;
constexpr int SOURCE_MARGIN_X = 8*DIM/250;
constexpr int SOURCE_RADIUS_YZ = DIM/20;
constexpr int SOURCE_CENTER_Y = 9*DIM/10;
constexpr int SOURCE_CENTER_Z = DIM/2;

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
#ifdef _MSC_VER
#define OPENMP_FOR __pragma("omp parallel for")
#define OPENMP_FOR_COLLAPSE __pragma("omp parallel for collapse(3)")
#else
#define OPENMP_FOR _Pragma("omp parallel for")
#define OPENMP_FOR_COLLAPSE _Pragma("omp parallel for collapse(3)")
#endif
#else
#define OPENMP_FOR
#define OPENMP_FOR_COLLAPSE
#endif
