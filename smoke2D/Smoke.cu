#define _USE_MATH_DEFINES
#include <math.h>

#include "constants.h"
#include "Smoke.h"

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <algorithm>

#include "helper_cuda.h"

#define FOR_EACH_CELL  \
    for (int j = 0; j < yRes; ++j) \
        for (int i = 0; i < xRes; ++i)

#define FOR_EACH_FACE_X  \
    for (int j = 0; j < yRes; ++j) \
        for (int i = 0; i < xRes + 1; ++i)

#define FOR_EACH_FACE_Y  \
    for (int j = 0; j < yRes + 1; ++j) \
        for (int i = 0; i < xRes; ++i)


Smoke::Smoke()
    : dt(0.0f), t(0.0f), next_shutter_time(0.0f), isTimeToRender(false)
{
    //dt = std::min(CFL * DX / INIT_VELOCITY, 1.0f / FPS);
    dt = DT;
}

Smoke::~Smoke()
{
}

void Smoke::initialize()
{
    // initial setup
    printf("initializing data...");
    initVelocity();
    initDensity();
    initTemperature();
    setObstacles();

    d_force_y_data.resize(xRes * yRes);
    d_pressure_data.resize(xRes * yRes);
    d_Ax_data.resize(xRes * yRes);
    d_divergence_data.resize(xRes * yRes);
    d_direction_data.resize(xRes * yRes);

    force_y.data = d_force_y_data.data();
    pressure.data = d_pressure_data.data();
    Ax.data = d_Ax_data.data();
    divergence.data = d_divergence_data.data();
    direction.data = d_direction_data.data();

    d_speed.resize(xBlockMaxDim * yBlockMaxDim);

    printf("Done\n");
}

void Smoke::setNextShutterTime()
{
    next_shutter_time = t + 1.0f / FPS;
}


void Smoke::initVelocity()
{
    thrust::host_vector<float> h_u0_scanline((xRes + 1) * yRes);
    thrust::host_vector<float> h_v0_scanline(xRes * (yRes + 1));

    d_u0_data.resize((xRes + 1) * yRes);
    d_v0_data.resize(xRes * (yRes + 1));

    d_u_data.resize((xRes + 1) * yRes);
    d_v_data.resize(xRes * (yRes + 1));

    // u
    OPENMP_FOR_COLLAPSE
        FOR_EACH_FACE_X
    {
        int offset = i + j * (xRes + 1);
        h_u0_scanline[offset] = 0.0f;
    }

        // v
    OPENMP_FOR_COLLAPSE
        FOR_EACH_FACE_Y
    {
        int offset = i + j * xRes;
        h_v0_scanline[offset] = 0.0f;
        if (i >= (xRes - SOURCE_SIZE_X) / 2 && i < (xRes + SOURCE_SIZE_X) / 2 && j >= yRes - SOURCE_Y_MERGIN - SOURCE_SIZE_Y && j < yRes - SOURCE_Y_MERGIN)
        {
            h_v0_scanline[offset] = -INIT_VELOCITY;
        }
    }

    d_u0_data = h_u0_scanline;
    d_v0_data = h_v0_scanline;

    u0.data = d_u0_data.data();
    v0.data = d_v0_data.data();

    u.data = d_u_data.data();
    v.data = d_v_data.data();
}

void Smoke::initDensity()
{
    thrust::host_vector<float> h_density0_scanline(xRes * yRes);

    d_density0_data.resize(xRes * yRes);
    d_density_data.resize(xRes * yRes);

    OPENMP_FOR_COLLAPSE
        FOR_EACH_CELL
    {
        int offset = i + j * xRes;
        if (i >= (xRes - SOURCE_SIZE_X) / 2 && i < (xRes + SOURCE_SIZE_X) / 2 && j >= yRes - SOURCE_Y_MERGIN - SOURCE_SIZE_Y && j < yRes - SOURCE_Y_MERGIN)
        {
            h_density0_scanline[offset] = INIT_DENSITY;
        }
    }

    d_density0_data = h_density0_scanline;

    density0.data = d_density0_data.data();
    density.data = d_density_data.data();
}

void Smoke::initTemperature()
{
    thrust::host_vector<float> h_temperature0_scanline(xRes * yRes);

    d_temperature0_data.resize(xRes * yRes);
    d_temperature_data.resize(xRes * yRes);

    OPENMP_FOR_COLLAPSE
        FOR_EACH_CELL
    {
        int offset = i + j * xRes;
        if (i >= (xRes - SOURCE_SIZE_X) / 2 && i < (xRes + SOURCE_SIZE_X) / 2 && j >= yRes - SOURCE_Y_MERGIN - SOURCE_SIZE_Y && j < yRes - SOURCE_Y_MERGIN)
        {
            h_temperature0_scanline[offset] = INIT_TEMPERATURE;
        }
    }

    d_temperature0_data = h_temperature0_scanline;

    temperature0.data = d_temperature0_data.data();
    temperature.data = d_temperature_data.data();
}

void Smoke::setObstacles()
{
    std::vector<bool> h_scanline_obstacles(xRes * yRes);

    d_obstacles_data.resize(xRes * yRes);

    OPENMP_FOR_COLLAPSE
        FOR_EACH_CELL
    {
        float r2 = ((static_cast<float>(i) + 0.5f) - COLLISION_CENTER_X)* ((static_cast<float>(i) + 0.5f) - COLLISION_CENTER_X) + ((static_cast<float>(j) + 0.5f) - COLLISION_CENTER_Y)* ((static_cast<float>(j) + 0.5f) - COLLISION_CENTER_Y);
        h_scanline_obstacles[i + j * xRes] = (r2 <= R2);
        //h_scanline_obstacles[i + j * xRes] = false;
    }

    d_obstacles_data = h_scanline_obstacles;

    obstacles.data = d_obstacles_data.data();
}


