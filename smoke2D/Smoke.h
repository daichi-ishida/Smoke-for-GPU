#pragma once
#include "FieldStructure.h"

#include <thrust/device_vector.h>


struct Smoke
{
    // ### fields ###
    ScalarField density;
    ScalarField density0;

    ScalarField temperature;
    ScalarField temperature0;

    uField u;
    uField u0;

    vField v;
    vField v0;

    ScalarField force_y;

    ScalarField pressure;
    ScalarField Ax;
    ScalarField divergence;
    ScalarField direction;

    Obstacles obstacles;

    float obstacle_cx;
    float obstacle_cy;
    float obstacle_u;
    float obstacle_v;

    // time step
    float dt;
    float t;
    float next_shutter_time;
    bool isTimeToRender;
    thrust::device_vector<float> d_speed;

    Smoke();
    ~Smoke();

    void initialize();
    void setNextShutterTime();

    // initial setup
    void initVelocity();
    void initDensity();
    void initTemperature();
    void setObstacles();

    // d_density
    thrust::device_vector<float> d_density_data;

    // d_density0
    thrust::device_vector<float> d_density0_data;

    // d_temperature
    thrust::device_vector<float> d_temperature_data;

    // d_temperature0
    thrust::device_vector<float> d_temperature0_data;

    // d_u
    thrust::device_vector<float> d_u_data;

    // d_u0
    thrust::device_vector<float> d_u0_data;

    // d_v
    thrust::device_vector<float> d_v_data;

    // d_v0
    thrust::device_vector<float> d_v0_data;

    // d_force_y
    thrust::device_vector<float> d_force_y_data;

    // d_pressure
    thrust::device_vector<float> d_pressure_data;

    // d_Ax, d_divergence, d_direction
    thrust::device_vector<float> d_Ax_data;
    thrust::device_vector<float> d_divergence_data;
    thrust::device_vector<float> d_direction_data;

    // obstacles
    thrust::device_vector<bool> d_obstacles_data;
};