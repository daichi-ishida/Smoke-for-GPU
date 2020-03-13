#pragma once

#include <cuda_runtime.h>

#include "helper_math.h"

#include "constants.h"

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

struct Camera
{
    Camera() = default;
    Camera(float r, float horizontalAngle, float verticalAngle, float FoV)
    {
        pos = make_float3(r * sin(verticalAngle) * sin(horizontalAngle),
            -r * cos(verticalAngle),
            r * sin(verticalAngle) * cos(horizontalAngle));

        target = make_float3(0.0f, 0.0f, 0.0f);
        front = normalize(target - pos);
        right = make_float3(cos(horizontalAngle), 0.0f, -sin(horizontalAngle));
        up = cross(right, front);
        invhalffov = 1.0f / tan(FoV / 2.0f);
    }

    float3 pos, front, right, up, target;
    float invhalffov;

    __device__ Ray generateRay(int x, int y) const
    {
        float nx = ((float)x / (float)WIN_WIDTH) * 2.0f - 1.0f;
        float ny = ((float)y / (float)WIN_HEIGHT) * 2.0f - 1.0f;

        Ray ray = { pos, normalize(invhalffov * front + nx * right + ny * up) };
        return ray;
    }
};