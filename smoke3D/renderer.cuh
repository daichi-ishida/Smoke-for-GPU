#pragma once
#define _USE_MATH_DEFINES
#include <math.h>

#include "camera.h"
#include "helper_cuda.h"
#include "helper_math.h"

constexpr float M_PIf = static_cast<float>(M_PI);

inline __device__ float3 expf(const float3& a)
{
    return make_float3(expf(a.x), expf(a.y), expf(a.z));
}


// Henyey-Greenstein function:
// Let x = angle between light and camera vectors
//     g = Mie scattering coefficient (ScatterFalloff)
// f(x) = 1 - g^2 / (4PI * (1 + g^2 - 2g*cos(x))^[3/2])
__device__ float mieScatter(const float3& light_dir, const float3& cam_dir, const float& g)
{
    float n = 1.0f - g * g;
    float c = dot(light_dir, cam_dir);
    float d = 1.0f + g * g - 2.0f * g * c; // 1 + g^2 - 2g*cos(x)
    return n / (4.0f * M_PIf * pow(d, 1.5f));
}

__device__ bool isIntersectBox(const Ray& r, const float3& boxmin, const float3& boxmax, float* tnear, float* tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

__device__ bool isInGridBox(const float3& grid_pos)
{
    if ((grid_pos.x >= 0.0f) && (grid_pos.x <= (float)xRes) &&
        (grid_pos.y >= 0.0f) && (grid_pos.y <= (float)yRes) &&
        (grid_pos.z >= 0.0f) && (grid_pos.z <= (float)zRes))
    {
        return true;
    }
    return false;
}

__device__ float3 convertToGridCoordinate(float3 pos)
{
    pos += make_float3(2.0f, 1.0f, 1.0f);
    pos *= 0.5f;
    pos *= (float)DIM;
    return pos;
}

__device__ float3 lambertian(const float3& m_col, const float3& lightDir, const float3& normal)
{
    float cosine = dot(normal, lightDir);
    return m_col * max(cosine, 0.0f) * (1.0f / M_PI);
}

__device__ float phong(const float3& lightDir, const float3& normal, const float3& cam_dir)
{
    float shininess = 30.0f;
    float cos_in = dot(normal, lightDir);
    float3 reflect = normalize(2.0f * normal * cos_in - lightDir);
    float cos_out_m = pow(max(0.0f, dot(reflect, cam_dir)), shininess);
    return cos_out_m / (M_PI * cos_in);
}

__device__ float3 drawWall(const float3& wpos, const float3& light_wpos, const float3& cam_dir)
{
    float3 normal;
    if (wpos.x <= -2.0f)
    {
        normal = make_float3(-1.0f, 0.0f, 0.0f);
    }
    else if (wpos.x >= 2.0f)
    {
        normal = make_float3(1.0f, 0.0f, 0.0f);
    }
    else if (wpos.y <= -2.0f)
    {
        normal = make_float3(0.0f, 1.0f, 0.0f);
    }
    else if (wpos.y >= 2.0f)
    {
        normal = make_float3(0.0f, -1.0f, 0.0f);
    }
    else if (wpos.z <= -2.0f)
    {
        normal = make_float3(0.0f, 0.0f, -1.0f);
    }
    else
    {
        normal = make_float3(0.0f, 0.0f, 1.0f);
    }

    float3 light_dir = normalize(light_wpos - wpos);
    float3 m_col = make_float3(0.961f, 0.664f, 0.141f);
    float3 diffuse = lambertian(m_col, light_dir, normal);
    float3 specular = make_float3(phong(light_dir, normal, cam_dir));
    float3 col = diffuse + specular;
    col = make_float3(min(col.x, 1.0f), min(col.y, 1.0f), min(col.z, 1.0f));

    return col;
}

__device__ float grayScale(const float3& color)
{
    return 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
}