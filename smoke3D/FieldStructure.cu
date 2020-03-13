
#include "FieldStructure.h"

#include <thrust/execution_policy.h>
#include <thrust/swap.h>

#include "helper_cuda.h"
#include "helper_math.h"

#include <cassert>

#include "constants.h"


__host__ __device__
Index::Index(int _i, int _j, int _k) : i(_i), j(_j), k(_k)
{
    assert(i < xRes);
    assert(j < yRes);
    assert(k < zRes);
}

inline __host__ __device__
int Index::globalOffset() const
{
    return i + (j + k * yRes) * xRes;
}

__host__ __device__
Index Index::front() const
{
    if (k == 0)
    {
        return Index(i, j, k);
    }
    return Index(i, j , k - 1);
}
__host__ __device__
Index Index::back() const
{
    if (k == zRes - 1)
    {
        return Index(i, j, k);
    }
    return Index(i, j, k + 1);
}
__host__ __device__
Index Index::top() const
{
    if (j == 0)
    {
        return Index(i, j, k);
    }
    return Index(i, j - 1, k);
}
__host__ __device__
Index Index::bottom() const
{
    if (j == yRes - 1)
    {
        return Index(i, j, k);
    }
    return Index(i, j + 1, k);
}
__host__ __device__
Index Index::left() const
{
    if (i == 0)
    {
        return Index(i, j, k);
    }
    return Index(i - 1, j, k);
}
__host__ __device__
Index Index::right() const
{
    if (i == xRes - 1)
    {
        return Index(i, j, k);
    }
    return Index(i + 1, j, k);
}

__device__
bool Obstacles::globalSmapler(float x, float y, float z) const
{
    x = fminf(fmaxf(0.0f, x), (float)(xRes - 1));
    y = fminf(fmaxf(0.0f, y), (float)(yRes - 1));
    z = fminf(fmaxf(0.0f, z), (float)(zRes - 1));

    int i = (int)(x);
    int j = (int)(y);
    int k = (int)(z);

    return data[i + (j + k * yRes) * xRes];
}

__host__ __device__
bool Obstacles::indexSampler(const int i, const int j, const int k) const
{
    return data[i + (j + k * yRes) * xRes];
}


__host__ __device__
bool Obstacles::indexSampler(const Index& index) const
{
    return data[index.globalOffset()];
}

__host__
void ScalarField::swap(ScalarField& other)
{
    thrust::swap(data, other.data);
}

__device__
float ScalarField::boxSampler(float x, float y, float z) const
{
    x -= 0.5f;
    y -= 0.5f;
    z -= 0.5f;

    x = fminf(fmaxf(0.0f, x), (float)(xRes - 1));
    y = fminf(fmaxf(0.0f, y), (float)(yRes - 1));
    z = fminf(fmaxf(0.0f, z), (float)(zRes - 1));

    int grid_x0 = (int)(x);
    int grid_y0 = (int)(y);
    int grid_z0 = (int)(z);

    float t_x = x - (float)(grid_x0);
    float t_y = y - (float)(grid_y0);
    float t_z = z - (float)(grid_z0);

    int grid_x1 = grid_x0 + 1;
    int grid_y1 = grid_y0 + 1;
    int grid_z1 = grid_z0 + 1;

    if (grid_x1 >= xRes) grid_x1 = xRes - 1;
    if (grid_y1 >= yRes) grid_y1 = yRes - 1;
    if (grid_z1 >= zRes) grid_z1 = zRes - 1;

    float s00 = (*this)(grid_x0, grid_y0, grid_z0);
    float s10 = (*this)(grid_x1, grid_y0, grid_z0);
    float s01 = (*this)(grid_x0, grid_y1, grid_z0);
    float s11 = (*this)(grid_x1, grid_y1, grid_z0);

    float tmp0 = lerp(s00, s10, t_x);
    float tmp1 = lerp(s01, s11, t_x);

    float z0 = lerp(tmp0, tmp1, t_y);

    s00 = (*this)(grid_x0, grid_y0, grid_z1);
    s10 = (*this)(grid_x1, grid_y0, grid_z1);
    s01 = (*this)(grid_x0, grid_y1, grid_z1);
    s11 = (*this)(grid_x1, grid_y1, grid_z1);

    tmp0 = lerp(s00, s10, t_x);
    tmp1 = lerp(s01, s11, t_x);

    float z1 = lerp(tmp0, tmp1, t_y);

    return lerp(z0, z1, t_z);
}


__host__ __device__ 
const float& ScalarField::operator()(const int i, const int j, const int k) const
{
    assert(i < xRes);
    assert(j < yRes);
    assert(k < zRes);
    return *(data.get() + i + (j + k * yRes) * xRes);
}

__host__ __device__
float& ScalarField::operator()(const int i, const int j, const int k)
{
    assert(i < xRes);
    assert(j < yRes);
    assert(k < zRes);
    return *(data.get() + i + (j + k * yRes) * xRes);
}

__host__ __device__
const float& ScalarField::operator()(const Index& index) const
{
    return *(data.get() + index.globalOffset());
}

__host__ __device__
float& ScalarField::operator()(const Index& index)
{
    return *(data.get() + index.globalOffset());
}


__host__
void uField::swap(uField& other)
{
    thrust::swap(data, other.data);
}

__device__
float uField::boxSampler(float x, float y, float z) const
{
    y -= 0.5f;
    z -= 0.5f;

    x = fminf(fmaxf(0.0f, x), (float)xRes);
    y = fminf(fmaxf(0.0f, y), (float)(yRes - 1));
    z = fminf(fmaxf(0.0f, z), (float)(zRes - 1));

    int grid_x0 = (int)(x);
    int grid_y0 = (int)(y);
    int grid_z0 = (int)(z);

    float t_x = x - (float)(grid_x0);
    float t_y = y - (float)(grid_y0);
    float t_z = z - (float)(grid_z0);

    int grid_x1 = grid_x0 + 1;
    int grid_y1 = grid_y0 + 1;
    int grid_z1 = grid_z0 + 1;

    if (grid_x1 >= xRes + 1) grid_x1 = xRes;
    if (grid_y1 >= yRes) grid_y1 = yRes - 1;
    if (grid_z1 >= zRes) grid_z1 = zRes - 1;

    float s00 = (*this)(grid_x0, grid_y0, grid_z0);
    float s10 = (*this)(grid_x1, grid_y0, grid_z0);
    float s01 = (*this)(grid_x0, grid_y1, grid_z0);
    float s11 = (*this)(grid_x1, grid_y1, grid_z0);

    float tmp0 = lerp(s00, s10, t_x);
    float tmp1 = lerp(s01, s11, t_x);

    float z0 = lerp(tmp0, tmp1, t_y);

    s00 = (*this)(grid_x0, grid_y0, grid_z1);
    s10 = (*this)(grid_x1, grid_y0, grid_z1);
    s01 = (*this)(grid_x0, grid_y1, grid_z1);
    s11 = (*this)(grid_x1, grid_y1, grid_z1);

    tmp0 = lerp(s00, s10, t_x);
    tmp1 = lerp(s01, s11, t_x);

    float z1 = lerp(tmp0, tmp1, t_y);

    return lerp(z0, z1, t_z);
}

__host__ __device__ 
const float& uField::operator()(const int i, const int j, const int k) const
{
    assert(i < xRes + 1);
    assert(j < yRes);
    assert(k < zRes);
    return *(data.get() + i + (j + k * yRes) * (xRes + 1));
}

__host__ __device__ 
float& uField::operator()(const int i, const int j, const int k)
{
    assert(i < xRes + 1);
    assert(j < yRes);
    assert(k < zRes);
    return *(data.get() + i + (j + k * yRes) * (xRes + 1));
}


__host__
void vField::swap(vField& other)
{
    thrust::swap(data, other.data);
}

__device__
float vField::boxSampler(float x, float y, float z) const
{
    x -= 0.5f;
    z -= 0.5f;

    x = fminf(fmaxf(0.0f, x), (float)(xRes - 1));
    y = fminf(fmaxf(0.0f, y), (float)yRes);
    z = fminf(fmaxf(0.0f, z), (float)(zRes - 1));

    int grid_x0 = (int)(x);
    int grid_y0 = (int)(y);
    int grid_z0 = (int)(z);

    float t_x = x - (float)(grid_x0);
    float t_y = y - (float)(grid_y0);
    float t_z = z - (float)(grid_z0);

    int grid_x1 = grid_x0 + 1;
    int grid_y1 = grid_y0 + 1;
    int grid_z1 = grid_z0 + 1;

    if (grid_x1 >= xRes) grid_x1 = xRes - 1;
    if (grid_y1 >= yRes + 1) grid_y1 = yRes;
    if (grid_z1 >= zRes) grid_z1 = zRes - 1;

    float s00 = (*this)(grid_x0, grid_y0, grid_z0);
    float s10 = (*this)(grid_x1, grid_y0, grid_z0);
    float s01 = (*this)(grid_x0, grid_y1, grid_z0);
    float s11 = (*this)(grid_x1, grid_y1, grid_z0);

    float tmp0 = lerp(s00, s10, t_x);
    float tmp1 = lerp(s01, s11, t_x);

    float z0 = lerp(tmp0, tmp1, t_y);

    s00 = (*this)(grid_x0, grid_y0, grid_z1);
    s10 = (*this)(grid_x1, grid_y0, grid_z1);
    s01 = (*this)(grid_x0, grid_y1, grid_z1);
    s11 = (*this)(grid_x1, grid_y1, grid_z1);

    tmp0 = lerp(s00, s10, t_x);
    tmp1 = lerp(s01, s11, t_x);

    float z1 = lerp(tmp0, tmp1, t_y);

    return lerp(z0, z1, t_z);
}

__host__ __device__ 
const float& vField::operator()(const int i, const int j, const int k) const
{
    assert(i < xRes);
    assert(j < yRes + 1);
    assert(k < zRes);
    return *(data.get() + i + (j + k * (yRes + 1)) * xRes);
}


__host__ __device__ 
float& vField::operator()(const int i, const int j, const int k)
{
    assert(i < xRes);
    assert(j < yRes + 1);
    assert(k < zRes);
    return *(data.get() + i + (j + k * (yRes + 1)) * xRes);
}


__host__
void wField::swap(wField& other)
{
    thrust::swap(data, other.data);
}

__device__
float wField::boxSampler(float x, float y, float z) const
{
    x -= 0.5f;
    y -= 0.5f;

    x = fminf(fmaxf(0.0f, x), (float)(xRes - 1));
    y = fminf(fmaxf(0.0f, y), (float)(yRes - 1));
    z = fminf(fmaxf(0.0f, z), (float)zRes);

    int grid_x0 = (int)(x);
    int grid_y0 = (int)(y);
    int grid_z0 = (int)(z);

    float t_x = x - (float)(grid_x0);
    float t_y = y - (float)(grid_y0);
    float t_z = z - (float)(grid_z0);

    int grid_x1 = grid_x0 + 1;
    int grid_y1 = grid_y0 + 1;
    int grid_z1 = grid_z0 + 1;

    if (grid_x1 >= xRes) grid_x1 = xRes - 1;
    if (grid_y1 >= yRes) grid_y1 = yRes - 1;
    if (grid_z1 >= zRes + 1) grid_z1 = zRes;

    float s00 = (*this)(grid_x0, grid_y0, grid_z0);
    float s10 = (*this)(grid_x1, grid_y0, grid_z0);
    float s01 = (*this)(grid_x0, grid_y1, grid_z0);
    float s11 = (*this)(grid_x1, grid_y1, grid_z0);

    float tmp0 = lerp(s00, s10, t_x);
    float tmp1 = lerp(s01, s11, t_x);

    float z0 = lerp(tmp0, tmp1, t_y);

    s00 = (*this)(grid_x0, grid_y0, grid_z1);
    s10 = (*this)(grid_x1, grid_y0, grid_z1);
    s01 = (*this)(grid_x0, grid_y1, grid_z1);
    s11 = (*this)(grid_x1, grid_y1, grid_z1);

    tmp0 = lerp(s00, s10, t_x);
    tmp1 = lerp(s01, s11, t_x);

    float z1 = lerp(tmp0, tmp1, t_y);

    return lerp(z0, z1, t_z);
}

__host__ __device__
const float& wField::operator()(const int i, const int j, const int k) const
{
    assert(i < xRes);
    assert(j < yRes);
    assert(k < zRes + 1);
    return *(data.get() + i + (j + k * yRes) * xRes);
}


__host__ __device__
float& wField::operator()(const int i, const int j, const int k)
{
    assert(i < xRes);
    assert(j < yRes);
    assert(k < zRes + 1);
    return *(data.get() + i + (j + k * yRes) * xRes);
}