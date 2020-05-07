
#include "FieldStructure.h"

#include <thrust/execution_policy.h>
#include <thrust/swap.h>

#include "helper_cuda.h"
#include "helper_math.h"

#include <cassert>

#include "constants.h"




__host__ __device__
Index::Index(int _i, int _j) : i(_i), j(_j)
{
    assert(i < xRes);
    assert(j < yRes);
}

inline __host__ __device__
int Index::globalOffset() const
{
    return i + j * xRes;
}

__host__ __device__
Index Index::top() const
{
    if (j == 0)
    {
        return Index(i, j);
    }
    return Index(i, j - 1);
}
__host__ __device__
Index Index::bottom() const
{
    if (j == yRes - 1)
    {
        return Index(i, j);
    }
    return Index(i, j + 1);
}
__host__ __device__
Index Index::left() const
{
    if (i == 0)
    {
        return Index(i, j);
    }
    return Index(i - 1, j);
}
__host__ __device__
Index Index::right() const
{
    if (i == xRes - 1)
    {
        return Index(i, j);
    }
    return Index(i + 1, j);
}

__host__ __device__
char Obstacles::indexSampler(const int i, const int j) const
{
    return data[i + j * xRes];
}


__host__ __device__
char Obstacles::indexSampler(const Index& index) const
{
    return data[index.globalOffset()];
}

__host__
void ScalarField::swap(ScalarField& other)
{
    thrust::swap(data, other.data);
}

__device__
float ScalarField::boxSampler(float x, float y) const
{
    x -= 0.5f;
    y -= 0.5f;

    x = fminf(fmaxf(0.0f, x), (float)(xRes - 1));
    y = fminf(fmaxf(0.0f, y), (float)(yRes - 1));

    int grid_x0 = (int)(x);
    int grid_y0 = (int)(y);

    float t_x = x - (float)(grid_x0);
    float t_y = y - (float)(grid_y0);

    int grid_x1 = grid_x0 + 1;
    int grid_y1 = grid_y0 + 1;

    if (grid_x1 >= xRes) grid_x1 = xRes - 1;
    if (grid_y1 >= yRes) grid_y1 = yRes - 1;

    float s00 = (*this)(grid_x0, grid_y0);
    float s10 = (*this)(grid_x1, grid_y0);
    float s01 = (*this)(grid_x0, grid_y1);
    float s11 = (*this)(grid_x1, grid_y1);

    float tmp0 = lerp(s00, s10, t_x);
    float tmp1 = lerp(s01, s11, t_x);

    return lerp(tmp0, tmp1, t_y);
}


__host__ __device__ 
const float& ScalarField::operator()(const int i, const int j) const
{
    assert(i < xRes);
    assert(j < yRes);
    return *(data.get() + i + j * xRes);
}

__host__ __device__
float& ScalarField::operator()(const int i, const int j)
{
    assert(i < xRes);
    assert(j < yRes);
    return *(data.get() + i + j * xRes);
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
float uField::boxSampler(float x, float y) const
{
    y -= 0.5f;

    x = fminf(fmaxf(0.0f, x), (float)xRes);
    y = fminf(fmaxf(0.0f, y), (float)(yRes - 1));

    int grid_x0 = (int)(x);
    int grid_y0 = (int)(y);

    float t_x = x - (float)(grid_x0);
    float t_y = y - (float)(grid_y0);

    int grid_x1 = grid_x0 + 1;
    int grid_y1 = grid_y0 + 1;

    if (grid_x1 >= xRes + 1) grid_x1 = xRes;
    if (grid_y1 >= yRes) grid_y1 = yRes - 1;

    float u00 = (*this)(grid_x0, grid_y0);
    float u10 = (*this)(grid_x1, grid_y0);
    float u01 = (*this)(grid_x0, grid_y1);
    float u11 = (*this)(grid_x1, grid_y1);

    float tmp0 = lerp(u00, u10, t_x);
    float tmp1 = lerp(u01, u11, t_x);

    return lerp(tmp0, tmp1, t_y);
}

__host__ __device__ 
const float& uField::operator()(const int i, const int j) const
{
    assert(i < xRes + 1);
    assert(j < yRes);
    return *(data.get() + i + j * (xRes+1));
}

__host__ __device__ 
float& uField::operator()(const int i, const int j)
{
    assert(i < xRes + 1);
    assert(j < yRes);
    return *(data.get() + i + j * (xRes + 1));
}


__host__
void vField::swap(vField& other)
{
    thrust::swap(data, other.data);
}

__device__
float vField::boxSampler(float x, float y) const
{
    x -= 0.5f;

    x = fminf(fmaxf(0.0f, x), (float)(xRes - 1));
    y = fminf(fmaxf(0.0f, y), (float)yRes);

    int grid_x0 = (int)(x);
    int grid_y0 = (int)(y);

    float t_x = x - (float)(grid_x0);
    float t_y = y - (float)(grid_y0);

    int grid_x1 = grid_x0 + 1;
    int grid_y1 = grid_y0 + 1;

    if (grid_x1 >= xRes) grid_x1 = xRes - 1;
    if (grid_y1 >= yRes + 1) grid_y1 = yRes;

    float v00 = (*this)(grid_x0, grid_y0);
    float v10 = (*this)(grid_x1, grid_y0);
    float v01 = (*this)(grid_x0, grid_y1);
    float v11 = (*this)(grid_x1, grid_y1);

    float tmp0 = lerp(v00, v10, t_x);
    float tmp1 = lerp(v01, v11, t_x);

    return lerp(tmp0, tmp1, t_y);
}

__host__ __device__ 
const float& vField::operator()(const int i, const int j) const
{
    assert(i < xRes);
    assert(j < yRes + 1);
    return *(data.get() + i + j * xRes);
}


__host__ __device__ 
float& vField::operator()(const int i, const int j)
{
    assert(i < xRes);
    assert(j < yRes + 1);
    return *(data.get() + i + j * xRes);
}