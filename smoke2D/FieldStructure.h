#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>


struct Index
{
    int i, j;
    __host__ __device__ Index(int _i, int _j);

    inline __host__ __device__ int globalOffset() const;

    __host__ __device__ Index top() const;
    __host__ __device__ Index bottom() const;
    __host__ __device__ Index left() const;
    __host__ __device__ Index right() const;
};


struct Obstacles
{
    thrust::device_ptr<bool> data;

    __host__ __device__ bool indexSampler(const int i, const int j) const;
    __host__ __device__ bool indexSampler(const Index& index) const;
};


struct ScalarField
{
    thrust::device_ptr<float> data;

    __host__ void swap(ScalarField& other);

    __device__ float boxSampler(float x, float y) const;

    __host__ __device__ const float& operator()(const int i, const int j) const;
    __host__ __device__ float& operator()(const int i, const int j);

    __host__ __device__ const float& operator()(const Index& index) const;
    __host__ __device__ float& operator()(const Index& index);
};


struct uField
{
    thrust::device_ptr<float> data;

    __host__ void swap(uField& other);

    __device__ float boxSampler(float x, float y) const;

    __host__ __device__ const float& operator()(const int i, const int j) const;
    __host__ __device__ float& operator()(const int i, const int j);
};


struct vField
{
    thrust::device_ptr<float> data;

    __host__ void swap(vField& other);

    __device__ float boxSampler(float x, float y) const;

    __host__ __device__ const float& operator()(const int i, const int j) const;
    __host__ __device__ float& operator()(const int i, const int j);
};