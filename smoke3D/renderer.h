#pragma once
#include "camera.h"
#include "Smoke.h"

#include <thrust/device_vector.h>

#include <memory>

struct Renderer
{
    Renderer(std::unique_ptr<Smoke>& data);
    ~Renderer();

    void initialize();
    void render();
    void saveImage();

    thrust::host_vector<uchar3> h_image;
    thrust::device_vector<uchar3> d_image;

    std::unique_ptr<Smoke> &m_data;
    std::unique_ptr<Camera> host_camera;

    cudaArray *cuda_density;

private:
    void bindDensityTexture();
    void unbindDensityTexture();
    void assignTexture();
};