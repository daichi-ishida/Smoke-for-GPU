#pragma once
#include "Smoke.h"

#define GLFW_INCLUDE_GLU
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory>

struct Renderer
{
    Renderer(std::unique_ptr<Smoke>& data);
    ~Renderer();

    void initialize();
    void render();
    void saveImage();

    GLuint pbo;
    GLuint tex_buffer;
    struct cudaGraphicsResource* cuda_pbo_resource;

    std::unique_ptr<Smoke>& m_data;
};