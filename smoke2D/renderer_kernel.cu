#include "Smoke.h"
#include "renderer.h"
#include "constants.h"

#include <cuda_gl_interop.h>

#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_cudaVS.h"


__global__ void render_k(uchar4* out_color, const ScalarField field, const Obstacles obstacles)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if ((x >= WIN_WIDTH) || (y >= WIN_HEIGHT)) return;

    float u = ((float)x + 0.5f) * ((float)xRes / (float)WIN_WIDTH);
    float v = ((float)y + 0.5f) * ((float)yRes / (float)WIN_HEIGHT);

    float value = 10.0f * field.boxSampler(u, v);

    u -= 0.5f;
    v -= 0.5f;

    u = fminf(fmaxf(0.0f, u), (float)(xRes - 1));
    v = fminf(fmaxf(0.0f, v), (float)(yRes - 1));

    int grid_x = (int)u;
    int grid_y = (int)v;

    float3 color = make_float3(0.0f);

    if(obstacles.indexSampler(grid_x, grid_y))
    {
        color.x = 0.3f;
    }

    color += 0.7f * make_float3(value);

    out_color[offset].x = fminf(1.0f, color.x) * 255;
    out_color[offset].y = fminf(1.0f, color.y) * 255;
    out_color[offset].z = fminf(1.0f, color.z) * 255;
    out_color[offset].w = 255;
}

Renderer::~Renderer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex_buffer);
    }

}

void Renderer::initialize()
{
    glEnable(GL_TEXTURE_2D);

    if (pbo)
    {
        printf("unregister buffer...");
        // unregister this buffer object from CUDA C
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex_buffer);
        printf("Done\n");
    }

    printf("generating buffer...");

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIN_WIDTH * WIN_HEIGHT * 4, 0, GL_DYNAMIC_DRAW_ARB);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

    // create texture for display
    glGenTextures(1, &tex_buffer);
    glBindTexture(GL_TEXTURE_2D, tex_buffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIN_WIDTH, WIN_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);
    getLastCudaError("Renderer initializing failed\n");

    printf("Done\n");
}

void Renderer::render()
{
    // map PBO to get CUDA device pointer
    uchar4* dev_ptr;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &num_bytes, cuda_pbo_resource));

    // clear image
    checkCudaErrors(cudaMemset(dev_ptr, 0, WIN_WIDTH * WIN_HEIGHT * 4));

    // call CUDA kernel, writing results to PBO
    dim3 blocks((WIN_WIDTH + 31) / 32, (WIN_HEIGHT + 31) / 32);
    dim3 render_threads(32, 32);

    CALL_KERNEL(render_k, blocks, render_threads)(dev_ptr, m_data->density0, m_data->obstacles);

    getLastCudaError("render failed\n");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // glDrawPixels is slow so use texture instead

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex_buffer);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIN_WIDTH, WIN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
}