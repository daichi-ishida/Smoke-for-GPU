#include "Smoke.h"
#include "renderer.h"
#include "constants.h"

#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_cudaVS.h"


__global__ void render_k(uchar3* out_color, const ScalarField field, const Obstacles obstacles)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if ((x >= WIN_WIDTH) || (y >= WIN_HEIGHT)) return;

    float u = ((float)x + 0.5f) * ((float)xRes / (float)WIN_WIDTH);
    float v = ((float)y + 0.5f) * ((float)yRes / (float)WIN_HEIGHT);

    float value = 20.0f * field.boxSampler(u, v);

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

    color += 1.0f * make_float3(value);

    out_color[offset].x = fminf(1.0f, color.x) * 255;
    out_color[offset].y = fminf(1.0f, color.y) * 255;
    out_color[offset].z = fminf(1.0f, color.z) * 255;
}

Renderer::~Renderer()
{
}

void Renderer::initialize()
{
    // allocate image data
    h_image.resize(WIN_WIDTH * WIN_HEIGHT);
    d_image.resize(WIN_WIDTH * WIN_HEIGHT);
}

void Renderer::render()
{
    // call CUDA kernel, writing results
    dim3 blocks((WIN_WIDTH + 31) / 32, (WIN_HEIGHT + 31) / 32);
    dim3 render_threads(32, 32);

    CALL_KERNEL(render_k, blocks, render_threads)(thrust::raw_pointer_cast(d_image.data()), m_data->density0, m_data->obstacles);
}