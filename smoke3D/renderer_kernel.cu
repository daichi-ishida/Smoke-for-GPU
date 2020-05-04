#include "Smoke.h"
#include "renderer.h"
#include "constants.h"

#include "renderer.cuh"
#include "camera.h"

#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_cudaVS.h"


texture<float, 3, cudaReadModeElementType> densityTex;

__constant__ Camera c_camera;

__global__ void render_k(uchar3* d_output, const Obstacles obstacles)
{
    const float tstep = 2.0f / (float)DIM;
    float3 boxMin = make_float3(-2.0f, -1.0f, -1.0f);
    float3 boxMax = make_float3(2.0f, 1.0f, 1.0f);
    float3 albedo = make_float3(0.4f, 0.4f, 0.4f);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if ((x >= WIN_WIDTH) || (y >= WIN_HEIGHT)) return;

    Ray eyeRay = c_camera.generateRay(x, y);

    // find intersection with box to skip empty place
    float tnear, tfar;
    bool hit = isIntersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f; // clamp to near plane
    float t = tnear;
    float3 ray_wpos = eyeRay.o + eyeRay.d * tnear;
    float3 du = eyeRay.d * tstep;

    // float3 lightPos = make_float3(7.0f, -8.5f, 7.0f);
    float r = 7.0f;
    float verticalAngle = M_PI / 3.0f;
    float horizontalAngle = 0.0f;
    float3 lightPos = make_float3(r * sin(verticalAngle) * sin(horizontalAngle),
        -r * cos(verticalAngle),
        r * sin(verticalAngle) * cos(horizontalAngle));
    const float lightIntensity = 10.0f;

    // transmittance
    float3 T = make_float3(1.0f);
    float3 Lo = make_float3(0.0f);

    bool hitObstacle = false;
    bool hitWall = false;

    // while ray is in boundary box
    for (; ; t += tstep, ray_wpos += du)
    {
        if (t > tfar)
        {
            hitWall = true;
            break;
        }
        // world position to grid position
        float3 gridPos = convertToGridCoordinate(ray_wpos);

        // check intersection with obstacle
        if (obstacles.globalSmapler(gridPos.x, gridPos.y, gridPos.z))
        {
            hitObstacle = true;
            break;
        }

        // sample density
        float density = 200.0f * tex3D(densityTex, gridPos.x, gridPos.y, gridPos.z);

        // skip empty space
        if (density <= 0.0f) continue;
        float3 sigma_scatter = albedo * density;
        float3 sigma_absorption = make_float3(density) - sigma_scatter;
        float3 sigma_total = sigma_scatter + sigma_absorption;
        float3 omega = sigma_scatter / sigma_total;

        T *= expf(-sigma_total * tstep);
        if (T.x < 0.001f && T.y < 0.001f && T.z < 0.001f) break;

        float3 lightDir = normalize(lightPos - ray_wpos);
        float3 light_ray_wpos = ray_wpos;
        float3 dx = lightDir * tstep;

        // transmittance along light ray
        float3 lightT = make_float3(1.0f);

        for (; ; light_ray_wpos += dx)
        {
            // world position to grid position
            float3 light_gridPos = convertToGridCoordinate(light_ray_wpos);

            // check ray is in box
            if (!isInGridBox(light_gridPos)) break;

            // sample density
            float light_density = 200.0f * tex3D(densityTex, light_gridPos.x, light_gridPos.y, light_gridPos.z);

            // skip empty space
            if (light_density <= 0.0f) continue;
            float3 light_sigma_scatter = albedo * light_density;
            float3 light_sigma_absorption = make_float3(light_density) - light_sigma_scatter;
            float3 light_sigma_total = light_sigma_scatter + light_sigma_absorption;

            lightT *= expf(-light_sigma_total * tstep);
            if (lightT.x < 0.001f && lightT.y < 0.001f && lightT.z < 0.001f) break;
        }
        float3 Jss = lightIntensity * lightT * omega * mieScatter(lightDir, -eyeRay.d, 0.2f);
        Lo += T * sigma_total * Jss * tstep;
    }

    float alpha = 1.0f - grayScale(T);
    if (hitObstacle)
    {
        Lo += T * make_float3(0.2f, 0.2f, 0.3f);
        alpha = 1.0f;
    }
    else if (hitWall)
    {
        Lo += T * drawWall(ray_wpos, lightPos, -eyeRay.d);
        alpha = 1.0f;
    }

    Lo = fminf(Lo, make_float3(1.0f));

    d_output[offset].x = Lo.x * 255;
    d_output[offset].y = Lo.y * 255;
    d_output[offset].z = Lo.z * 255;
}

Renderer::~Renderer()
{
    unbindDensityTexture();
}

void Renderer::initialize()
{
    // allocate image data
    h_image.resize(WIN_WIDTH * WIN_HEIGHT);
    d_image.resize(WIN_WIDTH * WIN_HEIGHT);

    printf("setting camera...");
    host_camera = std::make_unique<Camera>(9.0f, 0.0f, M_PIf / 2.0f, 30.0f * M_PIf / 180.0f);
    checkCudaErrors(cudaMemcpyToSymbol(c_camera, host_camera.get(), sizeof(Camera)));
    printf("Done\n");

    bindDensityTexture();
}

void Renderer::render()
{
    // call CUDA kernel, writing results
    dim3 blocks((WIN_WIDTH+31)/32,(WIN_HEIGHT+31)/32);
    dim3 render_threads(32,32);

    // expand data for texture
    assignTexture();

    // rendering
    CALL_KERNEL(render_k, blocks, render_threads)(thrust::raw_pointer_cast(d_image.data()), m_data->obstacles);
}

void Renderer::bindDensityTexture()
{
    cudaChannelFormatDesc cdesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&cuda_density, &cdesc, make_cudaExtent(xRes, yRes, zRes)));
    getLastCudaError("cudaMallocArray failed\n");

    densityTex.filterMode = cudaFilterModeLinear;
    densityTex.addressMode[0] = cudaAddressModeClamp;
    densityTex.addressMode[1] = cudaAddressModeClamp;
    densityTex.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors(cudaBindTextureToArray(densityTex, cuda_density, cdesc));
    getLastCudaError("cudaBindTextureToArray failed\n");
}

void Renderer::unbindDensityTexture()
{
    checkCudaErrors(cudaUnbindTexture(densityTex));
}

void Renderer::assignTexture()
{
    cudaMemcpy3DParms parms = { 0 };

    parms.dstArray = cuda_density;
    parms.srcPtr = make_cudaPitchedPtr(thrust::raw_pointer_cast(m_data->density0.data), sizeof(float) * xRes, xRes, yRes);
    parms.extent = make_cudaExtent(xRes, yRes, zRes);
    parms.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&parms));
}
