#include "Smoke.h"
#include "simulator.h"

#include <cmath>
#include <cassert>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "helper_cuda.h"
#include "helper_cudaVS.h"

#include "constants.h"


__global__
void Laplacian(const ScalarField xField, const Obstacles obstacles, ScalarField AxField)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    Index index(x, y, z);
    float Ax = 0.0f;
    if (!obstacles.indexSampler(index))
    {
        Index front = (obstacles.indexSampler(index.front())) ? index : index.front();
        Index top = (obstacles.indexSampler(index.top())) ? index : index.top();
        Index left = (obstacles.indexSampler(index.left())) ? index : index.left();
        Index right = (obstacles.indexSampler(index.right())) ? index : index.right();
        Index bottom = (obstacles.indexSampler(index.bottom())) ? index : index.bottom();
        Index back = (obstacles.indexSampler(index.back())) ? index : index.back();

        Ax = xField(front) + xField(top) + xField(left) + xField(right) + xField(bottom) + xField(back) - 6.0f * xField(index);
    }

    __syncthreads();

    AxField(x, y, z) = Ax;
}

__global__
void applyPressureGradient_k(const ScalarField pressureField, const Obstacles obstacles, uField u, vField v, wField w, float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    Index index(x, y, z);

    //// ### calculate pressure gradient ###

    // update u
    if (x < xRes - 1)
    {
        if (!obstacles.indexSampler(index) && !obstacles.indexSampler(index.right()))
        {
            float old_u = u(x + 1, y, z);
            u(x + 1, y, z) = old_u - dt * (pressureField(x + 1, y, z) - pressureField(x, y, z)) / (RHO * DX);
        }
        else
        {
            u(x + 1, y, z) = 0.0f;
        }
    }

    // update v
    if (y < yRes - 1)
    {
        if (!obstacles.indexSampler(index) && !obstacles.indexSampler(index.bottom()))
        {
            float old_v = v(x, y + 1, z);
            v(x, y + 1, z) = old_v - dt * (pressureField(x, y + 1, z) - pressureField(x, y, z)) / (RHO * DX);
        }
        else
        {
            v(x, y + 1, z) = 0.0f;
        }
    }

    // update w
    if (z < zRes - 1)
    {
        if (!obstacles.indexSampler(index) && !obstacles.indexSampler(index.back()))
        {
            float old_w = w(x, y, z + 1);
            w(x, y, z + 1) = old_w - dt * (pressureField(x, y, z + 1) - pressureField(x, y, z)) / (RHO * DX);
        }
        else
        {
            w(x, y, z + 1) = 0.0f;
        }
    }
    __syncthreads();
}

void Simulator::cg()
{
    std::vector<float> residuals;

    cublasHandle_t cublasHandle = 0;
    checkCudaErrors(cublasCreate(&cublasHandle));

    constexpr float one = 1.0f;
    float alpha = 0.0f, minus_alpha = 0.0f, beta = 0.0f, delta0 = 0.0f, delta = 0.0f;
    float residual_0 = 1.0f, residual_1 = 1.0f, relative_residual = 1.0f;
    int maxIdx = 0;

    int pressure_size = xRes * yRes * zRes;

    // thrust::fill(m_data->d_pressure_data.begin(), m_data->d_pressure_data.end(), 0.0f);
    cudaMemset(thrust::raw_pointer_cast(m_data->d_pressure_data.data()), 0, sizeof(float) * pressure_size);

    checkCudaErrors(cublasScopy(cublasHandle, pressure_size, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, thrust::raw_pointer_cast(m_data->d_direction_data.data()), 1));

    // find max residual
    checkCudaErrors(cublasIsamax(cublasHandle, pressure_size, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, &maxIdx));
    residual_0 = m_data->d_divergence_data[maxIdx - 1];
    residual_0 = std::abs(residual_0);

    // delta0 = dot(r, r)
    checkCudaErrors(cublasSdot(cublasHandle, pressure_size, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, &delta0));

    int iteration = 0;
    while (iteration < MAX_ITER)
    {
        // q = Ad
        CALL_KERNEL(Laplacian, blocks, threads)(m_data->direction, m_data->obstacles, m_data->Ax);

        // alpha = delta0 / (d * Ad);
        checkCudaErrors(cublasSdot(cublasHandle, pressure_size, thrust::raw_pointer_cast(m_data->d_direction_data.data()), 1, thrust::raw_pointer_cast(m_data->d_Ax_data.data()), 1, &alpha));
        if (alpha)
        {
            alpha = delta0 / alpha;
            minus_alpha = -alpha;
        }

        // x = x + alpha * d
        checkCudaErrors(cublasSaxpy(cublasHandle, pressure_size, &alpha, thrust::raw_pointer_cast(m_data->d_direction_data.data()), 1, thrust::raw_pointer_cast(m_data->d_pressure_data.data()), 1));

        // r = r - alpha * q
        checkCudaErrors(cublasSaxpy(cublasHandle, pressure_size, &minus_alpha, thrust::raw_pointer_cast(m_data->d_Ax_data.data()), 1, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1));

        // find max residual
        checkCudaErrors(cublasIsamax(cublasHandle, pressure_size, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, &maxIdx));
        residual_1 = m_data->d_divergence_data[maxIdx - 1];
        residual_1 = std::abs(residual_1);

        // check convergence L_inf norm
        relative_residual = residual_1 / residual_0;
        residuals.push_back(relative_residual);
        if (relative_residual < TOLERANCE)
        {
            printf("SUCCESS: Convergence\n");
            ++iteration;
            break;
        }

        // delta = dot(r, r)
        checkCudaErrors(cublasSdot(cublasHandle, pressure_size, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, &delta));

        beta = delta / delta0;
        delta0 = delta;

        // d = r + beta * d
        checkCudaErrors(cublasSscal(cublasHandle, pressure_size, &beta, thrust::raw_pointer_cast(m_data->d_direction_data.data()), 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, pressure_size, &one, thrust::raw_pointer_cast(m_data->d_divergence_data.data()), 1, thrust::raw_pointer_cast(m_data->d_direction_data.data()), 1));

        ++iteration;
    }
    printf("iteration        : %3d\nrelative residual: %e\n", iteration, relative_residual);

    checkCudaErrors(cublasDestroy(cublasHandle));

    {
        FILE* outputfile;

        static int count = 0;
        char filename[1024];

        snprintf(filename, sizeof(filename), "log/residuals%03d.csv", count++);

        outputfile = fopen(filename, "w");
        if (outputfile == NULL)
        {
            printf("cannot open\n");
            exit(1);
        }

        for (const auto& residual : residuals)
        {
            fprintf(outputfile, "%f\n", residual);
        }

        fclose(outputfile);
    }

    // project out solution
    CALL_KERNEL(applyPressureGradient_k, blocks, threads)(m_data->pressure, m_data->obstacles, m_data->u0, m_data->v0, m_data->w0, m_data->dt);
}
