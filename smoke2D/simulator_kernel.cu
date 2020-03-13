#include "simulator.h"

#include <thrust/execution_policy.h>

#include <algorithm>

#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_cudaVS.h"

#include "constants.h"


__global__
void calculateBuoyancy_k(const ScalarField densityField, const ScalarField temperatureField, ScalarField forceField)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float density = densityField(x, y);
    float temperature = temperatureField(x, y);

    float force = ALPHA * density - BETA * (temperature - T_AMBIENT);

    forceField(x, y) = force;
}

__global__ 
void addForces_k(const ScalarField forceField, vField v, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y < yRes - 1)
    {
        float vel_y = v(x, y + 1);
        vel_y += 0.5f * dt * (forceField(x, y) + forceField(x, y + 1));
        v(x, y + 1) = vel_y;
    }
}

__global__
void setRhs_k(const Obstacles obstacles, const uField u0, const vField v0, ScalarField divergenceField, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // set rhs
    float U[4];
    U[0] = -v0(x, y);
    U[1] = -u0(x, y);
    U[2] = u0(x + 1, y);
    U[3] = v0(x, y + 1);

    Index index(x, y);

    if (y == 0) U[0] = INFLOW;
    else if (obstacles.indexSampler(index.top())) U[0] = 0.0f;

    if (x == 0 || obstacles.indexSampler(index.left())) U[1] = 0.0f;

    if (x == xRes - 1 || obstacles.indexSampler(index.right())) U[2] = 0.0f;

    if (y == yRes - 1) U[3] = -INFLOW;
    else if (obstacles.indexSampler(index.bottom())) U[3] = 0.0f;

    float divergence = RHO * DX * (U[0] + U[1] + U[2] + U[3]) / dt;
    if (obstacles.indexSampler(index))
    {
        divergence = 0.0f;
    }

    divergenceField(x, y) = divergence;
}

__global__
void advectScalar_k(const uField u, const vField v, const ScalarField src_densityField, const ScalarField src_temperatureField,
    ScalarField dst_densityField, ScalarField dst_temperatureField, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Mac-Cormack, linear interpolation
    float d_n, d_np1_hat, d_n_hat;
    float t_n, t_np1_hat, t_n_hat;

    // cell-centered pos
    float pos_x = (float)x + 0.5f;
    float pos_y = (float)y + 0.5f;

    // get scalar here
    d_n = src_densityField(x, y);
    t_n = src_temperatureField(x, y);

    // get velocity here
    float vel_x = u.boxSampler(pos_x, pos_y);
    float vel_y = v.boxSampler(pos_x, pos_y);

    pos_x -= dt * vel_x / DX;
    pos_y -= dt * vel_y / DX;

    // get density and temperature
    d_np1_hat = src_densityField.boxSampler(pos_x, pos_y);
    t_np1_hat = src_temperatureField.boxSampler(pos_x, pos_y);

    // get velocity for backward advection
    vel_x = u.boxSampler(pos_x, pos_y);
    vel_y = v.boxSampler(pos_x, pos_y);

    pos_x += dt * vel_x / DX;
    pos_y += dt * vel_y / DX;

    // get density and temperature
    d_n_hat = src_densityField.boxSampler(pos_x, pos_y);
    t_n_hat = src_temperatureField.boxSampler(pos_x, pos_y);

    dst_densityField(x, y) = fma(0.5f, (d_n - d_n_hat), d_np1_hat);
    dst_temperatureField(x, y) = fma(0.5f, (t_n - t_n_hat), t_np1_hat);
}

__global__
void advectU_k(const uField src_u, const vField src_v, uField dst_u, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Mac-Cormack, linear interpolation
    //-----------------------------------------
    // 0 to xThreadDim - 1, 0 to yThreadDim - 1
    //-----------------------------------------
    float u_n, u_np1_hat, u_n_hat;

    float pos_x = (float)x;
    float pos_y = (float)y + 0.5f;

    u_n = src_u(x, y);
    float vel_x = u_n;
    float vel_y = src_v.boxSampler(pos_x, pos_y);

    pos_x -= dt * vel_x / DX;
    pos_y -= dt * vel_y / DX;

    u_np1_hat = src_u.boxSampler(pos_x, pos_y);
    vel_x = u_np1_hat;
    vel_y = src_v.boxSampler(pos_x, pos_y);

    pos_x += dt * vel_x / DX;
    pos_y += dt * vel_y / DX;

    u_n_hat = src_u.boxSampler(pos_x, pos_y);

    dst_u(x, y) = fma(0.5f, (u_n - u_n_hat), u_np1_hat);

    //----------------------------------
    // xThreadDim, yThreadDim
    //----------------------------------
    if (y == 0) // 1 more line for u
    {
        // For no warp divergence, y is used instead of x 
        pos_x = (float)xRes;
        pos_y = (float)x + 0.5f;

        u_n = src_u(xRes, x);
        vel_x = u_n;
        vel_y = src_v.boxSampler(pos_x, pos_y);

        pos_x -= dt * vel_x / DX;
        pos_y -= dt * vel_y / DX;

        u_np1_hat = src_u.boxSampler(pos_x, pos_y);
        vel_x = u_np1_hat;
        vel_y = src_v.boxSampler(pos_x, pos_y);

        pos_x += dt * vel_x / DX;
        pos_y += dt * vel_y / DX;

        u_n_hat = src_u.boxSampler(pos_x, pos_y);

        dst_u(xRes, x) = fma(0.5f, (u_n - u_n_hat), u_np1_hat);
    }
    __syncthreads();
}

__global__
void advectV_k(const uField src_u, const vField src_v, vField dst_v, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Mac-Cormack, linear interpolation
    //-----------------------------------------
    // 0 to xThreadDim - 1, 0 to yThreadDim - 1
    //-----------------------------------------
    float v_n, v_np1_hat, v_n_hat;

    float pos_x = (float)x + 0.5f;
    float pos_y = (float)y;

    v_n = src_v(x, y);
    float vel_x = src_u.boxSampler(pos_x, pos_y);
    float vel_y = v_n;

    pos_x -= dt * vel_x / DX;
    pos_y -= dt * vel_y / DX;

    v_np1_hat = src_v.boxSampler(pos_x, pos_y);
    vel_x = src_u.boxSampler(pos_x, pos_y);
    vel_y = v_np1_hat;

    pos_x += dt * vel_x / DX;
    pos_y += dt * vel_y / DX;

    v_n_hat = src_v.boxSampler(pos_x, pos_y);

    dst_v(x, y) = fma(0.5f, (v_n - v_n_hat), v_np1_hat);

    //----------------------------------
    // xThreadDim, yThreadDim
    //----------------------------------
    if (y == 0) // 1 more line for v
    {
        pos_x = (float)x + 0.5f;
        pos_y = (float)yRes;

        v_n = src_v(x, yRes);
        vel_x = src_u.boxSampler(pos_x, pos_y);
        vel_y = v_n;

        pos_x -= dt * vel_x / DX;
        pos_y -= dt * vel_y / DX;

        v_np1_hat = src_v.boxSampler(pos_x, pos_y);
        vel_x = src_u.boxSampler(pos_x, pos_y);
        vel_y = v_np1_hat;

        pos_x += dt * vel_x / DX;
        pos_y += dt * vel_y / DX;

        v_n_hat = src_v.boxSampler(pos_x, pos_y);

        dst_v(x, yRes) = fma(0.5f, (v_n - v_n_hat), v_np1_hat);
    }
    __syncthreads();
}


__global__
void blockMaxSpeed(const uField u0, const vField v0, thrust::device_ptr<float> speed)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int share_offset = threadIdx.x + threadIdx.y * xThreadDim;
    int block_offset = blockIdx.x + blockIdx.y * xBlockMaxDim;

    __shared__ float share_speed[xThreadDim * yThreadDim];

    // cell-centered pos
    float pos_x = (float)x + 0.5f;
    float pos_y = (float)y + 0.5f;

    // get velocity here
    float vel_x = u0.boxSampler(pos_x, pos_y);
    float vel_y = v0.boxSampler(pos_x, pos_y);

    share_speed[share_offset] = hypotf(vel_x, vel_y);

    float max_speed = thrust::reduce(thrust::seq, share_speed, share_speed + xThreadDim * yThreadDim, 0.0f, thrust::maximum<float>());

    if (share_offset == 0)
    {
        speed[block_offset] = max_speed;
    }
}

void Simulator::decideTimeStep()
{
    m_data->t += m_data->dt;
    if (m_data->t - m_data->next_shutter_time >= 0.0f)
    {
        m_data->isTimeToRender = true;
        m_data->setNextShutterTime();
    }

    CALL_KERNEL(blockMaxSpeed, blocks, threads)(m_data->u0, m_data->v0, m_data->d_speed.data());
    cudaDeviceSynchronize();
    float max_speed = thrust::reduce(thrust::device, m_data->d_speed.begin(), m_data->d_speed.end());

    float dt = CFL * DX / max_speed;
    m_data->dt = dt;
    printf("max_speed = %f, dt = %f\n", max_speed, dt);
}


void Simulator::update()
{
    CALL_KERNEL(calculateBuoyancy_k, blocks, threads)(m_data->density0, m_data->temperature0, m_data->force_y);
    CALL_KERNEL(addForces_k, blocks, threads)(m_data->force_y, m_data->v0, m_data->dt);
    CALL_KERNEL(setRhs_k, blocks, threads)(m_data->obstacles, m_data->u0, m_data->v0, m_data->divergence, m_data->dt);

    // solve poisson equation with CG
    cg();

    CALL_KERNEL(advectScalar_k, blocks, threads)(m_data->u0, m_data->v0, m_data->density0, m_data->temperature0, m_data->density, m_data->temperature, m_data->dt);

    CALL_KERNEL(advectU_k, blocks, threads)(m_data->u0, m_data->v0, m_data->u, m_data->dt);
    CALL_KERNEL(advectV_k, blocks, threads)(m_data->u0, m_data->v0, m_data->v, m_data->dt);

    m_data->density0.swap(m_data->density);
    m_data->temperature0.swap(m_data->temperature);
    m_data->u0.swap(m_data->u);
    m_data->v0.swap(m_data->v);

    decideTimeStep();
}
