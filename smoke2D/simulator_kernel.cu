#include "simulator.h"

#include <thrust/execution_policy.h>

#include <algorithm>

#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_cudaVS.h"

#include "constants.h"


__global__
void resetObstacles(const float cx, const float cy, Obstacles obstacles)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float global_x = x + 0.5f;
    float global_y = y + 0.5f;

    float dx = global_x - cx;
    float dy = global_y - cy;

    float r2 = dx * dx + dy * dy;

    int offset = x + y * xRes;
    if(r2 <= R2)
    {
        obstacles.data[offset] = true;
    }
    else
    {
        obstacles.data[offset] = false;
    }
}



__global__
void calculateBuoyancy_k(const ScalarField densityField, const ScalarField temperatureField, ScalarField forceField)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float density = densityField(x, y);
    float temperature = temperatureField(x, y);

    float force = ALPHA * density - BETA * temperature;

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
void setRhs_k(const Obstacles obstacles, const uField u0, const vField v0, ScalarField divergenceField, const float dt, const float obstacle_u, const float obstacle_v)
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
    else if (obstacles.indexSampler(index.top())) U[0] = -obstacle_v;

    if (x == 0) U[1] = 0.0f;
    else if (obstacles.indexSampler(index.left())) U[1] = -obstacle_u;

    if (x == xRes - 1) U[2] = 0.0f;
    else if (obstacles.indexSampler(index.right())) U[2] = obstacle_u;

    if (y == yRes - 1) U[3] = -INFLOW;
    else if (obstacles.indexSampler(index.bottom())) U[3] = obstacle_v;

    float divergence = RHO * DX * (U[0] + U[1] + U[2] + U[3]) / dt;
    if (obstacles.indexSampler(index))
    {
        divergence = 0.0f;
    }

    divergenceField(x, y) = divergence;
}

__global__
void extrapolateU_k(uField src_u, const Obstacles obstacles, uField dst_u)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float sum_u = 0.0f;
    int count = 0;
    float avg_u;

    uField *src_p = &src_u;
    uField *dst_p = &dst_u;

    for(int i = 0; i < 3; ++i)
    {
        if(y > 0) // top
        {
            sum_u += src_p->operator()(x, y-1);
            ++count;
        }
        if(x > 0) // left
        {
            sum_u += src_p->operator()(x-1, y);
            ++count;
        }
    
        //right
        sum_u += src_p->operator()(x+1, y);
        ++count;
    
        if(y < yRes - 1) // bottom
        {
            sum_u += src_p->operator()(x, y+1);
            ++count;
        }
    
        avg_u = sum_u / (float)count;
        
        Index index(x, y);
        if(obstacles.indexSampler(index.left()) || obstacles.indexSampler(index))
        {
            dst_p->operator()(x, y) = avg_u;
        }
        else
        {
            dst_p->operator()(x, y) = src_p->operator()(x, y);
        }

        sum_u = 0.0f;
        count = 0;
        if(y == 0) // 1 more line for u
        {
            // For no warp divergence, x is used instead of y so [xRes, x]
            Index index_edge(xRes-1, x);
    
            if(x > 0) // top
            {
                sum_u += src_p->operator()(xRes, x-1);
                ++count;
            }

            // left
            sum_u += src_p->operator()(xRes-1, x);
            ++count;
    
            if(x < yRes - 1) // bottom
            {
                sum_u += src_p->operator()(xRes, x+1);
                ++count;
            }
        
            avg_u = sum_u / (float)count;
        
            if(obstacles.indexSampler(index_edge) || obstacles.indexSampler(index_edge.right()))
            {
                dst_p->operator()(xRes, x) = avg_u;
            }
            else
            {
                dst_p->operator()(xRes, x) = src_p->operator()(xRes, x);
            }
        }
        __syncthreads();
    
        // swap src and dst
        uField *tmp = src_p;
        src_p = dst_p;
        dst_p = tmp;
    }
}


__global__
void extrapolateV_k(vField src_v, const Obstacles obstacles, vField dst_v)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float sum_v = 0.0f;
    int count = 0;
    float avg_v;

    vField *src_p = &src_v;
    vField *dst_p = &dst_v;

    for(int i = 0; i < 3; ++i)
    {
        if(y > 0) // top
        {
            sum_v += src_p->operator()(x, y-1);
            ++count;
        }
        if(x > 0) // left
        {
            sum_v += src_p->operator()(x-1, y);
            ++count;
        }
    
        if(x < xRes - 1) //right
        {
            sum_v += src_p->operator()(x+1, y);
            ++count;
        }

        // bottom
        sum_v += src_p->operator()(x, y+1);
        ++count;
    
        avg_v = sum_v / (float)count;
        
        Index index(x, y);
        if(obstacles.indexSampler(index.top()) || obstacles.indexSampler(index))
        {
            dst_p->operator()(x, y) = avg_v;
        }
        else
        {
            dst_p->operator()(x, y) = src_p->operator()(x, y);
        }

        sum_v = 0.0f;
        count = 0;
        if(y == 0) // 1 more line for v
        {
            Index index_edge(x, yRes-1);
    
            // top
            sum_v += src_p->operator()(x, yRes-1);
            ++count;

            if(x > 0) // left
            {
                sum_v += src_p->operator()(x-1, yRes);
                ++count;
            }

            if(x < xRes - 1) // right
            {
                sum_v += src_p->operator()(x+1, yRes);
                ++count;
            }
        
            avg_v = sum_v / (float)count;
        
            if(obstacles.indexSampler(index_edge) || obstacles.indexSampler(index_edge.bottom()))
            {
                dst_p->operator()(x, yRes) = avg_v;
            }
            else
            {
                dst_p->operator()(x, yRes) = src_p->operator()(x, yRes);
            }
        }
        __syncthreads();
    
        // swap src and dst
        vField *tmp = src_p;
        src_p = dst_p;
        dst_p = tmp;
    }
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

    //CALL_KERNEL(blockMaxSpeed, blocks, threads)(m_data->u0, m_data->v0, m_data->d_speed.data());
    //cudaDeviceSynchronize();
    //float max_speed = thrust::reduce(thrust::device, m_data->d_speed.begin(), m_data->d_speed.end());

    //float dt = CFL * DX / max_speed;
    //m_data->dt = dt;
    //printf("max_speed = %f, dt = %f\n", max_speed, dt);
}


void Simulator::update()
{
    if(m_data->t <= ANIMATION_CHANGE_TIME)
    {
        m_data->obstacle_cx = COLLISION_CENTER_X;
        m_data->obstacle_cy = COLLISION_CENTER_Y - ANIMATION_AMP * std::sin(ANIMATION_OMEGA * m_data->t);
        m_data->obstacle_u = 0.0f;
        m_data->obstacle_v = - ANIMATION_AMP * DX * ANIMATION_OMEGA * std::cos(ANIMATION_OMEGA * m_data->t);
    }
    else
    {
        m_data->obstacle_cx = COLLISION_CENTER_X - ANIMATION_AMP * std::sin(ANIMATION_OMEGA * m_data->t);
        m_data->obstacle_cy = COLLISION_CENTER_Y;
        m_data->obstacle_u = - ANIMATION_AMP * DX * ANIMATION_OMEGA * std::cos(ANIMATION_OMEGA * m_data->t);
        m_data->obstacle_v = 0.0f;
    }

    CALL_KERNEL(resetObstacles, blocks, threads)(m_data->obstacle_cx, m_data->obstacle_cy, m_data->obstacles);

    CALL_KERNEL(calculateBuoyancy_k, blocks, threads)(m_data->density0, m_data->temperature0, m_data->force_y);
    CALL_KERNEL(addForces_k, blocks, threads)(m_data->force_y, m_data->v0, m_data->dt);
    CALL_KERNEL(setRhs_k, blocks, threads)(m_data->obstacles, m_data->u0, m_data->v0, m_data->divergence, m_data->dt, m_data->obstacle_u, m_data->obstacle_v);

    // solve poisson equation with CG
    cg();

    CALL_KERNEL(extrapolateU_k, blocks, threads)(m_data->u0, m_data->obstacles, m_data->u);
    CALL_KERNEL(extrapolateV_k, blocks, threads)(m_data->v0, m_data->obstacles, m_data->v);

    m_data->u0.swap(m_data->u);
    m_data->v0.swap(m_data->v);

    CALL_KERNEL(advectScalar_k, blocks, threads)(m_data->u0, m_data->v0, m_data->density0, m_data->temperature0, m_data->density, m_data->temperature, m_data->dt);

    CALL_KERNEL(advectU_k, blocks, threads)(m_data->u0, m_data->v0, m_data->u, m_data->dt);
    CALL_KERNEL(advectV_k, blocks, threads)(m_data->u0, m_data->v0, m_data->v, m_data->dt);

    m_data->density0.swap(m_data->density);
    m_data->temperature0.swap(m_data->temperature);
    m_data->u0.swap(m_data->u);
    m_data->v0.swap(m_data->v);

    decideTimeStep();
}
