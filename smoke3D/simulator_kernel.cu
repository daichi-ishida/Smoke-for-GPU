#include "simulator.h"

#include <thrust/execution_policy.h>

#include <algorithm>

#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_cudaVS.h"

#include "constants.h"


__global__
void calculateBuoyancy_k(ScalarField densityField, ScalarField temperatureField, ScalarField forceField)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    float density = densityField(x, y, z);
    float temperature = temperatureField(x, y, z);

    int dy = y - SOURCE_CENTER_Y;
    int dz = z - SOURCE_CENTER_Z;
    int d2yz = dy * dy + dz * dz;
    if(x >= SOURCE_MARGIN_X && x < SOURCE_MARGIN_X + SOURCE_SIZE_X && d2yz < SOURCE_RADIUS_YZ * SOURCE_RADIUS_YZ)
    {
        density = INIT_DENSITY;
        temperature = INIT_TEMPERATURE;
    }

    dy = y - (yRes - SOURCE_CENTER_Y);
    dz = z - (zRes - SOURCE_CENTER_Z);
    d2yz = dy * dy + dz * dz;
    if(x >= SOURCE_MARGIN_X && x < SOURCE_MARGIN_X + SOURCE_SIZE_X && d2yz < SOURCE_RADIUS_YZ * SOURCE_RADIUS_YZ)
    {
        density = INIT_DENSITY;
        temperature = INIT_COLD;
    }

    float force = ALPHA * density - BETA * temperature;

    densityField(x, y, z) = density;
    temperatureField(x, y, z) = temperature;
    forceField(x, y, z) = force;
}

__global__ 
void addForces_k(const ScalarField forceField, vField v, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (y < yRes - 1)
    {
        float vel_y = v(x, y + 1, z);
        vel_y += 0.5f * dt * (forceField(x, y, z) + forceField(x, y + 1, z));
        v(x, y + 1, z) = vel_y;
    }
}

__global__
void setRhs_k(const Obstacles obstacles, const uField u0, const vField v0, const wField w0, ScalarField divergenceField, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // set rhs
    float U[6];
    U[0] = -w0(x, y, z);
    U[1] = -v0(x, y, z);
    U[2] = -u0(x, y, z);
    U[3] = u0(x + 1, y, z);
    U[4] = v0(x, y + 1, z);
    U[5] = w0(x, y, z + 1);

    Index index(x, y, z);

    if (z == 0 || obstacles.indexSampler(index.front())) U[0] = 0.0f;

    if (y == 0 || obstacles.indexSampler(index.top())) U[1] = 0.0f;

    if (x == 0) U[2] = -INFLOW;
    else if (obstacles.indexSampler(index.left())) U[2] = 0.0f;

    if (x == xRes - 1) U[3] = INFLOW;
    else if (obstacles.indexSampler(index.right())) U[3] = 0.0f;

    if (y == yRes - 1 || obstacles.indexSampler(index.bottom())) U[4] = 0.0f;

    if (z == zRes - 1 || obstacles.indexSampler(index.back())) U[5] = 0.0f;

    float divergence = RHO * DX * (U[0] + U[1] + U[2] + U[3] + U[4] + U[5]) / dt;
    if (obstacles.indexSampler(index))
    {
        divergence = 0.0f;
    }

    divergenceField(x, y, z) = divergence;
}


__global__
void extrapolateU_k(uField src_u, const Obstacles obstacles, uField dst_u)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    float sum_u = 0.0f;
    int count = 0;
    float avg_u;

    uField *src_p = &src_u;
    uField *dst_p = &dst_u;

    for(int i = 0; i < 3; ++i)
    {
        if(z > 0) // front
        {
            sum_u += src_p->operator()(x, y, z-1);
            ++count;
        }
        if(y > 0) // top
        {
            sum_u += src_p->operator()(x, y-1, z);
            ++count;
        }
        if(x > 0) // left
        {
            sum_u += src_p->operator()(x-1, y, z);
            ++count;
        }
    
        //right
        sum_u += src_p->operator()(x+1, y, z);
        ++count;
    
        if(y < yRes - 1) // bottom
        {
            sum_u += src_p->operator()(x, y+1, z);
            ++count;
        }

        if(z < zRes - 1) // back
        {
            sum_u += src_p->operator()(x, y, z+1);
            ++count;
        }
    
        avg_u = sum_u / (float)count;
        
        Index index(x, y, z);
        if(obstacles.indexSampler(index.left()) || obstacles.indexSampler(index))
        {
            dst_p->operator()(x, y, z) = avg_u;
        }
        else
        {
            dst_p->operator()(x, y, z) = src_p->operator()(x, y, z);
        }

        sum_u = 0.0f;
        count = 0;
        if(x == 0) // 1 more line for u
        {
            Index index_edge(xRes-1, y, z);

            if(z > 0) // front
            {
                sum_u += src_p->operator()(xRes, y, z-1);
                ++count;
            }
    
            if(y > 0) // top
            {
                sum_u += src_p->operator()(xRes, y-1, z);
                ++count;
            }

            // left
            sum_u += src_p->operator()(xRes-1, y, z);
            ++count;
    
            if(y < yRes - 1) // bottom
            {
                sum_u += src_p->operator()(xRes, y+1, z);
                ++count;
            }

            if(z < zRes - 1) // back
            {
                sum_u += src_p->operator()(xRes, y, z+1);
                ++count;
            }
        
            avg_u = sum_u / (float)count;
        
            if(obstacles.indexSampler(index_edge) || obstacles.indexSampler(index_edge.right()))
            {
                dst_p->operator()(xRes, y, z) = avg_u;
            }
            else
            {
                dst_p->operator()(xRes, y, z) = src_p->operator()(xRes, y, z);
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
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    float sum_v = 0.0f;
    int count = 0;
    float avg_v;

    vField *src_p = &src_v;
    vField *dst_p = &dst_v;

    for(int i = 0; i < 3; ++i)
    {
        if(z > 0) // front
        {
            sum_v += src_p->operator()(x, y, z-1);
            ++count;
        }
        if(y > 0) // top
        {
            sum_v += src_p->operator()(x, y-1, z);
            ++count;
        }
        if(x > 0) // left
        {
            sum_v += src_p->operator()(x-1, y, z);
            ++count;
        }
    
        if(x < xRes - 1) // right
        {
            sum_v += src_p->operator()(x+1, y, z);
            ++count;
        }

        // bottom
        sum_v += src_p->operator()(x, y+1, z);
        ++count;

        if(z < zRes - 1) // back
        {
            sum_v += src_p->operator()(x, y, z+1);
            ++count;
        }
    
        avg_v = sum_v / (float)count;
        
        Index index(x, y, z);
        if(obstacles.indexSampler(index.top()) || obstacles.indexSampler(index))
        {
            dst_p->operator()(x, y, z) = avg_v;
        }
        else
        {
            dst_p->operator()(x, y, z) = src_p->operator()(x, y, z);
        }

        sum_v = 0.0f;
        count = 0;
        if(y == 0) // 1 more line for v
        {
            Index index_edge(x, yRes-1, z);
    
            if(z > 0) // front
            {
                sum_v += src_p->operator()(x, yRes, z-1);
                ++count;
            }

            // top
            sum_v += src_p->operator()(x, yRes-1, z);
            ++count;

            if(x > 0) // left
            {
                sum_v += src_p->operator()(x-1, yRes, z);
                ++count;
            }

            if(x < xRes - 1) // right
            {
                sum_v += src_p->operator()(x+1, yRes, z);
                ++count;
            }

            if(z < zRes - 1) // back
            {
                sum_v += src_p->operator()(x, yRes, z+1);
                ++count;
            }
        
            avg_v = sum_v / (float)count;
        
            if(obstacles.indexSampler(index_edge) || obstacles.indexSampler(index_edge.bottom()))
            {
                dst_p->operator()(x, yRes, z) = avg_v;
            }
            else
            {
                dst_p->operator()(x, yRes, z) = src_p->operator()(x, yRes, z);
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
void extrapolateW_k(wField src_w, const Obstacles obstacles, wField dst_w)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    float sum_w = 0.0f;
    int count = 0;
    float avg_w;

    wField *src_p = &src_w;
    wField *dst_p = &dst_w;

    for(int i = 0; i < 3; ++i)
    {
        if(z > 0) // front
        {
            sum_w += src_p->operator()(x, y, z-1);
            ++count;
        }
        if(y > 0) // top
        {
            sum_w += src_p->operator()(x, y-1, z);
            ++count;
        }
        if(x > 0) // left
        {
            sum_w += src_p->operator()(x-1, y, z);
            ++count;
        }
    
        if(x < xRes - 1) // right
        {
            sum_w += src_p->operator()(x+1, y, z);
            ++count;
        }

        if(y < yRes - 1) // bottom
        {
            sum_w += src_p->operator()(x, y+1, z);
            ++count;
        }

        // back
        sum_w += src_p->operator()(x, y, z+1);
        ++count;
    
        avg_w = sum_w / (float)count;
        
        Index index(x, y, z);
        if(obstacles.indexSampler(index.front()) || obstacles.indexSampler(index))
        {
            dst_p->operator()(x, y, z) = avg_w;
        }
        else
        {
            dst_p->operator()(x, y, z) = src_p->operator()(x, y, z);
        }

        sum_w = 0.0f;
        count = 0;
        if(z == 0) // 1 more line for w
        {
            Index index_edge(x, y, zRes-1);
    
            // front
            sum_w += src_p->operator()(x, y, zRes-1);
            ++count;

            if(y > 0) // top
            {
                sum_w += src_p->operator()(x, y-1, zRes);
                ++count;
            }

            if(x > 0) // left
            {
                sum_w += src_p->operator()(x-1, y, zRes);
                ++count;
            }

            if(x < xRes - 1) // right
            {
                sum_w += src_p->operator()(x+1, y, zRes);
                ++count;
            }

            if(y < yRes - 1) // bottom
            {
                sum_w += src_p->operator()(x, y+1, zRes);
                ++count;
            }

            avg_w = sum_w / (float)count;
        
            if(obstacles.indexSampler(index_edge) || obstacles.indexSampler(index_edge.back()))
            {
                dst_p->operator()(x, y, zRes) = avg_w;
            }
            else
            {
                dst_p->operator()(x, y, zRes) = src_p->operator()(x, y, zRes);
            }
        }
        __syncthreads();
    
        // swap src and dst
        wField *tmp = src_p;
        src_p = dst_p;
        dst_p = tmp;
    }
}

__global__
void advectScalar_k(const uField u, const vField v, const wField w, const ScalarField src_densityField, const ScalarField src_temperatureField,
    ScalarField dst_densityField, ScalarField dst_temperatureField, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // Mac-Cormack, linear interpolation
    float d_n, d_np1_hat, d_n_hat;
    float t_n, t_np1_hat, t_n_hat;

    // cell-centered pos
    float pos_x = (float)x + 0.5f;
    float pos_y = (float)y + 0.5f;
    float pos_z = (float)z + 0.5f;

    // get scalar here
    d_n = src_densityField(x, y, z);
    t_n = src_temperatureField(x, y, z);

    // get velocity here
    float vel_x = u.boxSampler(pos_x, pos_y, pos_z);
    float vel_y = v.boxSampler(pos_x, pos_y, pos_z);
    float vel_z = w.boxSampler(pos_x, pos_y, pos_z);

    pos_x -= dt * vel_x / DX;
    pos_y -= dt * vel_y / DX;
    pos_z -= dt * vel_z / DX;

    // get density and temperature
    d_np1_hat = src_densityField.boxSampler(pos_x, pos_y, pos_z);
    t_np1_hat = src_temperatureField.boxSampler(pos_x, pos_y, pos_z);

    // get velocity for backward advection
    vel_x = u.boxSampler(pos_x, pos_y, pos_z);
    vel_y = v.boxSampler(pos_x, pos_y, pos_z);
    vel_z = w.boxSampler(pos_x, pos_y, pos_z);

    pos_x += dt * vel_x / DX;
    pos_y += dt * vel_y / DX;
    pos_z += dt * vel_z / DX;

    // get density and temperature
    d_n_hat = src_densityField.boxSampler(pos_x, pos_y, pos_z);
    t_n_hat = src_temperatureField.boxSampler(pos_x, pos_y, pos_z);

    dst_densityField(x, y, z) = fma(0.5f, (d_n - d_n_hat), d_np1_hat);
    dst_temperatureField(x, y, z) = fma(0.5f, (t_n - t_n_hat), t_np1_hat);
}

__global__
void advectU_k(const uField src_u, const vField src_v, const wField src_w, uField dst_u, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // Mac-Cormack, linear interpolation
    //-----------------------------------------
    // 0 to xThreadDim - 1, 0 to yThreadDim - 1
    //-----------------------------------------
    float u_n, u_np1_hat, u_n_hat;

    float pos_x = (float)x;
    float pos_y = (float)y + 0.5f;
    float pos_z = (float)z + 0.5f;

    u_n = src_u(x, y, z);
    float vel_x = u_n;
    float vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
    float vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

    pos_x -= dt * vel_x / DX;
    pos_y -= dt * vel_y / DX;
    pos_z -= dt * vel_z / DX;

    u_np1_hat = src_u.boxSampler(pos_x, pos_y, pos_z);
    vel_x = u_np1_hat;
    vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
    vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

    pos_x += dt * vel_x / DX;
    pos_y += dt * vel_y / DX;
    pos_z += dt * vel_z / DX;

    u_n_hat = src_u.boxSampler(pos_x, pos_y, pos_z);

    dst_u(x, y, z) = fma(0.5f, (u_n - u_n_hat), u_np1_hat);

    //----------------------------------
    // xThreadDim, yThreadDim
    //----------------------------------
    if (x == 0) // 1 more line for u
    {
        // For no warp divergence, y -> x, z -> y
        pos_x = (float)xRes;
        pos_y = (float)y + 0.5f;
        pos_z = (float)z + 0.5f;

        u_n = src_u(xRes, y, z);
        vel_x = u_n;
        vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
        vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

        pos_x -= dt * vel_x / DX;
        pos_y -= dt * vel_y / DX;
        pos_z -= dt * vel_z / DX;

        u_np1_hat = src_u.boxSampler(pos_x, pos_y, pos_z);
        vel_x = u_np1_hat;
        vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
        vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

        pos_x += dt * vel_x / DX;
        pos_y += dt * vel_y / DX;
        pos_z += dt * vel_z / DX;

        u_n_hat = src_u.boxSampler(pos_x, pos_y, pos_z);

        dst_u(xRes, y, z) = fma(0.5f, (u_n - u_n_hat), u_np1_hat);
    }
    __syncthreads();
}

__global__
void advectV_k(const uField src_u, const vField src_v, const wField src_w, vField dst_v, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // Mac-Cormack, linear interpolation
    //-----------------------------------------
    // 0 to xThreadDim - 1, 0 to yThreadDim - 1
    //-----------------------------------------
    float v_n, v_np1_hat, v_n_hat;

    float pos_x = (float)x + 0.5f;
    float pos_y = (float)y;
    float pos_z = (float)z + 0.5f;

    v_n = src_v(x, y, z);
    float vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
    float vel_y = v_n;
    float vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

    pos_x -= dt * vel_x / DX;
    pos_y -= dt * vel_y / DX;
    pos_z -= dt * vel_z / DX;

    v_np1_hat = src_v.boxSampler(pos_x, pos_y, pos_z);
    vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
    vel_y = v_np1_hat;
    vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

    pos_x += dt * vel_x / DX;
    pos_y += dt * vel_y / DX;
    pos_z += dt * vel_z / DX;

    v_n_hat = src_v.boxSampler(pos_x, pos_y, pos_z);

    dst_v(x, y, z) = fma(0.5f, (v_n - v_n_hat), v_np1_hat);

    //----------------------------------
    // xThreadDim, yThreadDim
    //----------------------------------
    if (y == 0) // 1 more line for v
    {
        pos_x = (float)x + 0.5f;
        pos_y = (float)yRes;
        pos_z = (float)z + 0.5f;

        v_n = src_v(x, yRes, z);
        vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
        vel_y = v_n;
        vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

        pos_x -= dt * vel_x / DX;
        pos_y -= dt * vel_y / DX;
        pos_z -= dt * vel_z / DX;

        v_np1_hat = src_v.boxSampler(pos_x, pos_y, pos_z);
        vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
        vel_y = v_np1_hat;
        vel_z = src_w.boxSampler(pos_x, pos_y, pos_z);

        pos_x += dt * vel_x / DX;
        pos_y += dt * vel_y / DX;
        pos_z += dt * vel_z / DX;

        v_n_hat = src_v.boxSampler(pos_x, pos_y, pos_z);

        dst_v(x, yRes, z) = fma(0.5f, (v_n - v_n_hat), v_np1_hat);
    }
    __syncthreads();
}

__global__
void advectW_k(const uField src_u, const vField src_v, const wField src_w, wField dst_w, const float dt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // Mac-Cormack, linear interpolation
    //-----------------------------------------
    // 0 to xThreadDim - 1, 0 to yThreadDim - 1
    //-----------------------------------------
    float w_n, w_np1_hat, w_n_hat;

    float pos_x = (float)x + 0.5f;
    float pos_y = (float)y + 0.5f;
    float pos_z = (float)z;

    w_n = src_w(x, y, z);
    float vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
    float vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
    float vel_z = w_n;

    pos_x -= dt * vel_x / DX;
    pos_y -= dt * vel_y / DX;
    pos_z -= dt * vel_z / DX;

    w_np1_hat = src_w.boxSampler(pos_x, pos_y, pos_z);
    vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
    vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
    vel_z = w_np1_hat;

    pos_x += dt * vel_x / DX;
    pos_y += dt * vel_y / DX;
    pos_z += dt * vel_z / DX;

    w_n_hat = src_w.boxSampler(pos_x, pos_y, pos_z);

    dst_w(x, y, z) = fma(0.5f, (w_n - w_n_hat), w_np1_hat);

    //----------------------------------
    // xThreadDim, yThreadDim
    //----------------------------------
    if (z == 0) // 1 more line for v
    {
        pos_x = (float)x + 0.5f;
        pos_y = (float)y + 0.5f;
        pos_z = (float)zRes;

        w_n = src_w(x, y, zRes);
        vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
        vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
        vel_z = w_n;

        pos_x -= dt * vel_x / DX;
        pos_y -= dt * vel_y / DX;
        pos_z -= dt * vel_z / DX;

        w_np1_hat = src_w.boxSampler(pos_x, pos_y, pos_z);
        vel_x = src_u.boxSampler(pos_x, pos_y, pos_z);
        vel_y = src_v.boxSampler(pos_x, pos_y, pos_z);
        vel_z = w_np1_hat;

        pos_x += dt * vel_x / DX;
        pos_y += dt * vel_y / DX;
        pos_z += dt * vel_z / DX;

        w_n_hat = src_w.boxSampler(pos_x, pos_y, pos_z);

        dst_w(x, y, zRes) = fma(0.5f, (w_n - w_n_hat), w_np1_hat);
    }
    __syncthreads();
}


__global__
void blockMaxSpeed(const uField u0, const vField v0, const wField w0, thrust::device_ptr<float> speed)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int share_offset = threadIdx.x + (threadIdx.y + threadIdx.z * yThreadDim) * xThreadDim;
    int block_offset = blockIdx.x + (blockIdx.y + blockIdx.z * yBlockMaxDim) * xBlockMaxDim;

    __shared__ float share_speed[xThreadDim * yThreadDim * zThreadDim];

    // cell-centered pos
    float pos_x = (float)x + 0.5f;
    float pos_y = (float)y + 0.5f;
    float pos_z = (float)z + 0.5f;

    // get velocity here
    float vel_x = u0.boxSampler(pos_x, pos_y, pos_z);
    float vel_y = v0.boxSampler(pos_x, pos_y, pos_z);
    float vel_z = w0.boxSampler(pos_x, pos_y, pos_z);

    share_speed[share_offset] = sqrtf(vel_x * vel_x + vel_y * vel_y + vel_z * vel_z);

    float max_speed = thrust::reduce(thrust::seq, share_speed, share_speed + xThreadDim * yThreadDim * zThreadDim, 0.0f, thrust::maximum<float>());

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

    // CALL_KERNEL(blockMaxSpeed, blocks, threads)(m_data->u0, m_data->v0, m_data->w0, m_data->d_speed.data());
    // cudaDeviceSynchronize();
    // float max_speed = thrust::reduce(thrust::device, m_data->d_speed.begin(), m_data->d_speed.end());

    // float dt = CFL * DX / max_speed;
    // m_data->dt = dt;
    // printf("max_speed = %f, dt = %f\n", max_speed, dt);
}


void Simulator::update()
{
    CALL_KERNEL(calculateBuoyancy_k, blocks, threads)(m_data->density0, m_data->temperature0, m_data->force_y);
    CALL_KERNEL(addForces_k, blocks, threads)(m_data->force_y, m_data->v0, m_data->dt);
    buoyancy_timer.stop();

    solver_timer.start();
    CALL_KERNEL(setRhs_k, blocks, threads)(m_data->obstacles, m_data->u0, m_data->v0, m_data->w0, m_data->divergence, m_data->dt);

    // solve poisson equation with CG
    cg();
    solver_timer.stop();

    advection_timer.start();
    CALL_KERNEL(extrapolateU_k, blocks, threads)(m_data->u0, m_data->obstacles, m_data->u);
    CALL_KERNEL(extrapolateV_k, blocks, threads)(m_data->v0, m_data->obstacles, m_data->v);
    CALL_KERNEL(extrapolateW_k, blocks, threads)(m_data->w0, m_data->obstacles, m_data->w);

    m_data->u0.swap(m_data->u);
    m_data->v0.swap(m_data->v);
    m_data->w0.swap(m_data->w);

    CALL_KERNEL(advectScalar_k, blocks, threads)(m_data->u0, m_data->v0, m_data->w0, m_data->density0, m_data->temperature0, m_data->density, m_data->temperature, m_data->dt);

    CALL_KERNEL(advectU_k, blocks, threads)(m_data->u0, m_data->v0, m_data->w0, m_data->u, m_data->dt);
    CALL_KERNEL(advectV_k, blocks, threads)(m_data->u0, m_data->v0, m_data->w0, m_data->v, m_data->dt);
    CALL_KERNEL(advectW_k, blocks, threads)(m_data->u0, m_data->v0, m_data->w0, m_data->w, m_data->dt);
    advection_timer.stop();

    m_data->density0.swap(m_data->density);
    m_data->temperature0.swap(m_data->temperature);
    m_data->u0.swap(m_data->u);
    m_data->v0.swap(m_data->v);
    m_data->w0.swap(m_data->w);

    decideTimeStep();
}

void Simulator::printSimBreakdown()
{
    float buoyancy = buoyancy_timer.getAVG();
    float solver = solver_timer.getAVG();
    float advection = advection_timer.getAVG();

    float total_time = buoyancy + solver + + advection;

    printf("total: %f ms/step\n", total_time);
    printf("buoyancy: %f ms/step | %f %%\n", buoyancy, 100.0f * buoyancy / total_time);
    printf("solver: %f ms/step | %f %%\n", solver, 100.0f * solver / total_time);
    printf("advection: %f ms/step | %f %%\n", advection, 100.0f * advection / total_time);

}

void Simulator::saveSimBreakdown()
{
    FILE* outputfile;

    std::string str = "log/breakdown.txt";

    outputfile = fopen(str.c_str(), "w");
    if (outputfile == NULL)
    {
        printf("cannot open\n");
        exit(1);
    }
    float buoyancy = buoyancy_timer.getAVG();
    float solver = solver_timer.getAVG();
    float advection = advection_timer.getAVG();

    float total_time = buoyancy + solver + advection;

    fprintf(outputfile, "total: %f ms/step\n", total_time);
    fprintf(outputfile, "buoyancy: %f ms/step | %f %%\n", buoyancy, 100.0f * buoyancy / total_time);
    fprintf(outputfile, "solver: %f ms/step | %f %%\n", solver, 100.0f * solver / total_time);
    fprintf(outputfile, "advection: %f ms/step | %f %%\n", advection, 100.0f * advection / total_time);

    fclose(outputfile);
}