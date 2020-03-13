#pragma once
#include "Smoke.h"

#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "constants.h"


struct Simulator
{
public:
    Simulator(std::unique_ptr<Smoke>& data)
        : m_data(data)
        , blocks(xBlockMaxDim, yBlockMaxDim, zBlockMaxDim)
        , threads(xThreadDim, yThreadDim, zThreadDim)
    {}

    ~Simulator() {};

    void update();

private:
    void cg();
    void decideTimeStep();

    dim3 blocks;
    dim3 threads;

    std::unique_ptr<Smoke>& m_data;
};