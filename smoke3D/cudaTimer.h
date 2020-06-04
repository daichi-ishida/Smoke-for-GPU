#pragma once

#include <cuda_runtime.h>

struct cudaTimer
{
	cudaTimer();
	~cudaTimer();

	void start();
	void stop();

	float getAVG();

	cudaEvent_t _start, _stop;
	float duration;
	int counter;
};