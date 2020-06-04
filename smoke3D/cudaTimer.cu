#include "cudaTimer.h"


cudaTimer::cudaTimer(): duration(0.0f), counter(0)
{
	cudaEventCreate(&_start);
	cudaEventCreate(&_stop);
}

cudaTimer::~cudaTimer()
{
	cudaEventDestroy(_start);
	cudaEventDestroy(_stop);
}

void cudaTimer::start()
{
	cudaEventRecord(_start);
}

void cudaTimer::stop()
{
	cudaEventRecord(_stop);
	cudaEventSynchronize(_stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, _start, _stop);
	duration += milliseconds;
	++counter;
}

float cudaTimer::getAVG()
{
	return duration / static_cast<float>(counter);
}