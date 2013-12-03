#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "Vector.h"

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if(code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort)
			exit(code);
	}
}

inline void gpuAssert(CUresult code, char *file, int line, bool abort = true)
{
	if(code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: Error code: %d %s %d\n", code, file, line);
		if (abort)
			exit(code);
	}
}

__device__ inline void print(const Vector &v)
{
	printf("[%3.2f, %3.2f %3.2f]\n", v.x, v.y, v.z);
}