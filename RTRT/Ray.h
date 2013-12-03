#pragma once

#include "Vector.h"

class Ray
{
public:
	CUDA_HOST_DEVICE Ray(void)
	{
	}

	CUDA_HOST_DEVICE Ray(const Vector &origin, const Vector &direction)
		: origin(origin), direction(direction)
	{
	}

	CUDA_HOST_DEVICE ~Ray(void)
	{
	}

	Vector origin, direction;
};