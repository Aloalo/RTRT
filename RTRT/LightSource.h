#pragma once

#include "Vector.h"

struct LightSource
{
	CUDA_HOST_DEVICE LightSource(void)
	{
	}

	LightSource(const Vector &position, const Vector &intensity, const Vector &attenuation)
		: position(position), intensity(intensity), atten(attenuation)
	{

	}

	CUDA_HOST_DEVICE ~LightSource(void)
	{
	}

	CUDA_HOST_DEVICE Vector intensityAtPoint(const Vector &p) const
	{
		float d = position.distance(p);
		return intensity * (1.0f / (atten.x + atten.y * d + atten.z * d * d));
	}

	CUDA_HOST_DEVICE LightSource& operator=(const LightSource &other)
	{
		atten = other.atten;
		position = other.position;
		intensity = other.intensity;
		return *this;
	}

	Vector atten;
	Vector position;
	Vector intensity;
};

