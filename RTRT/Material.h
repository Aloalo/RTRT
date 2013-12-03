#pragma once

#include "Vector.h"

namespace rtrt
{
	struct Material
	{
		CUDA_HOST_DEVICE Material(void)
		{
		}

		Material(const Vector &color, float transparency, float ior, 
			float absorption, float reflectivity, float diff, float spec, short shininess)
			: color(color), ior(ior), reflectivity(reflectivity), 
			transparency(transparency), diff(diff), spec(spec), shininess(shininess)
		{
		}

		Material(const Vector &color, float reflectivity, float diff, float spec, short shininess)
			: color(color), ior(0.0f), reflectivity(reflectivity), transparency(0.0f), diff(diff), spec(spec), shininess(shininess)
		{
		}

		Material(const Vector &color, float diff, float spec, short shininess)
			: color(color), ior(0.0f), reflectivity(0.0f), transparency(0.0f), diff(diff), spec(spec), shininess(shininess)
		{

		}

		CUDA_HOST_DEVICE ~Material(void)
		{
		}

		CUDA_HOST_DEVICE Material& operator=(const Material &other)
		{
			color = other.color;
			ior = other.ior;
			reflectivity = other.reflectivity;
			transparency = other.transparency;
			diff = other.diff;
			spec = other.spec;
			shininess = other.shininess;
			return *this;
		}

		Vector color;
		float ior;
		float reflectivity, transparency;
		float diff, spec;
		short shininess;
	};
}

