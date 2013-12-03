#pragma once

#include "Essential.h"
#include "Material.h"
#include "Ray.h"

struct Sphere
{
	CUDA_HOST_DEVICE Sphere(void)
	{
	}

	Sphere(const rtrt::Material &mat, const Vector &center, float radius)
		: mat(mat), center(center), radius(radius), radius2(radius * radius)
	{
	}


	CUDA_HOST_DEVICE ~Sphere(void)
	{
	}

	CUDA_HOST_DEVICE bool intersect(const Ray &r, float &distance) const
	{
		Vector len = center - r.origin;
		float tca = Vector::dotProduct(len, r.direction);

		if (tca < 0.0f)
			return false;

		float d2 = Vector::dotProduct(len, len) - tca * tca;

		if (d2 > radius2)
			return false;

		float thc = sqrt(radius2 - d2);

		distance = (tca - thc) > 0.0f ? tca - thc : tca + thc;

		return true;
	}

	CUDA_HOST_DEVICE void getNormalAtPoint(const Vector &point, Vector &normal) const
	{
		normal = (point - center).normalize();
	}

	CUDA_HOST_DEVICE Sphere& operator=(const Sphere& other)
	{
		mat = other.mat;
		center = other.center;
		radius = other.radius;
		radius2 = other.radius2;
		return *this;
	}

	rtrt::Material mat;
	Vector center;
	float radius, radius2;
};

