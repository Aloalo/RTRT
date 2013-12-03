#pragma once

#include "Essential.h"
#include "Material.h"
#include "Ray.h"

struct Plane
{
	CUDA_HOST_DEVICE Plane(void)
	{
	}

	Plane(const rtrt::Material &mat, const Vector &n, const Vector &p)
		: mat(mat), n(n.normalized()), p(p)
	{

	}

	Plane(const rtrt::Material &mat, const Vector &t0, const Vector &t1, const Vector &t2)
		: mat(mat)
	{
		n = Vector::crossProduct(t0 - t1, t1 - t2).normalize();
		p = t2;
	}


	CUDA_HOST_DEVICE ~Plane(void)
	{
	}

	CUDA_HOST_DEVICE bool intersect(const Ray &r, float &distance) const
	{
		float t0 = Vector::dotProduct(n, p - r.origin);
		float t1 = Vector::dotProduct(r.direction, n);

		if(efl::abs(t0) <= efl::ZERO && efl::abs(t1) <= efl::ZERO)
		{
			distance = 0.0f;
			return true;
		}
		else if(efl::abs(t1) <= efl::ZERO || (t0 / t1) < 0)
			return false;
		else
		{
			distance = t0 / t1;
			return true;
		}
	}

	CUDA_HOST_DEVICE void getNormalAtPoint(const Vector &point, Vector &normal) const
	{
		normal = n;
	}

	CUDA_HOST_DEVICE Plane& operator=(const Plane &other)
	{
		mat = other.mat;
		n = other.n;
		p = other.p;
		return *this;
	}

	rtrt::Material mat;
	Vector n, p;
};

