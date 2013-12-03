#pragma once

#include "macros.h"
#include <cmath>

class Vector
{
public:
	float x, y, z;

	CUDA_HOST_DEVICE Vector(void)
	{
	}

	CUDA_HOST_DEVICE Vector(float t)
		: x(t), y(t), z(t)
	{
	}

	CUDA_HOST_DEVICE Vector(float x, float y, float z)
		: x(x), y(y), z(z)
	{
	}

	CUDA_HOST_DEVICE ~Vector(void)
	{
	}

	CUDA_HOST_DEVICE float& operator[](int id)
	{
		return *(&x+id);
	}

	CUDA_HOST_DEVICE float operator[](int id) const
	{
		return *(&x+id);
	}

	CUDA_HOST_DEVICE Vector operator+(const Vector &v) const
	{
		return Vector(x + v.x, y + v.y, z + v.z);
	}

	CUDA_HOST_DEVICE Vector& operator+=(const Vector &v)
	{
		return *this = *this + v;
	}

	CUDA_HOST_DEVICE Vector operator-(const Vector &v) const
	{
		return Vector(x - v.x, y - v.y, z - v.z);
	}

	CUDA_HOST_DEVICE Vector& operator-=(const Vector &v)
	{
		return *this = *this - v;
	}

	CUDA_HOST_DEVICE Vector operator*(float t) const
	{
		return Vector(t * x, t * y, t * z);
	}

	CUDA_HOST_DEVICE Vector operator*(const Vector &v) const
	{
		return Vector(v.x * x, v.y * y, v.z * z);
	}

	CUDA_HOST_DEVICE Vector operator/(float t) const
	{
		return Vector(x / t, y / t, z / t);
	}

	CUDA_HOST_DEVICE Vector& operator/=(float t)
	{
		return *this = *this / t;
	}

	CUDA_HOST_DEVICE Vector& operator*=(float t)
	{
		return *this = *this * t;
	}

	CUDA_HOST_DEVICE bool operator<(const Vector &v) const
	{
		return x < v.x || x == v.x && (y < v.y || y == v.y && z < v.z);
	}

	CUDA_HOST_DEVICE bool operator>(const Vector &v) const
	{
		return x > v.x || x == v.x && (y > v.y || y == v.y && z > v.z);
	}

	CUDA_HOST_DEVICE bool operator<=(const Vector &v) const
	{
		return !(*this > v);
	}

	CUDA_HOST_DEVICE bool operator>=(const Vector &v) const
	{
		return !(*this < v);
	}

	CUDA_HOST_DEVICE Vector operator-(void) const
	{
		return Vector(-x, -y, -z);
	}

	CUDA_HOST_DEVICE float length(void) const
	{
		return sqrt(length2());
	}

	CUDA_HOST_DEVICE float length2(void) const
	{
		return x * x + y * y + z * z;
	}

	CUDA_HOST_DEVICE float distance(const Vector &v) const
	{
		return (*this - v).length();
	}

	CUDA_HOST_DEVICE float distance2(const Vector &v) const
	{
		return (*this - v).length2();
	}

	CUDA_HOST_DEVICE Vector normalized(void) const
	{
		if(!x && !y && !z)
			return *this;
		return *this / length();
	}

	CUDA_HOST_DEVICE Vector& normalize(void)
	{
		if(!x && !y && !z)
			return *this;
		return *this /= length();
	}

	CUDA_HOST_DEVICE bool operator==(const Vector &v) const
	{
		return x == v.x && y == v.y && z == v.z;
	}

	CUDA_HOST_DEVICE bool operator!=(const Vector &v) const
	{
		return x != v.x || y != v.y || z != v.z;
	}

	CUDA_HOST_DEVICE void rotateX(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		float t = y;
		y = y * c - z * s;
		z = t * s + z * c;
	}

	CUDA_HOST_DEVICE void rotateY(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		float t = x;
		x = x * c + z * s;
		z = -t * s + z * c;
	}

	CUDA_HOST_DEVICE void rotateZ(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		float t = x;
		x = x * c - y * s;
		y = t * s + y * c;
	}

	CUDA_HOST_DEVICE static float dotProduct(const Vector &a, const Vector &b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	CUDA_HOST_DEVICE static Vector crossProduct(const Vector &a, const Vector &b)
	{
		return Vector
			(
			a.y * b.z - b.y * a.z,
			b.x * a.z - a.x * b.z,
			a.x * b.y - b.x * a.y
			);
	}

	CUDA_HOST_DEVICE static Vector pairwiseProduct(const Vector &a, const Vector &b)
	{
		return Vector(a.x * b.x, a.y * b.y, a.z * b.z);
	}

	CUDA_HOST_DEVICE static float mixedProduct(const Vector &a, const Vector &b, const Vector &c)
	{
		return dotProduct(a, crossProduct(b, c));
	}
};

//float dotProduct(const Vector &a, const Vector &b);
//Vector crossProduct(const Vector &a, const Vector &b);
//Vector pairwiseProduct(const Vector &a, const Vector &b);
//float mixedProduct(const Vector &a, const Vector &b, const Vector &c);
//

//CUDA_HOST_DEVICE Vector operator*(float t, const Vector &v)
//{
//	return v * t;
//}
