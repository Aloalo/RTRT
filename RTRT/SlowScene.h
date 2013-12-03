#pragma once

#include <vector>
#include "IntersectableObjects.h"
#include "LightSource.h"


#define MAX_SPHERES 104
#define MAX_LIGHTS 5
#define MAX_PLANES 6

struct SlowScene
{
	SlowScene(void)
	{
	}

	SlowScene(const std::vector<Sphere> &sphereVec, const std::vector<Plane> &planeVec, const std::vector<LightSource> &lightsVec, Vector &backgroundColor, float spaceIoR)
		: backgroundColor(backgroundColor), spaceIoR(spaceIoR)
	{
		numSpheres = sphereVec.size();
		numPlanes = planeVec.size();
		numLights = lightsVec.size();

		for(int i = 0; i < numSpheres; ++i)
			spheres[i] = sphereVec[i];

		for(int i = 0; i < numPlanes; ++i)
			planes[i] = planeVec[i];

		for(int i = 0; i < numLights; ++i)
			lights[i] = lightsVec[i];
	}

	SlowScene(const Vector &backgroundColor, float spaceIoR)
		: backgroundColor(backgroundColor), spaceIoR(spaceIoR)
	{
	}

	~SlowScene(void)
	{
	}

	CUDA_HOST_DEVICE bool hitObject(const Ray &r, float &distance, Vector &nhit, rtrt::Material &mat) const
	{
		distance = efl::INF;
		float t;
		int best, id = -1;

		#pragma unroll 6
		for(int i = 0; i < numPlanes; ++i)
			if(planes[i].intersect(r, t) && t < distance)
			{
				distance = t;
				id = 0;
				best = i;
			}

		#pragma unroll 10
		for(int i = 0; i < numSpheres; ++i)
			if(spheres[i].intersect(r, t) && t < distance)
			{
				distance = t;
				id = 1;
				best = i;
			}

		if(id == 0)
		{
			mat = planes[best].mat;
			planes[best].getNormalAtPoint(r.origin + r.direction * distance, nhit);
		}
		else if(id == 1)
		{
			mat = spheres[best].mat;
			spheres[best].getNormalAtPoint(r.origin + r.direction * distance, nhit);
		}
		return distance < efl::INF;
	}

	CUDA_HOST_DEVICE bool hitObjectFast(const Ray &r, float &distance) const
	{
		distance = efl::INF;
		float t;
		#pragma unroll 6
		for(int i = 0; i < numPlanes; ++i)
			if(planes[i].intersect(r, t) && t < distance)
				distance = t;
		
		#pragma unroll 10
		for(int i = 0; i < numSpheres; ++i)
			if(spheres[i].intersect(r, t) && t < distance)
				distance = t;

		return distance < efl::INF;
	}

	CUDA_HOST_DEVICE Vector shade(const Ray &r, const rtrt::Material &mat, const Vector &nhit, const Vector &phit) const
	{
		Vector outColor(0.0f);
		for(int i = 0; i < numLights; ++i)
		{
			float transmission = 1.0f;
			Vector lightDirection = (lights[i].position - phit).normalize();
			float tnear = phit.distance(lights[i].position);
			float tmp;
			if(hitObjectFast(Ray(phit + nhit * efl::BIAS, lightDirection), tmp) && tmp < tnear)
				transmission = 0.0f;

			if(transmission > efl::ZERO)//shade
			{
				Vector lightIntensity = lights[i].intensityAtPoint(phit);
				if(mat.diff > efl::ZERO)//diffuse
					outColor += lightIntensity * efl::max<float>(0.0f, Vector::dotProduct(lightDirection, nhit)) 
					* mat.color * mat.diff;
				if(mat.spec > efl::ZERO)//specular
				{
					Vector R = lightDirection - nhit * 2.0f * Vector::dotProduct(lightDirection, nhit);
					float dot = Vector::dotProduct(r.direction, R.normalize());
					if (dot > 0)
					{
						float spec = efl::fastPow(dot, 32) * mat.spec;
						outColor += lightIntensity * spec;
					}
				}
			}
		}
		return outColor;
	}

	float spaceIoR;
	Vector backgroundColor;

	int numPlanes, numSpheres;
	int numLights;
	Sphere spheres[MAX_SPHERES];
	Plane planes[MAX_PLANES];
	LightSource lights[MAX_LIGHTS];
};

struct SphereMasses
{
	CUDA_HOST_DEVICE float& operator[](int i)
	{
		return m[i];
	}

	float m[MAX_SPHERES];
};

struct SphereVelocities
{
	CUDA_HOST_DEVICE Vector& operator[](int i)
	{
		return v[i];
	}

	Vector v[MAX_SPHERES];
};
