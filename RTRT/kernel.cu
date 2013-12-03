#include <cstdio>
#include <ctime>

#include "kernel.h"
#include <helper_cuda.h>

__constant__ ImageInfo info;
__constant__ CameraInfo camInfo;
__constant__ SphereMasses M;
__constant__ Vector G;
__constant__ float damping = 0.9f;

surface<void, cudaSurfaceType2D> surfRef;

__device__ Vector trace(const Ray &r, float currentIoR, int depth, const SlowScene *scene)
{
	rtrt::Material mat;
	Vector nhit;
	float tnear;
	if(!scene->hitObject(r, tnear, nhit, mat))
		return scene->backgroundColor;
	
	Vector phit = r.origin + r.direction * tnear;
	Vector outColor = scene->shade(r, mat, nhit, phit);

	float dotProd = Vector::dotProduct(r.direction, nhit);
	float ior = mat.ior;
	bool inside = false;

	if(dotProd > 0)
	{
		dotProd = -dotProd;
		nhit = -nhit;
		if(efl::abs(currentIoR - ior) < efl::ZERO)
			ior = scene->spaceIoR;
		inside = true;
	}

	if(depth < info.maxDepth)
	{
		Vector refractionColor(0.0f), reflectionColor(0.0f);
		if(!inside && mat.reflectivity > efl::ZERO)
		{
			Vector refldir = r.direction - nhit * 2.0f * dotProd;
			reflectionColor = trace(Ray(phit + nhit * efl::BIAS, refldir), ior, depth + 1, scene) * mat.color * mat.reflectivity;
		}
		if(mat.transparency > efl::ZERO)
		{
			float n = currentIoR / ior;
			float cosT2 = 1.0f - n * n * (1.0f - dotProd * dotProd);
			if(cosT2 > 0.0f)
			{
				Vector refrdir = r.direction * n - nhit * (n * dotProd + sqrt(cosT2));
				refractionColor = trace(Ray(phit - nhit * efl::BIAS, refrdir), ior, depth + 1, scene) * mat.transparency;
			}
		}
		outColor += reflectionColor + refractionColor;
	}
	return outColor;
}

__device__
inline void copyScene(SlowScene &dst, const SlowScene &src, int threadNum)
{
	int t = src.numPlanes + src.numSpheres + src.numLights;
	if(threadNum < src.numSpheres)
		dst.spheres[threadNum] = src.spheres[threadNum];
	else if(threadNum < src.numPlanes + src.numSpheres)
		dst.planes[threadNum - src.numSpheres] = src.planes[threadNum - src.numSpheres];
	else if(threadNum < t)
		dst.lights[threadNum - src.numSpheres - src.numPlanes] = src.lights[threadNum - src.numSpheres - src.numPlanes];
	else if(threadNum < t+1)
		dst.numSpheres = src.numSpheres;
	else if(threadNum < t+2)
		dst.numPlanes = src.numPlanes;
	else if(threadNum < t+3)
		dst.numLights = src.numLights;
	else if(threadNum < t+4)
		dst.backgroundColor = src.backgroundColor;
	else if(threadNum < t+5)
		dst.spaceIoR = src.spaceIoR;
}

__device__ Ray ComputeCameraRay(float i, float j)
{
	float normalized_i = (i / info.width) - 0.5;
	float normalized_j = (j / info.height) - 0.5;
	Vector ray_direction = (-camInfo.getRight() * normalized_i +
		-camInfo.getUp() * normalized_j + -camInfo.getDirection()).normalized();
	return Ray(camInfo.position, ray_direction);
}

__global__ void 
__launch_bounds__(THREAD_DIM * THREAD_DIM)
renderKernelImage(const SlowScene *scene, Vector *outImage)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * info.AALevel;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * info.AALevel;

	__shared__ SlowScene sharedScene;
	copyScene(sharedScene, *scene, threadIdx.y * THREAD_DIM + threadIdx.x);
	__syncthreads();

	if(x >= info.width * info.AALevel || y >= info.height * info.AALevel)
		return;

	float from = -(float)info.AALevel / 2.0f + 1.0f;
	float to = (float)info.AALevel / 2.0f;
	Vector outColor(0.0f);
	for(float aay = from; !(aay > to); aay += 1.0f)
		for(float aax = from; !(aax > to); aax += 1.0f)
		{
			float xx = (2.0f * ((x + aax) * info.invWidth) - 1.0f) * info.angle * info.aspectRatio;
			float yy = (1.0f - 2.0f * ((y + aay) * info.invHeight)) * info.angle;
			Vector raydir(xx, yy, -1.0f);
			raydir.rotateX(camInfo.pitch);
			raydir.rotateY(camInfo.yaw);
			raydir.normalize();
			outColor += trace(Ray(camInfo.position, raydir), 1.0f, 0, &sharedScene);
		}
	outImage[y * info.width / info.AALevel + x / info.AALevel] = (outColor * (1.0f / float(info.AALevel * info.AALevel)));
}

__global__ void 
__launch_bounds__(THREAD_DIM * THREAD_DIM)
renderKernelTexture(const SlowScene *scene, cudaArray *output)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * info.AALevel;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * info.AALevel;

	__shared__ SlowScene sharedScene;
	copyScene(sharedScene, *scene, threadIdx.y * THREAD_DIM + threadIdx.x);
	__syncthreads();

	if(x >= info.width * info.AALevel || y >= info.height * info.AALevel)
		return;

	float from = -(float)info.AALevel / 2.0f + 1.0f;
	float to = (float)info.AALevel / 2.0f;
	Vector outColor(0.0f);
	for(float aay = from; !(aay > to); aay += 1.0f)
		for(float aax = from; !(aax > to); aax += 1.0f)
		{
			float xx = (2.0f * ((x + aax) * info.invWidth) - 1.0f) * info.angle * info.aspectRatio;
			float yy = (1.0f - 2.0f * ((y + aay) * info.invHeight)) * info.angle;
			Vector raydir(xx, yy, -1.0f);
			raydir.rotateX(camInfo.pitch);
			raydir.rotateY(camInfo.yaw);
			raydir.normalize();
			outColor += trace(Ray(camInfo.position, raydir), 1.0f, 0, &sharedScene);
		}
	outColor *= (1.0f / float(info.AALevel * info.AALevel));
	float4 out = make_float4(outColor.x, outColor.y, outColor.z, 1);
	surf2Dwrite(out, surfRef, x * sizeof(float4) / info.AALevel, info.height - y / info.AALevel - 1);
}

__global__ void 
__launch_bounds__(THREAD_DIM * THREAD_DIM)
physicsKernel(SlowScene *scene, SphereVelocities *V)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= y || y >= scene->numSpheres)
		return;

	Sphere s1 = scene->spheres[x];
	Sphere s2 = scene->spheres[y];
	
	float d = s1.center.distance(s2.center);
	float rr = s1.radius + s2.radius;

	if(d <= rr)
	{
		Vector v1 = (*V)[x];
		Vector v2 = (*V)[y];
		Vector t = (s1.center - s2.center).normalized();

		float dotProd1 = Vector::dotProduct(v1, t);
		float dotProd2 = Vector::dotProduct(v2, -t);
		if(dotProd1 > 0.0f && dotProd2 > 0.0f)
			return;

		float stride = (rr - d) / 2.0f;
		scene->spheres[x].center += t * stride;
		scene->spheres[y].center -= t * stride;

		Vector v1x = t * dotProd1;
		Vector v1y = v1 - v1x;

		Vector v2x = -t * dotProd2;
		Vector v2y = v2 - v2x;

		float mm = (M[x] + M[y]);
		(*V)[x] = v1x * (M[x] - M[y]) / mm + v2x * (2.0f * M[y]) / mm + v1y;
		(*V)[y] = v1x * (2.0f * M[x]) / mm + v2x * (M[y] - M[x]) / mm + v2y;
	}

	__syncthreads();
	int n = scene->numPlanes;
	for(int i = 0; i < n; ++i)
	{
		Plane p = scene->planes[i];
		Vector tmp = s2.center - p.p;
		float dotProd = Vector::dotProduct(tmp, p.n);
		float dotProd2 = Vector::dotProduct(p.n, (*V)[y]);
		if(dotProd <= s2.radius && dotProd2 < 0.0f)
		{
			(*V)[y] = ((*V)[y] - p.n * 2.0f * dotProd2) * damping;
			scene->spheres[y].center += p.n * (s2.radius - dotProd);
		}
	}
}

__global__ void 
__launch_bounds__(THREAD_DIM * THREAD_DIM)
updateKernel(SlowScene *scene, SphereVelocities *V, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= scene->numSpheres)
		return;

	(*V)[x] += G * dt;
	scene->spheres[x].center += (*V)[x]* dt;
}

void setSphereMasses(const SphereMasses &masses)
{
	Vector h_G(0.0f, -9.81f, 0.0f);
	gpuErrchk(cudaMemcpyToSymbol(G, &h_G, sizeof(Vector)));
	gpuErrchk(cudaMemcpyToSymbol(M, &masses, sizeof(SphereMasses)));
	gpuErrchk(cudaDeviceSynchronize());
}

void setImageInfo(const ImageInfo &imageinfo)
{
	gpuErrchk(cudaMemcpyToSymbol(info, &imageinfo, sizeof(ImageInfo)));
	gpuErrchk(cudaDeviceSynchronize());
}

void setCameraInfo(const CameraInfo &camerainfo)
{
	gpuErrchk(cudaMemcpyToSymbol(camInfo, &camerainfo, sizeof(CameraInfo)));
	gpuErrchk(cudaDeviceSynchronize());
}

void renderToTexture(const dim3 &dimBlock, const dim3 &dimGrid, const SlowScene *d_scene, cudaArray *output)
{
	gpuErrchk(cudaBindSurfaceToArray(surfRef, output));
	gpuErrchk(cudaDeviceSynchronize());
	renderKernelTexture<<<dimGrid, dimBlock>>>(d_scene, output);
	gpuErrchk(cudaDeviceSynchronize());
}

void renderToImage(const dim3 &dimBlock, const dim3 &dimGrid, const SlowScene *d_scene, Vector *d_output)
{
	renderKernelImage<<<dimGrid, dimBlock>>>(d_scene, d_output);
	gpuErrchk(cudaDeviceSynchronize());
}

void updatePhysics(const dim3 &dimBlock, const dim3 &dimGrid, SlowScene *d_scene, SphereVelocities *d_V)
{
	physicsKernel<<<dimGrid, dimBlock>>>(d_scene, d_V);
	gpuErrchk(cudaDeviceSynchronize());
}

void updateScene(const dim3 &dimBlock, const dim3 &dimGrid, SlowScene *d_scene, SphereVelocities *d_V, float dt)
{
	updateKernel<<<dimGrid, dimBlock>>>(d_scene, d_V, dt);
	gpuErrchk(cudaDeviceSynchronize());
}