#include "PhysicsProcessor.h"
#include "kernel.h"


PhysicsProcessor::PhysicsProcessor(void)
{
}


PhysicsProcessor::~PhysicsProcessor(void)
{
	gpuErrchk(cudaFree(d_V));
}

void PhysicsProcessor::init(const SlowScene &h_scene)
{
	gpuErrchk(cudaMalloc((void**)&d_V, sizeof(SphereVelocities)));
	gpuErrchk(cudaMemset(d_V, 0, sizeof(SphereVelocities)));

	SphereMasses h_M;
	int k = h_scene.numSpheres;
	for(int i = 0; i < k; ++i)
	{
		float r = h_scene.spheres[i].radius;
		h_M[i] = (4.0f / 3.0f) * r * r * r * efl::PI;
	}

	setSphereMasses(h_M);

	int n = k * (k - 1) / 2;
	dimBlock1 = dim3(THREAD_DIM, THREAD_DIM, 1);
	dimGrid1 = dim3(n / THREAD_DIM + (n % THREAD_DIM > 0), n / THREAD_DIM + (n % THREAD_DIM > 0), 1);

	dimBlock2 = dim3(THREAD_DIM, 1, 1);
	dimGrid2 = dim3(k / THREAD_DIM + (k % THREAD_DIM > 0), 1, 1);
}

void PhysicsProcessor::update(SlowScene *d_scene, float dt)
{
	updatePhysics(dimBlock1, dimGrid1, d_scene, d_V);
	updateScene(dimBlock2, dimGrid2, d_scene, d_V, dt);
}
