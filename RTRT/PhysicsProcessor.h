#pragma once

#include "SlowScene.h"
#include <vector_types.h>

class PhysicsProcessor
{
public:
	PhysicsProcessor(void);
	~PhysicsProcessor(void);

	void init(const SlowScene &h_scene);
	void update(SlowScene *d_scene, float dt);

private:
	SphereVelocities *d_V;
	dim3 dimGrid1, dimGrid2;
	dim3 dimBlock1, dimBlock2;
};

