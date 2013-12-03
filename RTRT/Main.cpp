#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "CUDARenderer.h"
#include "Engine.h"
#include "Input.h"
#include "Vector.h"
#include "InputParser.h"

void printVector(const Vector &vec)
{
	printf("%.2f %.2f %.2f\n", vec.x, vec.y, vec.z);
}

void resetCuda()
{
	gpuErrchk(cudaDeviceReset());
}

int main()
{
	atexit(resetCuda);
	InputParser parser;
	parser.parse();

	srand(time(0));
	std::vector<Plane> planes;
	std::vector<Sphere> spheres;
	std::vector<LightSource> lightVec;
	SphereMasses masses;

	planes.push_back(Plane(rtrt::Material(Vector(1.0f), 1.0f, 0.0f, 0), Vector(0.0f, 1.0f, 0.0f), Vector(0.0f, -5.0f, 0.0f)));
	planes.push_back(Plane(rtrt::Material(Vector(1.0f), 1.0f, 0.0f, 0), Vector(0.0f, 0.0f, 1.0f), Vector(0.0f, 0.0f, -40.0f)));
	planes.push_back(Plane(rtrt::Material(Vector(1.0f), 1.0f, 0.0f, 0), Vector(0.0f, 0.0f, -1.0f), Vector(0.0f, 0.0f, 3.0f)));
	planes.push_back(Plane(rtrt::Material(Vector(1.0f), 1.0f, 0.0f, 0), Vector(-1.0f, 0.0f, 0.0f), Vector(25.0f, 0.0f, 0.0f)));
	planes.push_back(Plane(rtrt::Material(Vector(1.0f), 1.0f, 0.0f, 0), Vector(1.0f, 0.0f, 0.0f), Vector(-25.0f, 0.0f, 0.0f)));

	int n = parser.numSpheres;
	int sqr = (int)ceil(sqrtf(n));
	for(int i = 0; i < sqr; i++)
		for(int j = 0; i * sqr + j < n && j < sqr; j++)
			spheres.push_back(Sphere(rtrt::Material(Vector(0.0, 1.0, 0.36), 1.0f, 0.8f, 20), Vector((i - 5) * 2 - 5, (j - 5) * 2 + 10, -30.0f + (float)(rand() % 10) / 10.0f), 0.8f));

	spheres.push_back(Sphere(rtrt::Material(Vector(0.3f, 0.3f, 0.8f), 1.0f, 0.5f, 1.0f, 32), Vector(15, 0, -25), 4.0f));
	spheres.push_back(Sphere(rtrt::Material(Vector(1.0, 0.32, 0.36), 1.0f, 0.8f, 8), Vector(0.0f, 0.0f, -20.0f), 3.5f));
	spheres.push_back(Sphere(rtrt::Material(Vector(1.0f), 1.0f, 0.1f, 1.0f, 64), Vector(5, -1, -15), 2.0f));
	spheres.push_back(Sphere(rtrt::Material(Vector(1.0, 1.0, 0.5), 1.0f, 1.5f, 0.1f, 0.2f, 0.1f, 0.9f, 16), Vector(-5.5, 0, -15), 3));

	lightVec.push_back(LightSource(Vector(-4, 70, -10), Vector(10.0f), Vector(0.0f, 1.0f, 0.0f)));
	lightVec.push_back(LightSource(Vector(-1, 10, -20), Vector(110.0f), Vector(0.0f, 1.0f, 1.5f)));
	SlowScene scene(spheres, planes, lightVec, Vector(0.8, 0.8, 0.8), 1.0f);

	Engine e(1. / 80., parser.rtInfo.width, parser.rtInfo.height);
	e.setWindowTitle("RTRT");
	Input input;
	input.setMouseMoveCallback();
	input.setMouseClickCallback();

	PPMRenderer snapshoter(parser.snInfo);
	CUDARenderer *renderer = new CUDARenderer(scene, parser.rtInfo, snapshoter);
	renderer->initCUDA();

	e.addToDisplayList(std::shared_ptr<Drawable>(renderer));
	e.addToUpdateList(std::shared_ptr<Updateable>(renderer));
	input.addInputObserver(std::shared_ptr<InputObserver>(renderer));
	renderer = NULL;
	e.start();
	e.stop();
	return 0;
}