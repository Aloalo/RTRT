#pragma once

#include "UnLitObject.h"
#include "OGLObjects.h"

#include "SlowScene.h"
#include "ImageInfo.h"
#include "kernel.h"
#include "InputObserver.h"
#include "Updateable.h"
#include "PPMRenderer.h"
#include "PhysicsProcessor.h"

class CUDARenderer : public UnLitObject, public InputObserver, public Updateable
{
public:
	CUDARenderer(const SlowScene &h_scene, const ImageInfo &imageinfo, const PPMRenderer &snapshoter);
	~CUDARenderer(void);

	void initCUDA();
	void initDrawing();
	void draw(const glm::mat4 &View, const glm::mat4 &Projection);
	void keyPress(int key, int scancode, int action, int mods);
	void mouseMove(double x, double y);
	void windowResize(int width, int height);
	void initState();
	void update(float deltaTime);

private:
	SlowScene *d_scene;
	SlowScene h_scene;
	ImageInfo h_info;
	dim3 dimBlock, dimGrid;
	cudaGraphicsResource_t cudaSurfRes;

	Texture2D OGLtexture;
	VertexArrayObject vao;
	BufferObject vertices;
	VertexAttribArray vertexAttrib;
	Program p;
	bool skipStep;
	bool physicsOn;

	PPMRenderer snapshot;
	RTRTCamera camera;
	PhysicsProcessor pp;

	void initTexture();
	void deleteTexture();
};
