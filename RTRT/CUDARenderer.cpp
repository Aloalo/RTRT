#pragma once

#include "CUDARenderer.h"
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>


CUDARenderer::CUDARenderer(const SlowScene &h_scene, const ImageInfo &imageinfo, const PPMRenderer &snapshoter)
	: h_info(imageinfo), snapshot(snapshoter),
	vao(), OGLtexture(GL_TEXTURE_2D), vertices(GL_ARRAY_BUFFER, GL_STATIC_DRAW),
	vertexAttrib(0, 3, GL_FLOAT, GL_FALSE), p("../RTRT/Shaders/passthrough"),
	camera(h_info.width, h_info.height), h_scene(h_scene)
{
	skipStep = false;
	physicsOn = true;
}

CUDARenderer::~CUDARenderer(void)
{
	gpuErrchk(cudaFree(d_scene));
	deleteTexture();
	vao.destroy();
	vertices.destroy();
}

void CUDARenderer::initCUDA()
{
	gpuErrchk(cudaSetDevice(0));
	gpuErrchk(cudaGLSetGLDevice(0));
	gpuErrchk(cudaMalloc((void **)&d_scene, sizeof(SlowScene)));
	gpuErrchk(cudaMemcpy(d_scene, &h_scene, sizeof(SlowScene), cudaMemcpyHostToDevice));

	setImageInfo(h_info);
	gpuErrchk(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 1024 * 10));

	dimBlock = dim3(THREAD_DIM, THREAD_DIM, 1);
	dimGrid = dim3(h_info.width / dimBlock.x + (h_info.width % dimBlock.x > 0), h_info.height / dimBlock.y + (h_info.height % dimBlock.y > 0), 1);

	snapshot.init();
}

void CUDARenderer::initDrawing()
{
	GLfloat quad[] = 
	{ 
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f,  1.0f, 0.0f,
	};

	vao.bind();
	vertices.setData(quad, sizeof(quad));
	vertexAttrib.attribPointer();

	initTexture();
}

void CUDARenderer::draw(const glm::mat4 &View, const glm::mat4 &Projection)
{
	setCameraInfo(camera.cam);
	gpuErrchk(cudaGraphicsMapResources(1, &cudaSurfRes));
	{
		cudaArray *viewCudaArray;
		gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, cudaSurfRes, 0, 0));
		renderToTexture(dimBlock, dimGrid, d_scene, viewCudaArray);
	}
	gpuErrchk(cudaGraphicsUnmapResources(1, &cudaSurfRes));
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void CUDARenderer::keyPress(int key, int scancode, int action, int mods) 
{
	camera.keyPress(key, scancode, action, mods);
	if(key == GLFW_KEY_ENTER && action == GLFW_PRESS)
	{
		snapshot.renderToPPM(d_scene, camera.cam);
		setImageInfo(h_info);
		skipStep = true;
	}
	if(key == GLFW_KEY_F && action == GLFW_PRESS)
		physicsOn = !physicsOn;
}

void CUDARenderer::mouseMove(double x, double y) 
{
	camera.mouseMove(x, y);
}

void CUDARenderer::windowResize(int width, int height) 
{
	skipStep = true;
	h_info.setWidthHeight(width, height);
	camera.width = h_info.width;
	camera.height = h_info.height;
	setImageInfo(h_info);
	dimBlock = dim3(THREAD_DIM, THREAD_DIM, 1);
	dimGrid = dim3(h_info.width / dimBlock.x + (h_info.width % dimBlock.x > 0), h_info.height / dimBlock.y + (h_info.height % dimBlock.y > 0), 1);

	deleteTexture();
	OGLtexture = Texture2D(GL_TEXTURE_2D);
	initTexture();
}

void CUDARenderer::initState() 
{
	pp.init(h_scene);
}

void CUDARenderer::update(float deltaTime) 
{
	if(skipStep)
	{
		skipStep = false;
		return;
	}
	camera.update(deltaTime);
	if(physicsOn)
		pp.update(d_scene, deltaTime);
}

void CUDARenderer::initTexture()
{
	OGLtexture.bind();
	OGLtexture.texImage(0, GL_RGBA32F, glm::vec3(h_info.width, h_info.height, 0), GL_RGBA, GL_UNSIGNED_BYTE, 0);

	OGLtexture.texParami(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	OGLtexture.texParami(GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
	OGLtexture.texParami(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	OGLtexture.texParami(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	gpuErrchk(cudaGraphicsGLRegisterImage(&cudaSurfRes, OGLtexture.getID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

	p.use();
	glActiveTexture(GL_TEXTURE0);
	p.setUniform("renderedTexture", 0);
}

void CUDARenderer::deleteTexture()
{
	gpuErrchk(cudaGraphicsUnregisterResource(cudaSurfRes));
	OGLtexture.destroy();
}
