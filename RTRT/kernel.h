#pragma once

#include "SlowScene.h"
#include "ImageInfo.h"
#include "Utils.h"
#include "RTRTCamera.h"

#define THREAD_DIM 16
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void renderToImage(const dim3 &dimBlock, const dim3 &dimGrid, const SlowScene *d_scene, Vector *d_output);
void renderToTexture(const dim3 &dimBlock, const dim3 &dimGrid, const SlowScene *d_scene, cudaArray *output);
void updatePhysics(const dim3 &dimBlock, const dim3 &dimGrid, SlowScene *d_scene, SphereVelocities *d_V);
void updateScene(const dim3 &dimBlock, const dim3 &dimGrid, SlowScene *d_scene, SphereVelocities *d_V, float dt);

void setImageInfo(const ImageInfo &imageinfo);
void setCameraInfo(const CameraInfo &camerainfo);
void setSphereMasses(const SphereMasses &masses);
void setScene(const SlowScene &slowScene);