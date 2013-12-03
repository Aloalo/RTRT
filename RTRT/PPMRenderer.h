#pragma once

#include "kernel.h"
#include "ImageInfo.h"


class PPMRenderer
{
public:
	PPMRenderer(const ImageInfo &info);
	~PPMRenderer(void);

	void init();
	void renderToPPM(const SlowScene *d_scene, const CameraInfo &cam);
	void saveToPPM() const;

private:
	Vector *d_output;
	Vector *h_output;
	dim3 dimBlock, dimGrid;
	ImageInfo info;
};

