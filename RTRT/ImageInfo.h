#pragma once

#include <fstream>
#include "Vector.h"
#include "Essential.h"

struct ImageInfo
{
	ImageInfo(void)
	{
	}

	ImageInfo(int width, int height, float fieldOfView, int maxDepth, int AALevel = 1)
		: width(width), height(height), fieldOfView(fieldOfView), 
		aspectRatio((float) width / height), maxDepth(maxDepth), AALevel(AALevel)
	{
		angle = tan(efl::PI * 0.5 * fieldOfView / 180.0f);
		invWidth = 1.0f / (float) width / AALevel;
		invHeight = 1.0f / (float) height / AALevel;
	}

	~ImageInfo(void)
	{
	}

	void setWidthHeight(int w, int h)
	{
		width = w;
		height = h;
		aspectRatio = (float) width / (float) height;
		invWidth = 1.0f / (float) (width * AALevel);
		invHeight = 1.0f / (float) (height * AALevel);
	}

	void construct()
	{
		aspectRatio = (float) width / height;
		angle = tan(efl::PI * 0.5 * fieldOfView / 180.0f);
		aspectRatio = (float) width / (float) height;
		invWidth = 1.0f / (float) (width * AALevel);
		invHeight = 1.0f / (float) (height * AALevel);
	}

	int width, height;
	float invWidth, invHeight;
	float fieldOfView;
	float aspectRatio;
	float angle;
	int maxDepth, AALevel;
};

