#pragma once

#include "ImageInfo.h"

class InputParser
{
public:
	InputParser(void);
	~InputParser(void);

	void parse();

	ImageInfo rtInfo;
	ImageInfo snInfo;
	int numSpheres;
};

