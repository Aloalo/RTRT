#include "InputParser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


InputParser::InputParser(void)
{
}


InputParser::~InputParser(void)
{
}

void InputParser::parse()
{
	std::ifstream f("GraphicsSettings.txt");
	if(f.is_open())
	{
		int mode = -1;
		try
		{
			while(!f.eof())
			{
				std::string s;
				f >> s;
				if(s == "#scene")
					mode = 0;
				else if(s == "#realtime")
					mode = 1;
				else if(s == "#snapshot")
					mode = 2;
				else if(s == "numSpheres")
				{
					f >> s;
					numSpheres = std::stoi(s);
				}
				else if(s == "windowWidth")
				{
					f >> s;
					mode == 1 ? rtInfo.width = std::stoi(s) : snInfo.width = std::stoi(s);
				}
				else if(s == "windowHeight")
				{
					f >> s;
					mode == 1 ? rtInfo.height = std::stoi(s) : snInfo.height = std::stoi(s);
				}
				else if(s == "AALevel")
				{
					f >> s;
					mode == 1 ? rtInfo.AALevel = std::stoi(s) : snInfo.AALevel = std::stoi(s);
				}
				else if(s == "maxRayDepth")
				{
					f >> s;
					mode == 1 ? rtInfo.maxDepth = std::stoi(s) : snInfo.maxDepth = std::stoi(s);
				}
				else if(s == "FOV")
				{
					f >> s;
					mode == 1 ? rtInfo.fieldOfView = std::stof(s) : snInfo.fieldOfView = std::stof(s);
				}
			}
		}
		catch(std::exception *ex)
		{
			printf("%s\n", ex->what());
			exit(0);
		}
		rtInfo.construct();
		snInfo.construct();
	}
	else
	{
		printf("Cannot open graphics settings");
		exit(0);
	}
	f.close();
}
