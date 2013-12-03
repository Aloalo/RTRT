#include "PPMRenderer.h"

PPMRenderer::PPMRenderer(const ImageInfo &info)
	: info(info)
{
	h_output = new Vector[info.height * info.width];
}


PPMRenderer::~PPMRenderer(void)
{
	printf("hello\n");
	delete[] h_output;
	gpuErrchk(cudaFree(d_output));
}

void PPMRenderer::renderToPPM(const SlowScene *d_scene, const CameraInfo &cam)
{
	setCameraInfo(cam);
	setImageInfo(info);
	renderToImage(dimBlock, dimGrid, d_scene, d_output);
	gpuErrchk(cudaMemcpy(h_output, d_output, sizeof(Vector) * info.height * info.width, cudaMemcpyDeviceToHost));
	saveToPPM();
}

void PPMRenderer::init()
{
	dimBlock = dim3(THREAD_DIM, THREAD_DIM, 1);
	dimGrid = dim3(info.width / dimBlock.x + (info.width % dimBlock.x > 0), info.height / dimBlock.y + (info.height % dimBlock.y > 0), 1);
	gpuErrchk(cudaMalloc((void**)&d_output, info.height * info.width * sizeof(Vector)));
}

void PPMRenderer::saveToPPM() const
{
	std::ofstream ofs("trace.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << info.width << " " << info.height << "\n255\n";
	for(unsigned int i = 0; i < info.width * info.height; ++i)
	{
		ofs << (unsigned char)(efl::min<float>(1.0f, h_output[i].x) * 255) << 
			(unsigned char)(efl::min<float>(1.0f, h_output[i].y) * 255) <<
			(unsigned char)(efl::min<float>(1.0f, h_output[i].z) * 255); 
	}
	ofs.close();
}
