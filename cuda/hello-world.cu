/*
 ============================================================================
 Name        : hello-world.cu
 Author      : Raul Vidal (github Raulvo)
 Version     : 1.0
 Copyright   : Public Domain
 Description : CUDA Hello World string reverse
 ============================================================================
 */

#include <iostream>
#include <string>
#include <numeric>
#include <stdlib.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


__global__ void helloWorldKernel(char* helloworld,size_t string_size) {
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int j = string_size - i - 1;
	char tmp;
	if (i < j && i < string_size) {
		tmp = helloworld[i];
		helloworld[i] = helloworld[j];
		helloworld[j] = tmp;
	}
	__syncthreads(); /* Only needed if there are more than 32 threads */

}

void gpuHelloWorld(std::string& helloworld)
{
	char* hosthwascii = new char[helloworld.length()+1];
	size_t hwsize = helloworld.length();
	size_t copied = helloworld.copy(hosthwascii,helloworld.size());
	char* gpuhwascii;
	static const int BLOCK_SIZE = 32;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuhwascii, sizeof(char)*hwsize));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuhwascii, hosthwascii, sizeof(char)*hwsize, cudaMemcpyHostToDevice));

	helloWorldKernel<<<1, BLOCK_SIZE>>> (gpuhwascii,hwsize);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(hosthwascii, gpuhwascii, sizeof(char)*hwsize, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuhwascii));
	hosthwascii[hwsize] = '\0';
	std::cout<<"gpu Hello World = "<< std::string(hosthwascii) << std::endl;
	delete[] hosthwascii;
}

void cpuHelloWorld(std::string& helloworld) {
	char tmp = 0;

	for (unsigned int i = 0, j = helloworld.length()-1;
			i < j;
			i++, j--)
	{
		tmp = helloworld.at(i);
		helloworld.replace(i,1,&helloworld[j],1);
		helloworld.replace(j,1,&tmp,1);
	}
	std::cout<<"cpu Hello World = "<< helloworld << std::endl;
}


int main(void)
{
	std::string helloworld = std::string("dlroW olleH");
	cpuHelloWorld(helloworld);
	gpuHelloWorld(helloworld);
	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

