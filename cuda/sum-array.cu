/*
 ============================================================================
 Name        : sum-array.cu
 Author      :
 Version     :
 Copyright   : Creative Commons Attribution-NonCommercial-ShareAlike 3.0
 Description : CUDA Sum Array
 ============================================================================
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "timing.h"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
		C[idx] = A[idx] + B[idx];
	}
}


void launchKernel(float* A, float* B, float* C, const int nBytes) {
	//Memory allocation:
	float *d_A, *d_B, *d_C;
	CUDA_CHECK_RETURN(cudaMalloc((float **)&d_A, nBytes));
	CUDA_CHECK_RETURN(cudaMalloc((float **)&d_B, nBytes));
	CUDA_CHECK_RETURN(cudaMalloc((float **)&d_C, nBytes));
	// Transfer the data from the CPU memory to the GPU global memory
	CUDA_CHECK_RETURN(cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_B, B, nBytes, cudaMemcpyHostToDevice));

	sumArraysOnGPU<<<1,32>>>(d_A,d_B,d_C,nBytes/sizeof(float));

	// with the parameter cudaMemcpyHostToDevice specifying the transfer direction.
	//Copy the result from the GPU memory back to the host:
	CUDA_CHECK_RETURN(cudaMemcpy(C, d_C, nBytes, cudaMemcpyDeviceToHost));
	// Release the memory used on the GPU
	CUDA_CHECK_RETURN(cudaFree(d_A));
	CUDA_CHECK_RETURN(cudaFree(d_B));
	CUDA_CHECK_RETURN(cudaFree(d_C));
}



void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	int idx;
	for (idx = 0; idx < N; idx++)
	{
		C[idx] = A[idx] + B[idx];
	}
}

void initialData(float *ip, int size)
{
	// generate different seed for random number
	time_t t;
	srand((unsigned) time(&t));
	int i;
	for (i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
	return;
}

int main(int argc, char **argv)
{
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);
	float *h_A, *h_B, *h_C;

	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);

	initialData(h_A, nElem);
	initialData(h_B, nElem);
	sumArraysOnHost(h_A, h_B, h_C, nElem);
	launchKernel(h_A, h_B, h_C, nElem);
	free(h_A);
	free(h_B);
	free(h_C);
	return(0);
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
