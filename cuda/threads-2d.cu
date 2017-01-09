#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cassert>


__global__ void threads2d() {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
	"gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
	blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
	gridDim.x,gridDim.y,gridDim.z);
}

int main (int argc, char** argv) {
	int nElem = 36;
	dim3 block(3,3);
	dim3 grid((nElem + block.x - 1) / block.x, (nElem + block.x - 1) / block.x);
	fprintf(stdout,"grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	fprintf(stdout,"block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
	threads2d<<<grid, block>>>();
	cudaDeviceSynchronize();
	return 0;
}
