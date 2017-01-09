/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* PARALLEL MATRIX-MATRIX PRODUCT WITH CUDA                                  */
/*                                                                           */
/* File:         mmpcuda.cu                                                  */
/* Original Author:  Alberto Pou Quirós (Github: bertini36)                  */
/* Modified version: Raul Vidal Ortiz (Github: raulvo)                       */
/*                   Luis Andres Vazquez                                     */
/* Description:  This program performs a matrix product (A * B = C)          */
/*               parallelizing the computation with Nvidia CUDA technology   */
/* Compilation:  nvcc -o mmpcuda mmpcuda.cu                                  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cassert>
#include <getopt.h>
#include <omp.h>
#include "timing.h"

typedef struct arguments_t {
	double *matrix_a;
	double *matrix_b;
	double *matrix_c;
	double *d_a;
	double *d_b;
	double *d_c;
	dim3 dimGrid;
	dim3 dimBlock;
	size_t mwidth;
	size_t shmem_width;
	size_t shmem_height;
} arguments_t;

typedef struct paramaters_t {
	size_t mwidth;
	size_t block_size_x;
	size_t block_size_y;
	int show_results;
	int use_cuda_events;
	int use_gettimeofday;
	int check_result;
	void (*matmul_fun_t)(arguments_t*);
} parameters_t;

#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

void parse_args(int argc, char** argv, parameters_t* args);
void print_help(char** argv);

inline void checkCuda(cudaError_t e) {
	if (e != cudaSuccess) {
		err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
	}
}

void checkCorrectness(double *serial, double *computed, size_t mwidth) {
	for (int i = 0; i < mwidth; i++) {
		for (int j = 0; j < mwidth; j++) {
			if (serial[i * mwidth + j] != computed[i * mwidth + j]) {
				fprintf(stderr,
						"Serial result and used function result are not equal\n");
				fprintf(stderr, "Serial[%d,%d]:\t%f\nComputed[%d,%d]:\t%f\n", i,
						j, serial[i * mwidth + j], i, j,
						computed[i * mwidth + j]);
				exit(-1);
			}
		}
	}
}

void matrixProductCPUSerial(arguments_t* args) {
	int i, j, k = 0;
	for (i = 0; i < args->mwidth; i++) {
		for (j = 0; j < args->mwidth; j++) {
			args->matrix_c[i * args->mwidth + j] = 0.0;
			for (k = 0; k < args->mwidth; k++)
				args->matrix_c[i * args->mwidth + j] +=
						args->matrix_a[i * args->mwidth + k]
						               * args->matrix_b[k * args->mwidth + j];
		}
	}
}

__global__ void matrixProductShared(double *matrix_a, double *matrix_b,
		double *matrix_c, size_t mwidth, size_t twidth, size_t theight) {
	extern __shared__ double block[];
	double sum = 0;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y * theight + ty;
	int col = blockIdx.x * twidth + tx;
	double* ma = (double*) &block[0];
	double* mb = (double*) &block[twidth * theight];

	for (int step = 0; step < mwidth / twidth; step++) {
		if ((row < mwidth) && (step * twidth + tx < mwidth))
			ma[ty * twidth + tx] = matrix_a[row * mwidth + step * twidth + tx];

		if ((col < mwidth) && (step * theight + ty < mwidth))
			mb[ty * twidth + tx] = matrix_b[(step * theight + ty) * mwidth + col];
		__syncthreads();

		for (int k = 0; k < twidth; k++) {
			sum += ma[ty * twidth + k ] * mb[k * twidth + tx];
		}
		__syncthreads();
	}
	if (row < mwidth && col < mwidth)
		matrix_c[row * mwidth + col] = sum;
}

__global__ void matrixProduct(double *matrix_a, double *matrix_b,
		double *matrix_c, int mwidth) {
	double sum = 0;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	if (col < mwidth && row < mwidth) {
		for (int k = 0; k < mwidth; k++) {
			sum += matrix_a[row * mwidth + k] * matrix_b[k * mwidth + col];
		}
		matrix_c[row * mwidth + col] = sum;
	}
}

void launchMatrixProductCUDA(arguments_t* args) {
	matrixProduct<<<args->dimGrid, args->dimBlock>>>(args->d_a, args->d_b,
			args->d_c, args->mwidth);
}

void launchMatrixProductCUDAShared(arguments_t* args) {
	cudaDeviceProp props;
	size_t shmem_bytes = args->shmem_width * args->shmem_height * 2 * sizeof(double);
	cudaGetDeviceProperties(&props, 0);
	if (props.sharedMemPerBlock < shmem_bytes) {
		fprintf(stderr,
				"Not enough shared memory. Allowed per block: %lu\n Demanded: %lu\n",
				props.sharedMemPerBlock, shmem_bytes);
		exit(-1);
	}

	matrixProductShared<<<args->dimGrid, args->dimBlock, shmem_bytes>>>(args->d_a, args->d_b,
			args->d_c, args->mwidth, args->shmem_width, args->shmem_height);
}

void initializeMatrices(int mwidth, double* matrix_a, double* matrix_b) {
	srand(time(NULL));
	for (int i = 0; i < mwidth; i++) {
		for (int j = 0; j < mwidth; j++) {
			matrix_a[i * mwidth + j] = (double) i - j;
			matrix_b[i * mwidth + j] = (double) i + j;
		}
	}
}

void showResults(int mwidth, double* matrix_a, double* matrix_b,
		double* matrix_c) {
	printf("***** MATRIX A ***** \n");
	for (int i = 0; i < mwidth; i++) {
		for (int j = 0; j < mwidth; j++) {
			(j % mwidth == mwidth - 1) ?
					printf("%.1f \n", matrix_a[i * mwidth + j]) :
					printf("%.1f,", matrix_a[i * mwidth + j]);
		}
	}
	printf("***** MATRIX B ***** \n");
	for (int i = 0; i < mwidth; i++) {
		for (int j = 0; j < mwidth; j++) {
			(j % mwidth == mwidth - 1) ?
					printf("%.1f \n", matrix_b[i * mwidth + j]) :
					printf("%.1f,", matrix_b[i * mwidth + j]);
		}
	}
	printf("***** RESULT MATRIX ***** \n");
	for (int i = 0; i < mwidth; i++) {
		for (int j = 0; j < mwidth; j++) {
			(j % mwidth == mwidth - 1) ?
					printf("%.1f \n", matrix_c[i * mwidth + j]) :
					printf("%.1f,", matrix_c[i * mwidth + j]);
		}
	}

}

void printMatrix(double *matrix, size_t mwidth) {
	if (matrix != NULL) {
		printf("***** RESULT MATRIX ***** \n");
		for (int i = 0; i < mwidth; i++) {
			for (int j = 0; j < mwidth; j++) {
				(j % mwidth == mwidth - 1) ?
						printf("%.1f \n", matrix[i * mwidth + j]) :
						printf("%.1f,", matrix[i * mwidth + j]);
			}
		}
	}
}

int main(int argc, char** argv) {
	parameters_t params;
	arguments_t args;

	memset(&params, 0, sizeof(parameters_t));
	memset(&args, 0, sizeof(arguments_t));
	params.matmul_fun_t = &launchMatrixProductCUDA;
	parse_args(argc, argv, &params);
	int mwidth = params.mwidth;
	int block_size_x = params.block_size_x;
	int block_size_y = params.block_size_y;
	int use_cuda_events = params.use_cuda_events;
	int use_gettimeofday = params.use_gettimeofday;
	int show_results = params.show_results;
	double *h_a, *h_b, *h_c, *h_d;
	double *d_a, *d_b, *d_c;
	double size = (double) mwidth * mwidth * sizeof(double);
	h_a = new double[mwidth * mwidth];
	h_b = new double[mwidth * mwidth];
	h_c = new double[mwidth * mwidth];
	if (params.check_result)
		h_d = new double[mwidth * mwidth];
	initializeMatrices(mwidth, h_a, h_b);
	args.matrix_a = h_a;
	args.matrix_b = h_b;
	args.matrix_c = h_c;
	args.mwidth = params.mwidth;
	args.shmem_width = block_size_x;
	args.shmem_height = block_size_y;
	// Allocate memory in the device

	checkCuda(cudaMalloc((void **) &d_a, size));
	checkCuda(cudaMalloc((void **) &d_b, size));
	checkCuda(cudaMalloc((void **) &d_c, size));

	// Copy the information in the device
	checkCuda(cudaMemcpy(d_a, args.matrix_a, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_b, args.matrix_b, size, cudaMemcpyHostToDevice));

	// CUDA threads structure definition
	dim3 dimGrid((mwidth + block_size_x - 1) / block_size_x,
			(mwidth + block_size_y - 1) / block_size_y);
	dim3 dimBlock(block_size_x, block_size_y);

	args.d_a = d_a;
	args.d_b = d_b;
	args.d_c = d_c;
	args.dimGrid = dimGrid;
	args.dimBlock = dimBlock;

	fprintf(stderr,
			"Thread Block width: %d\nThread Block height: %d\nCUDA threads per block: %d\n\
Grid block width: %d\nGrid block height: %d\nTotal blocks: %d\nMatrix width: %d\n",
			block_size_x, block_size_y, dimBlock.x * dimBlock.y * dimBlock.z,
			dimGrid.x, dimGrid.y, dimGrid.x * dimGrid.y * dimGrid.z, mwidth);

	struct timeval time_a, time_b;
	double elapsed = 0.;
	// Create events
	cudaEvent_t event1, event2;
	float dt_ms;
	if (use_cuda_events) {
		checkCuda(cudaEventCreate(&event1));
		checkCuda(cudaEventCreate(&event2));

		// Record events around kernel launch
		checkCuda(cudaEventRecord(event1, 0));
	}
	if (use_gettimeofday)
		gettimeofday(&time_a, NULL);

	/* DO MATRIX MULTIPLICATION */
	params.matmul_fun_t(&args);
	/****************************/

	if (use_cuda_events) {
		checkCuda(cudaEventRecord(event2, 0));
		// Synchronize
		checkCuda(cudaEventSynchronize(event1));
		checkCuda(cudaEventSynchronize(event2)); // Wait for the event to be executed!

		// Calculate compute time
		checkCuda(cudaEventElapsedTime(&dt_ms, event1, event2));
	}
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaGetLastError());
	if (use_gettimeofday) {
		gettimeofday(&time_b, NULL);
		elapsed = getElapsedMsec(&time_a, &time_b);
		fprintf(stderr, "Compute time (gettimeofday): %.1f ms\n", elapsed);
		fprintf(stdout, "%d,%d,%d,%.1f\n", mwidth, block_size_x, block_size_y,
				elapsed);
	}
	if (use_cuda_events) {
		fprintf(stderr, "Compute time (cuda event): %.1f ms \n", dt_ms);
		fprintf(stdout, "%d,%d,%d,%.1f\n", mwidth, block_size_x, block_size_y,
				dt_ms);
	}
	// Copy results from device to the host
	checkCuda(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(d_a));
	checkCuda(cudaFree(d_b));
	checkCuda(cudaFree(d_c));

	if (show_results)
		showResults(mwidth, h_a, h_b, h_c);

	if (params.check_result) {
			args.matrix_c = h_d;
			matrixProductCPUSerial(&args);
			if (show_results) printMatrix(h_d,args.mwidth);
			checkCorrectness(h_d, h_c, args.mwidth);
	}
	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	if (params.check_result)
		delete[] h_d;
	cudaDeviceReset();
	return 0;
}

void parse_args(int argc, char** argv, parameters_t* params) {
	int c;
	params->mwidth = 1024;
	params->block_size_x = 16;
	params->block_size_y = 16;
	static struct option long_options[] = {
			{ "cuda-shared", required_argument,	0, 'c' },
			{ "cuda", required_argument, 0, 'u' },
			{ "sequential",required_argument, 0, 's' },
			{ "size", required_argument, 0, 'n' },
			{ "threadsx", required_argument, 0, 'x' },
			{ "threadsy", required_argument, 0, 'y' },
			{ "verify", no_argument, &params->check_result, 1 },
			{ "help", no_argument, 0, 'h' },
			{ "results", no_argument, &params->show_results, 1 },
			{ "cudaevents", no_argument, &params->use_cuda_events, 1 },
			{ "gettimeofday", no_argument, &params->use_gettimeofday, 1 },
			{ 0, 0, 0, 0 }
	};

	int option_index = 0;
	opterr = 1;
	while ((c = getopt_long(argc, argv, "cusn:x:y:vhreg", long_options,
			&option_index)) > 0) {
		switch (c) {
		case 'c':
			params->matmul_fun_t = &launchMatrixProductCUDAShared;
			break;
		case 'u':
			params->matmul_fun_t = &launchMatrixProductCUDA;
			break;
		case 's':
			params->matmul_fun_t = &matrixProductCPUSerial;
			break;
		case 'n':
			params->mwidth = atoi(optarg);
			break;
		case 'x':
			params->block_size_x = atoi(optarg);
			params->block_size_y = atoi(optarg);
			break;
		case 'y':
			params->block_size_y = atoi(optarg);
			break;
		case 'v':
			params->check_result = 1;
			break;
		case 'e':
			params->use_cuda_events = 1;
			break;
		case 'g':
			params->use_gettimeofday = 1;
			break;
		case 'r':
			params->show_results = 1;
			break;
		case 'h':
			print_help(argv);
			exit(0);
			break;
		default:
			print_help(argv);
			exit(-1);
			break;
		}

	}
	if (optind < argc) {
		fprintf(stdout, "Additional non needed arguments were received: ");
		while (optind < argc) {
			fprintf(stderr, "%s", argv[optind]);
			optind++;
		}
		fprintf(stdout, "\n");
	}
}

void print_help(char** argv) {
	fprintf(stderr,
			"Usage: %s [-c | -u | -s ] [-n size] [-x threadsx] [-y threadsy] [-v] [-e] [-g] [-r] [-h]\n",
			argv[0]);
	fprintf(stderr,
			"-c --cuda-shared:\t\tDo matmul with CUDA using shared memory\n");
	fprintf(stderr, "-u --cuda:\t\tDo matmul with CUDA\n");
	fprintf(stderr, "-s --sequential:\t\tDo matmul with Sequential\n");
	fprintf(stderr, "-n --size:\t\tSelect matrix width\n");
	fprintf(stderr, "-x --threadsx:\t\tSelect CUDA Thread Block width\n");
	fprintf(stderr, "-y --threadsy:\t\tSelect CUDA Thread Block height\n");
	fprintf(stderr,
			"-v --verify:\t\tVerify result against a sequential computation\n");
	fprintf(stderr, "-e --cudaevents:\t\tMeasure time with CUDA events\n");
	fprintf(stderr, "-g --gettimeofday:\t\tMeasure time with gettimeofday()\n");
	fprintf(stderr, "-r --results:\t\tPrint matrices\n");
	fprintf(stderr, "-h --help:\t\tPrint this help\n");
}
