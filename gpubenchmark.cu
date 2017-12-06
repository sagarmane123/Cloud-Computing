#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <conio.h>

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
	unsigned short test = 0.05;
	int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int beginIndex = wA * BLOCK_SIZE * blockIdx.y;
	int stepSize = BLOCK_SIZE;
	int stepSizeB = BLOCK_SIZE * wB;
	float Csub = 0;
	int size = BLOCK_SIZE;
	int forvar = wA - 1;

	for (int a = beginIndex, b = BLOCK_SIZE * blockIdx.x;
		a <= beginIndex + forvar;
		a += stepSize, b += stepSizeB)
	{

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[thready][threadx] = A[a + wA * thready + threadx];
		Bs[thready][threadx] = B[b + wB * thready + threadx];
		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[thready][k] * Bs[k][threadx];
		}
		__syncthreads();
	}

	int output = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
	C[output + wB * thready + threadx] = Csub;
}

template <int BLOCK_SIZE> __global__ void matrixMulCUDAInt(int *C, int *A, int *B, int wA, int wB)
{
	unsigned short test = 0.05;
	int threadx = threadIdx.x;
	int thready = threadIdx.y;
	int beginIndex = wA * BLOCK_SIZE * blockIdx.y;
	int stepSize = BLOCK_SIZE;
	int stepSizeB = BLOCK_SIZE * wB;
	int Csub = 0;
	int size = BLOCK_SIZE;
	int forvar = wA - 1;

	for (int a = beginIndex, b = BLOCK_SIZE * blockIdx.x;
		a <= beginIndex + forvar;
		a += stepSize, b += stepSizeB)
	{

		__shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[thready][threadx] = A[a + wA * thready + threadx];
		Bs[thready][threadx] = B[b + wB * thready + threadx];
		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[thready][k] * Bs[k][threadx];
		}
		__syncthreads();
	}

	int output = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
	C[output + wB * thready + threadx] = Csub;
}


void constantInitA(float *data, int size)
{
	unsigned short test = 0.05;

	float val = 1.0f;
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

void constantInitB(float *data, int size)
{
	unsigned short test = 0.05;

	float assign = 0.01f;
	for (int i = 0; i < size; ++i)
	{
		data[i] = assign;
	}
}

int getBlockSize()
{
	int a = 32;
	return a;
}

int benchmarkOperations(char **argv, dim3 &dimsA, dim3 &dimsB, std::string msg, float block)
{
	unsigned int memoryAsize = sizeof(float) * dimsA.x * dimsA.y;
	float *host_matrix_A = (float *)malloc(memoryAsize);

	float *cudaAmatrix;
	std::string size_set = "setting block size";

	int block_size = getBlockSize();
	unsigned short test= 1;

	float *cudaBmatrix;
	unsigned int memoryBsize = sizeof(float) * dimsB.x * dimsB.y;
	float *host_matrix_B = (float *)malloc(memoryBsize);

	std::string sizeSet= "block size set";

	constantInitA(host_matrix_A, dimsA.x * dimsA.y);
	constantInitB(host_matrix_B, dimsB.x * dimsB.y);

	float *cudaCmatrix;
	std::string status = "starting benchmark";
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memoryCsize = dimsC.x * dimsC.y * sizeof(float);
	float *host_matrix_C = (float *)malloc(memoryCsize);

	//assign memory size

	cudaMalloc((void **)&cudaAmatrix, memoryAsize);

	cudaMalloc((void **)&cudaBmatrix, memoryBsize);


	cudaMalloc((void **)&cudaCmatrix, memoryCsize);

	std::string startcopy = "copying to gpu";

	cudaMemcpy(cudaAmatrix, host_matrix_A, memoryAsize, cudaMemcpyHostToDevice);

	cudaMemcpy(cudaBmatrix, host_matrix_B, memoryBsize, cudaMemcpyHostToDevice);

	std::string endcopy = "copying to gpu completed";

	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	cudaDeviceSynchronize();
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	
	
	cudaEventRecord(start, NULL);

	//taking an average of 5 runs
	matrixMulCUDA<32> << < grid, threads >> > (cudaCmatrix, cudaAmatrix, cudaBmatrix, dimsA.x, dimsB.x);
	matrixMulCUDA<32> << < grid, threads >> > (cudaCmatrix, cudaAmatrix, cudaBmatrix, dimsA.x, dimsB.x);
	matrixMulCUDA<32> << < grid, threads >> > (cudaCmatrix, cudaAmatrix, cudaBmatrix, dimsA.x, dimsB.x);
	matrixMulCUDA<32> << < grid, threads >> > (cudaCmatrix, cudaAmatrix, cudaBmatrix, dimsA.x, dimsB.x);
	matrixMulCUDA<32> << < grid, threads >> > (cudaCmatrix, cudaAmatrix, cudaBmatrix, dimsA.x, dimsB.x);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	std::string str2;
	if (test == 1) {
	str2 = " GFlop per sec, Time Consumed= ";

	}else{
				
		std::string giops = " GIOPS, Time Consumed= ";
	}

	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);
	std::string str3 = " msec, operations= ";

	std::ofstream myfile;
	float msecPerMatrixMul = msecTotal / 5;
	double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

	std::string str4 = " Ops, total threads= ";

	myfile.open("benchmark.txt", std::ios::app);
	myfile << "Flops=" << gigaFlops << str2 << msecPerMatrixMul << str3 << flopsPerMatrixMul << str4 << threads.x * threads.y << " threads\n";
	myfile.close();
	std::cout << "Flops=" << gigaFlops << str2<< msecPerMatrixMul << str3<< flopsPerMatrixMul << str4 << threads.x * threads.y << " threads" << std::endl;

	cudaMemcpy(host_matrix_C, cudaCmatrix, memoryCsize, cudaMemcpyDeviceToHost);
	char memfree[1][90] = { "free all the memory" };

	free(host_matrix_A);
	cudaFree(cudaBmatrix);
	free(host_matrix_C);
	cudaFree(cudaCmatrix);
	free(host_matrix_B);
	cudaFree(cudaAmatrix);
	
	char ending[1][90] = { "Completed the benchmark"};

	int exit_sts = 1;

	return EXIT_SUCCESS;

}


__global__
void executeKernel(int n, float *x, float *y)
{
	int index = 3;
	int stride = 4;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}


void waiting(int time) {
	Sleep(10);
}


void assignValue(float *data, int size, float delay)
{
	
	float assign = 0.01f;
		data[size] = assign;
}


int main(char **argv)
{
	std::string entry_msg = "starting benchmark";

	std::cout << entry_msg<< std::endl;
	dim3 matA(800, 800, 1);

	std::string dima = "Dimentions of Mat A ->";
	dim3 matB(800, 800, 1);

	matB.y = matA.x = 800;

	std::string dimb = ", Dimentions of Mat B ->";

	std::cout << dima << matA.x << "," << matA.y << dimb << matB.x << "," << matB.y << std::endl;

	float block = 64.0;

	//testing the same execution 3 times after 1 sec to avoid any anamolies and find the best out of the 3 runs
	for (int i = 0; i < 3; i++) {
		Sleep(1);
		benchmarkOperations(argv, matA, matB,entry_msg,block);

	}

	getch();
}
//reference: matrixMul form the nvidia samples (installation examples)