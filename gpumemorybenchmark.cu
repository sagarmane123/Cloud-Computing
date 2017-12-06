#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>  

#include <stdio.h>
#include <conio.h>


__global__ void memoryBenc(int *g_out, int *g_in, int N, int inner_reps)
{
	
    int idx = blockIdx.x * blockDim.x ;
	int threadx = idx + threadIdx.x;
    if (threadx > N){
		int a = 10;
	}
	else {
        for (int i=0; i<inner_reps; ++i)
        {
            g_out[threadx] = g_in[threadx] + 1;
        }
    }
}

#define STREAM_COUNT 4

int *h_data_source;
int *h_data_sink;

int *h_data_in[STREAM_COUNT];
int *d_data_in[STREAM_COUNT];

int *h_data_out[STREAM_COUNT];
int *d_data_out[STREAM_COUNT];


cudaEvent_t cycleDone[STREAM_COUNT];
cudaStream_t stream[STREAM_COUNT];

cudaEvent_t start, stop;

int N = 1 << 22;

//taking an average of 25 operations

int nreps = 25;           
int inner_reps = 5;

int memsize;

dim3 block(512);
dim3 grid;

int thread_blocks;

float processWithStreams(int streams_used);

std::string getRandOut() {
	std::string randOut = "sequential multi stream read write with ^ streams in GBps =";
	return randOut;
}

std::string getdOut() {
	std::string randOut = "sequential read write with single stream in GBps =";
	return randOut;
}

int setDev() {
	return 0;
}


int main(int argc, char *argv[])
{
	cudaError error;
	float test_blocks = 1500;
	int cuda_device = setDev();
	cuda_device = 0;
	float scale_factor;
    cudaDeviceProp deviceProp;
	const char *name = "simpleMultiCopy";

		error = cudaGetDeviceProperties(&deviceProp, 0);

	std::cout<<"size   of array = "<<N<<"\n";

    memsize = N * sizeof(int);

    thread_blocks = N / block.x;

    grid.x = thread_blocks % 65535;
	int val = 65535 + 1;
    grid.y = (thread_blocks / val);


    h_data_source = (int *) malloc(memsize);
    h_data_sink = (int *) malloc(memsize);

    for (int i =0; i<STREAM_COUNT; ++i)
    {

		error = cudaHostAlloc(&h_data_in[i], memsize,
                                      cudaHostAllocDefault);
		error = cudaMalloc(&d_data_in[i], memsize);

		error = cudaHostAlloc(&h_data_out[i], memsize,
                                      cudaHostAllocDefault);
		error = cudaMalloc(&d_data_out[i], memsize);

		error = cudaStreamCreate(&stream[i]);
		error = cudaEventCreate(&cycleDone[i]);

        cudaEventRecord(cycleDone[i], stream[i]);
    }

    cudaEventCreate(&start);

	float memcpy_h2d_time; 
	cudaEventCreate(&stop);

	for (int i = 0; i<N; ++i)
	{
		h_data_source[i] = 0;
	}

	for (int i = 0; i<STREAM_COUNT; ++i)
	{
		memcpy(h_data_in[i], h_data_source, memsize);
	}
	cudaEventRecord(start,0);
	error = cudaMemcpyAsync(d_data_in[0], h_data_in[0], memsize,
                                    cudaMemcpyHostToDevice,0);
    cudaEventRecord(stop,0);

	double kt=0.00;
	cudaEventSynchronize(stop);

	long kernelTime = 0;

    cudaEventElapsedTime(&memcpy_h2d_time, start, stop);

    cudaEventRecord(start,0);
	error = cudaMemcpyAsync(h_data_out[0], d_data_out[0], memsize,
                                    cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float memcpy_d2h_time;
    cudaEventElapsedTime(&memcpy_d2h_time, start, stop);

    cudaEventRecord(start,0);
    memoryBenc<<<grid, block,0,0>>>(d_data_out[0], d_data_in[0], N, inner_reps);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

	float time1 = (memsize * 1e-6) / memcpy_h2d_time;

	std::cout << "write throughput in GBps= "<<time1<<std::endl;

	float time2 = (memsize * 1e-6) / memcpy_d2h_time;
	std::cout << "read throughput in GBps= " << time2 << std::endl;

    float serial_time = processWithStreams(1);
    float overlap_time = processWithStreams(STREAM_COUNT);
	 
	std::cout << "\nread write outputs - \n" << std::endl;

	std::string serialOut = getdOut();
	float temp1 = (memsize * 2e-6) / serial_time;
	float seqOut = nreps * temp1;

	std::cout << serialOut << seqOut<<std::endl;
	
    free(h_data_source);
	std::cout << "streams used = " << STREAM_COUNT << "\n";
	std::string randOut = getRandOut();
	free(h_data_sink);
	float temp2 = (memsize * 2e-6) / overlap_time;
	float randOutval = nreps * temp2;
	cudaFreeHost(h_data_in[1]);
		cudaFree(d_data_in[1]);

		cudaFreeHost(h_data_out[1]);
		cudaFree(d_data_out[1]);

		cudaStreamDestroy(stream[1]);
		cudaEventDestroy(cycleDone[1]);

		cudaFreeHost(h_data_in[2]);
		cudaFree(d_data_in[2]);

		cudaFreeHost(h_data_out[2]);
		cudaFree(d_data_out[2]);

		cudaStreamDestroy(stream[2]);
		cudaEventDestroy(cycleDone[2]);

		cudaFreeHost(h_data_in[3]);
		cudaFree(d_data_in[3]);

		cudaFreeHost(h_data_out[3]);
		cudaFree(d_data_out[3]);

		cudaStreamDestroy(stream[3]);
		cudaEventDestroy(cycleDone[3]);

		cudaFreeHost(h_data_in[0]);
		cudaFree(d_data_in[0]);

		cudaFreeHost(h_data_out[0]);
		cudaFree(d_data_out[0]);

		cudaStreamDestroy(stream[0]);
		cudaEventDestroy(cycleDone[0]);


	std::cout << randOut << randOutval << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	getch();

    exit(EXIT_SUCCESS);
}

float processWithStreams(int streams_used)
{

    int current_stream = 0;

    float time;

	std::string tpy = "this is similar to threading in cpu";

    cudaEventRecord(start, 0);

    for (int i=0; i<nreps; ++i)
    {
        int next_stream = (current_stream + 1) % streams_used;


		cudaError error =cudaEventSynchronize(cycleDone[next_stream]);

        memoryBenc<<<grid, block, 0, stream[current_stream]>>>(
            d_data_out[current_stream],
            d_data_in[current_stream],
            N,
            inner_reps);
        checkCudaErrors(cudaMemcpyAsync(
                            d_data_in[next_stream],
                            h_data_in[next_stream],
                            memsize,
                            cudaMemcpyHostToDevice,
                            stream[next_stream]));

        checkCudaErrors(cudaMemcpyAsync(
                            h_data_out[current_stream],
                            d_data_out[current_stream],
                            memsize,
                            cudaMemcpyDeviceToHost,
                            stream[current_stream]));

        checkCudaErrors(cudaEventRecord(
                            cycleDone[current_stream],
                            stream[current_stream]));
        current_stream = next_stream;
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    return time;
}
//reference: simplemulticopy form the nvidia samples (installation examples)