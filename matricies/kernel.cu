#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIZE_I 2048
#define SIZE_J 1024
#define SIZE_K 1000

cudaError_t addWithCuda(float* c, const float* a, const float* b, unsigned int i, unsigned int j, unsigned int k);

__global__ void addKernel(float*c, const float*a, const float*b, const unsigned int kSize)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    float total = 0;
    for (int k = 0; k < kSize; k++) {
        total += a[i * kSize + k] * b[k * blockDim.x + j];
    }
    c[i * blockDim.x + j] = total;
}

/*
int main()
{
    float *a = (float*)calloc(SIZE_I * SIZE_K * sizeof(float), 1);
    float *b = (float*)calloc(SIZE_J * SIZE_K * sizeof(float), 1);
    float *c = (float*)calloc(SIZE_I * SIZE_J * sizeof(float), 1);

    // Initialize test values.
    srand(time(NULL));
    for (int i = 0; i < SIZE_I * SIZE_K; i++) {
        a[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < SIZE_J * SIZE_K; i++) {
        b[i] = (float)(rand() % 100);
    }

    printf("starting GPU calculation\n");

    // Multiply matricies in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, SIZE_I, SIZE_J, SIZE_K);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("finished GPU calculation\n");

    float* c_cpu = (float*)calloc(SIZE_I * SIZE_J * sizeof(float), 1);
    for (int i = 0; i < SIZE_I; i++) {
        for (int j = 0; j < SIZE_J; j++) {
            for (int k = 0; k < SIZE_K; k++) {
                c_cpu[SIZE_J * i + j] += a[SIZE_K * i + k] * b[SIZE_J * k + j];
            }
        }
    }

    printf("finished CPU calculation\n");

    float delta = 0;
    for (int i = 0; i < SIZE_I * SIZE_J; i++) {
        delta += fabs(c[i] - c_cpu[i]);
    }
    
    printf("delta: %f\n", delta);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}*/

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float*c, const float*a, const float*b, unsigned int i, unsigned int j, unsigned int k)
{
    float*dev_a = 0;
    float*dev_b = 0;
    float*dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, i * j * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, i * k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, j * k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, i * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, j * k * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<i, j>>>(dev_c, dev_a, dev_b, k);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, i * j * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
