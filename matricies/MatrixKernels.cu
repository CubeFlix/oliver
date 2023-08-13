// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// MatrixKernels.cu
// CUDA matrix operation kernels.

#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matrix.h"
#include "MatrixKernels.cuh"

#define BLOCK_SIZE 32

namespace Oliver {
    const char* CUDAException::what() {
        return cudaGetErrorString(m_err);
    }

    // Allocate and copy a buffer to a device. If it fails, it will not deallocate it.
    cudaError_t cudaAllocCopy(void **devPtr, void *buf, size_t s) {
        cudaError_t cudaStatus = cudaMalloc(devPtr, s);
        if (cudaStatus != cudaSuccess) {
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(*devPtr, buf, s, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            return cudaStatus;
        }
        return cudaStatus;
    }

    // Matrix addition kernel.
    __global__ void addKernel(float* a, const float* b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] + b[i];
        }
    }

    // GPU matrix addition function.
    cudaError_t cudaAdd(Matrix *a, Matrix *b, int device) {
        float* dev_a = 0;
        float* dev_b = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_b), b->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_b);

        return cudaStatus;
    }

    // Matrix scalar addition kernel.
    __global__ void addScalarKernel(float* a, const float b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] + b;
        }
    }

    // GPU matrix scalar addition function.
    cudaError_t cudaScalarAdd(Matrix* a, float b, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        addScalarKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix subtraction kernel.
    __global__ void subKernel(float* a, const float* b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] - b[i];
        }
    }

    // GPU matrix subtraction function.
    cudaError_t cudaSub(Matrix* a, Matrix* b, int device) {
        float* dev_a = 0;
        float* dev_b = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_b), b->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        subKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_b);

        return cudaStatus;
    }

    // Matrix scalar subtraction kernel.
    __global__ void subScalarKernel(float* a, const float b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] - b;
        }
    }

    // GPU matrix scalar subtraction function.
    cudaError_t cudaScalarSub(Matrix* a, float b, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        subScalarKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix multiplication kernel.
    __global__ void mulKernel(float* a, const float* b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] * b[i];
        }
    }

    // GPU matrix multiplication function.
    cudaError_t cudaMul(Matrix* a, Matrix* b, int device) {
        float* dev_a = 0;
        float* dev_b = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_b), b->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        mulKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_b);

        return cudaStatus;
    }

    // Matrix scalar multiplication kernel.
    __global__ void mulScalarKernel(float* a, const float b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] * b;
        }
    }

    // GPU matrix scalar multiplication function.
    cudaError_t cudaScalarMul(Matrix* a, float b, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        mulScalarKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix division kernel.
    __global__ void divKernel(float* a, const float* b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] / b[i];
        }
    }

    // GPU matrix division function.
    cudaError_t cudaDiv(Matrix* a, Matrix* b, int device) {
        float* dev_a = 0;
        float* dev_b = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_b), b->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        divKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_b);

        return cudaStatus;
    }

    // Matrix scalar division kernel.
    __global__ void divScalarKernel(float* a, const float b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] / b;
        }
    }

    // GPU matrix scalar division function.
    cudaError_t cudaScalarDiv(Matrix* a, float b, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        divScalarKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix scalar exponentiation kernel.
    __global__ void powScalarKernel(float* a, const float b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = pow(a[i], b);
        }
    }

    // GPU matrix scalar exponentiation function.
    cudaError_t cudaScalarPow(Matrix* a, float b, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        powScalarKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix negation kernel.
    __global__ void negKernel(float* a, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = -a[i];
        }
    }

    // GPU matrix negation function.
    cudaError_t cudaNeg(Matrix* a, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        negKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix inverse kernel.
    __global__ void invKernel(float* a, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = 1.0 / a[i];
        }
    }

    // GPU matrix inverse function.
    cudaError_t cudaInv(Matrix* a, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        invKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix exp kernel.
    __global__ void expKernel(float* a, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = exp(a[i]);
        }
    }

    // GPU matrix exp function.
    cudaError_t cudaExp(Matrix* a, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        expKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix log kernel.
    __global__ void logKernel(float* a, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = log(a[i]);
        }
    }

    // GPU matrix log function.
    cudaError_t cudaLog(Matrix* a, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        logKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix heaviside kernel.
    __global__ void heavisideKernel(float* a, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            if (a[i] > 0.0) {
                a[i] = 1.0;
            }
            else {
                a[i] = 0.0;
            }
        }
    }

    // GPU matrix heaviside function.
    cudaError_t cudaHeaviside(Matrix* a, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        heavisideKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix add bias kernel.
    __global__ void addBiasKernel(float* a, const float* b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = a[i] + b[colIdx];
        }
    }

    // GPU matrix add bias function.
    cudaError_t cudaAddBias(Matrix* a, Matrix* b, int device) {
        float* dev_a = 0;
        float* dev_b = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_b), b->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        addBiasKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_b);

        return cudaStatus;
    }

    // Matrix scalar max kernel.
    __global__ void maxScalarKernel(float* a, const float b, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            int i = colIdx + rowIdx * cols;
            a[i] = fmaxf(a[i], b);
        }
    }

    // GPU matrix scalar max function.
    cudaError_t cudaScalarMax(Matrix* a, float b, int device) {
        float* dev_a = 0;
        size_t s = a->rows() * a->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), s);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        maxScalarKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, b, a->rows(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(a->buf(), dev_a, s, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);

        return cudaStatus;
    }

    // Matrix transpose kernel.
    __global__ void transposeKernel(float* a, float* out, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            out[rowIdx * cols + colIdx] = a[colIdx * rows + rowIdx];
        }
    }

    // GPU matrix transpose function.
    cudaError_t cudaTranspose(Matrix* a, Matrix* out, int device) {
        float* dev_a = 0;
        float* dev_out = 0;
        size_t size_a = a->rows() * a->cols() * sizeof(float);
        size_t size_out = out->rows() * out->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), size_a);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMalloc(((void**)&dev_out), size_out);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((out->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (out->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        transposeKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_out, out->rows(), out->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(out->buf(), dev_out, size_out, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_out);

        return cudaStatus;
    }
    
    // Matrix dot product kernel.
    __global__ void dotKernel(float* a, float* b, float* out, const unsigned int rows, const unsigned int cols, const unsigned int n) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            float sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += a[rowIdx * n + j] * b[j * cols + colIdx];
            }
            out[rowIdx * cols + colIdx] = sum;
        }
    }

    // GPU matrix dot product function.
    cudaError_t cudaDot(Matrix* a, Matrix* b, Matrix* out, int device) {
        float* dev_a = 0;
        float* dev_b = 0;
        float* dev_out = 0;
        size_t size_a = a->rows() * a->cols() * sizeof(float);
        size_t size_b = b->rows() * b->cols() * sizeof(float);
        size_t size_out = out->rows() * out->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), size_a);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_b), b->buf(), size_b);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMalloc(((void**)&dev_out), size_out);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((out->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (out->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        dotKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_out, out->rows(), out->cols(), a->cols());

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(out->buf(), dev_out, size_out, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_out);

        return cudaStatus;
    }

    // Matrix row sum kernel.
    __global__ void sumOverRowsKernel(float* a, float* out, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            atomicAdd(&out[rowIdx], a[rowIdx * cols + colIdx]);
        }
    }

    // Matrix column sum kernel.
    __global__ void sumOverColumnsKernel(float* a, float* out, const unsigned int rows, const unsigned int cols) {
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

        if (rowIdx < rows && colIdx < cols) {
            atomicAdd(&out[colIdx], a[rowIdx * cols + colIdx]);
        }
    }

    // GPU matrix sum function.
    cudaError_t cudaSum(Matrix* a, Matrix* out, enum SumDirection dir, int device) {
        float* dev_a = 0;
        float* dev_out = 0;
        size_t size_a = a->rows() * a->cols() * sizeof(float);
        size_t size_out = out->rows() * out->cols() * sizeof(float);
        cudaError_t cudaStatus;

        // Clear the output matrix.
        std::fill(out->buf(), &out->buf()[out->rows() * out->cols()], 0.0);

        // Move data to the device.
        cudaStatus = cudaSetDevice(device);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaAllocCopy(((void**)&dev_a), a->buf(), size_a);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMalloc(((void**)&dev_out), size_out);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

        // Calculate the thread and block dimensions.
        const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid((a->cols() + BLOCK_SIZE - 1) / BLOCK_SIZE, (a->rows() + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Run the kernel.
        switch (dir) {
        case SumOverRows:
            sumOverRowsKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_out, a->rows(), a->cols());
            break;
        case SumOverColumns:
            sumOverColumnsKernel << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_out, a->rows(), a->cols());
            break;
        }

        // Clean up.
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }
        cudaStatus = cudaMemcpy(out->buf(), dev_out, size_out, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            goto Error;
        }

    Error:
        cudaFree(dev_a);
        cudaFree(dev_out);

        return cudaStatus;
    }
}