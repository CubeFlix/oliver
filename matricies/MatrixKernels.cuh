// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// MatrixKernels.cu
// CUDA matrix operation kernels.

#pragma once

#include "cuda_runtime.h"
#include "Matrix.h"

namespace Oliver {
	class CUDAException : public std::exception {
	public:
		CUDAException(cudaError_t err) : m_err(err) {}
		const char* what();
	private:
		cudaError_t m_err;
	};

	// Allocate and copy a buffer to a device. If it fails, it will not deallocate it.
	cudaError_t cudaAllocCopy(void** devPtr, void* buf, size_t s);

	cudaError_t cudaAdd(Matrix* a, Matrix* b, int device);
	cudaError_t cudaScalarAdd(Matrix* a, float b, int device);
	cudaError_t cudaSub(Matrix* a, Matrix* b, int device);
	cudaError_t cudaScalarSub(Matrix* a, float b, int device);
	cudaError_t cudaMul(Matrix* a, Matrix* b, int device);
	cudaError_t cudaScalarMul(Matrix* a, float b, int device);
	cudaError_t cudaDiv(Matrix* a, Matrix* b, int device);
	cudaError_t cudaScalarDiv(Matrix* a, float b, int device);
	cudaError_t cudaScalarPow(Matrix* a, float b, int device);
	cudaError_t cudaNeg(Matrix* a, int device);
	cudaError_t cudaInv(Matrix* a, int device);
	cudaError_t cudaExp(Matrix* a, int device);
	cudaError_t cudaLog(Matrix* a, int device);
	cudaError_t cudaAddBias(Matrix* a, Matrix* b, int device);
	cudaError_t cudaScalarMax(Matrix* a, float b, int device);
	cudaError_t cudaTranspose(Matrix* a, Matrix* out, int device);
	cudaError_t cudaDot(Matrix* a, Matrix* b, Matrix* out, int device);
	cudaError_t cudaSum(Matrix* a, Matrix* out, enum SumDirection dir, int device);
}