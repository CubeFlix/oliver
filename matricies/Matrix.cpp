// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Matrix.cpp
// Matrix and matrix operations.

#include "Matrix.h"
#include "MatrixKernels.cuh"

namespace Oliver {
	char* MatrixException::what() {
		return m_message;
	}

	Matrix::Matrix(unsigned const int rows, unsigned const int cols) : m_rows(rows), m_cols(cols) {
		m_buf = new float[m_rows * m_cols];
	}

	Matrix::~Matrix() {
		delete[] m_buf;
	}

	unsigned const int Matrix::rows() {
		return m_rows;
	}

	unsigned const int Matrix::cols() {
		return m_cols;
	}

	float* Matrix::buf() {
		return m_buf;
	}

	void Matrix::add(Matrix *x, int device) {
		// Ensure the dimensions match.
		if (m_rows != x->rows() || m_cols != x->cols()) {
			throw MatrixException("dimensions do not match for matrix-matrix add");
		}

		cudaError_t cudaError = cudaAdd(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::add(float x, int device) {
		cudaError_t cudaError = cudaScalarAdd(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::sub(Matrix* x, int device) {
		// Ensure the dimensions match.
		if (m_rows != x->rows() || m_cols != x->cols()) {
			throw MatrixException("dimensions do not match for matrix-matrix sub");
		}

		cudaError_t cudaError = cudaSub(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::sub(float x, int device) {
		cudaError_t cudaError = cudaScalarSub(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::mul(Matrix* x, int device) {
		// Ensure the dimensions match.
		if (m_rows != x->rows() || m_cols != x->cols()) {
			throw MatrixException("dimensions do not match for matrix-matrix mul");
		}

		cudaError_t cudaError = cudaMul(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::mul(float x, int device) {
		cudaError_t cudaError = cudaScalarMul(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::div(Matrix* x, int device) {
		// Ensure the dimensions match.
		if (m_rows != x->rows() || m_cols != x->cols()) {
			throw MatrixException("dimensions do not match for matrix-matrix div");
		}

		cudaError_t cudaError = cudaDiv(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::div(float x, int device) {
		cudaError_t cudaError = cudaScalarDiv(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::pow(float x, int device) {
		cudaError_t cudaError = cudaScalarPow(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::neg(int device) {
		cudaError_t cudaError = cudaNeg(this, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::inv(int device) {
		cudaError_t cudaError = cudaInv(this, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::exp(int device) {
		cudaError_t cudaError = cudaExp(this, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::log(int device) {
		cudaError_t cudaError = cudaLog(this, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::addBias(Matrix* x, int device) {
		// Ensure the dimensions match.
		if (m_cols != x->cols()) {
			throw MatrixException("bias matrix size must match");
		}
		if (x->rows() != 1) {
			throw MatrixException("bias matrix must have one row");
		}

		cudaError_t cudaError = cudaAddBias(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void Matrix::max(float x, int device) {
		cudaError_t cudaError = cudaScalarMax(this, x, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	Matrix* Matrix::transpose(int device) {
		// Create the output matrix.
		Matrix* out = new Matrix(m_cols, m_rows);
		cudaError_t cudaError = cudaTranspose(this, out, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}

		return out;
	}

	Matrix* Matrix::dot(Matrix* x, int device) {
		// Ensure the dimensions match.
		if (m_cols != x->rows()) {
			throw MatrixException("dimensions do not match for matrix-matrix dot product");
		}

		// Create the new matrix.
		Matrix* out = new Matrix(m_rows, x->cols());

		cudaError_t cudaError = cudaDot(this, x, out, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}

		return out;
	}

	Matrix* Matrix::sum(enum SumDirection dir, int device) {
		// Create the new matrix.
		Matrix* out;
		switch (dir) {
		case SumOverRows:
			out = new Matrix(m_rows, 1);
			break;
		case SumOverColumns:
			out = new Matrix(1, m_cols);
			break;
		default:
			throw MatrixException("invalid sum direction");
		}

		// This should never throw a invalid sum direction because we already checked
		// at the beginning.
		cudaError_t cudaError = cudaSum(this, out, dir, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}

		return out;
	}

	Matrix* Matrix::copy() {
		Matrix* c = new Matrix(m_rows, m_cols);
		std::memcpy(c->buf(), m_buf, m_rows * m_cols * sizeof(float));
		return c;
	}

	void dot(Matrix* a, Matrix* b, Matrix* out, int device) {
		// Ensure the dimensions match.
		if (a->cols() != b->rows()) {
			throw MatrixException("dimensions do not match for matrix-matrix dot product");
		}
		if (out->rows() != a->rows() || out->cols() != b->cols()) {
			throw MatrixException("output matrix dimensions are incorrect");
		}

		cudaError_t cudaError = cudaDot(a, b, out, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void transpose(Matrix* a, Matrix* out, int device) {
		// Ensure the dimensions match.
		if (a->rows() != out->cols() || a->cols() != out->rows()) {
			throw MatrixException("output matrix dimensions are incorrect");
		}

		cudaError_t cudaError = cudaTranspose(a, out, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}

	void sum(Matrix* a, Matrix* out, enum SumDirection dir, int device) {
		switch (dir) {
		case SumOverRows:
			if (out->rows() != a->rows() || out->cols() != 1) {
				throw MatrixException("output matrix dimensions are incorrect");
			}
			break;
		case SumOverColumns:
			if (out->rows() != 1 || out->cols() != a->cols()) {
				throw MatrixException("output matrix dimensions are incorrect");
			}
			break;
		default:
			throw MatrixException("invalid sum direction");
		}

		// This should never throw a invalid sum direction because we already checked
		// at the beginning.
		cudaError_t cudaError = cudaSum(a, out, dir, device);
		if (cudaError != cudaSuccess) {
			throw CUDAException(cudaError);
		}
	}
}