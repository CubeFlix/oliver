// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Matrix.h
// Matrix and matrix operations.

#pragma once

#include <iostream>

namespace Oliver {
	class MatrixException : public std::exception {
	public:
		MatrixException(char* msg) : m_message(msg) {}
		char* what();
	private:
		char* m_message;
	};

	enum SumDirection {
		SumOverRows,
		SumOverColumns
	};

	class Matrix {
	public:
		Matrix(unsigned const int rows, unsigned const int cols);
		~Matrix();

		unsigned const int rows();
		unsigned const int cols();
		float* buf();

		void add(Matrix *x, int device);
		void add(float x, int device);
		void sub(Matrix* x, int device);
		void sub(float x, int device);
		void mul(Matrix* x, int device);
		void mul(float x, int device);
		void div(Matrix* x, int device);
		void div(float x, int device);
		void pow(float x, int device);
		void neg(int device);
		void inv(int device);
		void exp(int device);
		void log(int device);
		void heaviside(int device);
		void addBias(Matrix* x, int device);
		void max(float x, int device);

		Matrix* transpose(int device);
		Matrix* dot(Matrix* x, int device);
		Matrix* sum(enum SumDirection dir, int device);

		Matrix* copy();
	private:
		unsigned const int m_rows;
		unsigned const int m_cols;
		float* m_buf;
	};

	void dot(Matrix* a, Matrix* b, Matrix* out, int device);
	void transpose(Matrix* a, Matrix* out, int device);
	void sum(Matrix* a, Matrix* out, enum SumDirection dir, int device);
}