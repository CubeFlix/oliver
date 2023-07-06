// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Loss.h
// Base loss function class.

#pragma once

#include "Matrix.h"

namespace Oliver {
	// Loss base class.
	class Loss {
	public:
		virtual ~Loss() = 0;

		virtual unsigned int inputSize() = 0;

		virtual void initTraining() = 0;
		virtual float forward(Matrix* input, Matrix* y, Matrix* outputLoss, int device) = 0;
		virtual void backward(Matrix* y, Matrix* prevGrad, int device) = 0;
	};
}
