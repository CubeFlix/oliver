// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Layer.h
// Base layer class.

#pragma once

#include "Matrix.h"
#include "Optimizer.h"

namespace Oliver {
	// Base layer class.
	class Layer {
	public:
		virtual ~Layer() = 0;

		virtual unsigned int inputSize() = 0;
		virtual unsigned int outputSize() = 0;
		
		virtual void initTraining(OptimizerSettings* settings) = 0;
		virtual void forward(Matrix* input, Matrix* output, int device) = 0;
		virtual void backward(Matrix* outputGrad, Matrix* prevGrad, int device) = 0;
		virtual void update(int device) = 0;
	};
}

