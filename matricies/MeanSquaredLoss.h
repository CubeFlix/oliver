// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// MeanSquaredLoss.h
// Mean squared loss function class.

#pragma once

#include "Loss.h"

namespace Oliver {
	// Mean squared loss class.
	class MeanSquaredLoss : public Loss {
	public:
		MeanSquaredLoss(unsigned int inputSize);
		unsigned int inputSize();

		void initTraining();
		float forward(Matrix* input, Matrix* y, Matrix* outputLoss, int device);
		void backward(Matrix* y, Matrix* prevGrad, int device);
	private:
		unsigned int m_inputSize;

		Matrix* m_inputCache;

		bool m_trainable;
	};
}
