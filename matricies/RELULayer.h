// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// RELULayer.h
// RELU layer class.

#pragma once

#include "Layer.h"
#include "Initializer.h"
#include "Optimizer.h"

namespace Oliver {
	// RELU layer class. 
	class RELULayer : public Layer {
	public:
		RELULayer(unsigned int inputSize);
		~RELULayer();

		unsigned int inputSize();
		unsigned int outputSize();

		void initTraining(OptimizerSettings* settings);
		void forward(Matrix* input, Matrix* output, int device);
		void backward(Matrix* outputGrad, Matrix* prevGrad, int device);
		void update(int device);
	private:
		unsigned int m_inputSize;
		Matrix* m_inputCache;

		bool m_trainable;
	};
}