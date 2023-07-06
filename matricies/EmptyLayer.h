// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// EmptyLayer.h
// Empty layer class.

#pragma once

#include "Layer.h"

namespace Oliver {
	class EmptyLayer : public Layer {
	public:
		EmptyLayer(unsigned int inputSize, unsigned int outputSize);
	
		unsigned int inputSize();
		unsigned int outputSize();

		void initTraining(OptimizerSettings* settings);
		void forward(Matrix* input, Matrix* output, int device);
		void backward(Matrix* outputGrad, Matrix* prevGrad, int device);
		void update(int device);
	private:
		unsigned int m_inputSize;
		unsigned int m_outputSize;
		
		bool m_trainable;
	};
}