// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// DenseLayer.h
// Dense layer class.

#pragma once

#include "Layer.h"
#include "Initializer.h"
#include "Optimizer.h"

namespace Oliver {
	// Dense layer class. 
    class DenseLayer : public Layer {
    public:
		DenseLayer(unsigned int inputSize, unsigned int outputSize, Initializer* weightInit, Initializer* biasInit);
		~DenseLayer();

		unsigned int inputSize();
		unsigned int outputSize();

		void initTraining(OptimizerSettings* settings);
		void forward(Matrix* input, Matrix* output, int device);
		void backward(Matrix* outputGrad, Matrix* prevGrad, int device);
		void update(int device);
	private:
		unsigned int m_inputSize;
		unsigned int m_outputSize;
		Initializer* m_weightInit;
		Initializer* m_biasInit;
		Matrix* m_weights;
		Matrix* m_biases;
		Matrix* m_inputCache;
	public: Matrix* m_weightGrad;
	public: Matrix* m_biasGrad;
		Optimizer* m_weightOpt;
		Optimizer* m_biasOpt;

		bool m_trainable;
	};
}