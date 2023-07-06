// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// EmptyLayer.cpp
// Empty layer class.

#include "EmptyLayer.h"
#include "Network.h"

namespace Oliver {
	EmptyLayer::EmptyLayer(unsigned int inputSize, unsigned int outputSize) : m_inputSize(inputSize), m_outputSize(outputSize), m_trainable(false) {
		if (m_inputSize == 0 || m_outputSize == 0) {
			throw NetworkException("input and output size must not be zero");
		}
	}

	unsigned int EmptyLayer::inputSize() {
		return m_inputSize;
	}

	unsigned int EmptyLayer::outputSize() {
		return m_outputSize;
	}

	void EmptyLayer::initTraining(OptimizerSettings* settings) {
		m_trainable = true;
	}

	void EmptyLayer::forward(Matrix* input, Matrix* output, int device) {
		// Ensure the sizes match.
		if (input->rows() != output->rows() || input->cols() != output->cols()) {
			throw NetworkException("invalid input and output matrix sizes for layer");
		}

		memcpy(output->buf(), input->buf(), input->rows() * input->cols() * sizeof(float));
	}

	void EmptyLayer::backward(Matrix* outputGrad, Matrix* prevGrad, int device) {
		if (!m_trainable) {
			throw NetworkException("layer not initialized for training");
		}

		// Ensure the sizes match.
		if (prevGrad->rows() != outputGrad->rows() || prevGrad->cols() != outputGrad->cols()) {
			throw NetworkException("invalid input, previous gradient, and output gradient matrix sizes for layer");
		}

		memcpy(prevGrad->buf(), outputGrad->buf(), outputGrad->rows() * outputGrad->cols() * sizeof(float));
	}

	void EmptyLayer::update(int device) {
		if (!m_trainable) {
			throw NetworkException("layer not initialized for training");
		}
	}
}