// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// RELULayer.cpp
// RELU layer class.

#include "RELULayer.h"
#include "Initializer.h"
#include "Network.h"

namespace Oliver {
	RELULayer::RELULayer(unsigned int inputSize)
		: m_trainable(false), m_inputSize(inputSize) {
		if (m_inputSize == 0) {
			throw NetworkException("input size must not be zero");
		}

		m_inputCache = NULL;
	}

	RELULayer::~RELULayer() {
		if (m_trainable) {
			delete m_inputCache;
		}
	}

	unsigned int RELULayer::inputSize() {
		return m_inputSize;
	}

	unsigned int RELULayer::outputSize() {
		return m_inputSize;
	}

	void RELULayer::initTraining(OptimizerSettings* settings) {
		m_trainable = true;
	}

	void RELULayer::forward(Matrix* input, Matrix* output, int device) {
		// Ensure the sizes match.
		if (input->rows() != output->rows() || input->cols() != m_inputSize || output->cols() != m_inputSize) {
			throw NetworkException("invalid input and output matrix sizes for layer");
		}

		if (m_trainable) {
			// If we need to train, cache the input.
			m_inputCache = input->copy();
		}

		memcpy(output->buf(), input->buf(), input->rows() * input->cols() * sizeof(float));
		output->max(0.0, device);
	}

	void RELULayer::backward(Matrix* outputGrad, Matrix* prevGrad, int device) {
		if (!m_trainable) {
			throw NetworkException("layer not initialized for training");
		}
		if (!m_inputCache) {
			throw NetworkException("forward pass has not been run");
		}

		// Ensure the sizes match.
		if (m_inputCache->rows() != prevGrad->rows() || prevGrad->rows() != outputGrad->rows()) {
			throw NetworkException("invalid input, previous gradient, and output gradient matrix sizes for layer");
		}
		if (m_inputCache->cols() != m_inputSize || prevGrad->cols() != m_inputSize || outputGrad->cols() != m_inputSize) {
			throw NetworkException("invalid input, previous gradient, and output gradient matrix sizes for layer");
		}

		// Calculate max(x, 0)/max(x, 0) * grad.
		memcpy(prevGrad->buf(), m_inputCache->buf(), m_inputCache->rows() * m_inputCache->cols() * sizeof(float));
		prevGrad->max(0.0, device);
		prevGrad->div(prevGrad, device);
		prevGrad->mul(outputGrad, device);
		
		delete m_inputCache;
		m_inputCache = NULL;
	}

	void RELULayer::update(int device) {
		if (!m_trainable) {
			throw NetworkException("layer not initialized for training");
		}
	}
}