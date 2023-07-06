// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// DenseLayer.cpp
// Dense layer class.

#include "DenseLayer.h"
#include "Initializer.h"
#include "Network.h"

namespace Oliver {
	DenseLayer::DenseLayer(unsigned int inputSize, unsigned int outputSize, Initializer* weightInit, Initializer* biasInit) 
			: m_trainable(false), m_inputSize(inputSize), m_outputSize(outputSize), m_weightInit(weightInit), m_biasInit(biasInit) {
		if (m_inputSize == 0 || m_outputSize == 0) {
			throw NetworkException("input and output size must not be zero");
		}

		// Allocate the matricies.
		m_weights = new Matrix(m_inputSize, m_outputSize);
		m_biases = new Matrix(1, m_outputSize);

		m_inputCache = NULL;
		m_weightGrad = NULL;
		m_weightOpt = NULL;
		m_biasGrad = NULL;
		m_biasOpt = NULL;
	}

	DenseLayer::~DenseLayer() {
		delete m_weights;
		delete m_biases;
		if (m_trainable) {
			delete m_inputCache;
			delete m_weightGrad;
			delete m_biasGrad;
			delete m_weightOpt;
			delete m_biasOpt;
		}
	}

	unsigned int DenseLayer::inputSize() {
		return m_inputSize;
	}

	unsigned int DenseLayer::outputSize() {
		return m_outputSize;
	}

	void DenseLayer::initTraining(OptimizerSettings* settings) {
		// Allocate the matricies.
		m_weightGrad = new Matrix(m_inputSize, m_outputSize);
		m_biasGrad = new Matrix(1, m_outputSize);

		// Initialize the weights and biases.
		m_weightInit->init(m_weights);
		m_biasInit->init(m_biases);

		// Create the optimizers.
		m_weightOpt = settings->create(m_weights, m_weightGrad);
		m_biasOpt = settings->create(m_biases, m_biasGrad);

		m_trainable = true;
	}

	void DenseLayer::forward(Matrix* input, Matrix* output, int device) {
		// Ensure the sizes match.
		if (input->rows() != output->rows() || input->cols() != m_inputSize || output->cols() != m_outputSize) {
			throw NetworkException("invalid input and output matrix sizes for layer");
		}

		if (m_trainable) {
			// If we need to train, cache the input.
			m_inputCache = input->copy();
		}

		dot(input, m_weights, output, device);
		output->addBias(m_biases, device);
	}

	void DenseLayer::backward(Matrix* outputGrad, Matrix* prevGrad, int device) {
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
		if (m_inputCache->cols() != m_inputSize || prevGrad->cols() != m_inputSize || outputGrad->cols() != m_outputSize) {
			throw NetworkException("invalid input, previous gradient, and output gradient matrix sizes for layer");
		}

		Matrix* weightT = m_weights->transpose(device);
		dot(outputGrad, weightT, prevGrad, device);

		Matrix* xT = m_inputCache->transpose(device);
		dot(xT, outputGrad, m_weightGrad, device);

		sum(outputGrad, m_biasGrad, SumOverColumns, device);
		delete weightT;
		delete xT;
		delete m_inputCache;
		m_inputCache = NULL;
	}

	void DenseLayer::update(int device) {
		if (!m_trainable) {
			throw NetworkException("layer not initialized for training");
		}
		m_weightOpt->update(device);
		m_biasOpt->update(device);
	}
}