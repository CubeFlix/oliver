// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// MeanSquaredLoss.cpp
// Mean squared loss function class.

#include "MeanSquaredLoss.h"
#include "Network.h"

namespace Oliver {
	MeanSquaredLoss::MeanSquaredLoss(unsigned int inputSize) : m_inputSize(inputSize), m_trainable(false), m_inputCache(NULL) {
		if (m_inputSize == 0) {
			throw NetworkException("input size must not be zero");
		}
	};

	unsigned int MeanSquaredLoss::inputSize() {
		return m_inputSize;
	}

	void MeanSquaredLoss::initTraining() {
		m_trainable = true;
	};

	float MeanSquaredLoss::forward(Matrix* input, Matrix* y, Matrix* outputLoss, int device) {
		// Check that the input size is correct.
		if (input->cols() != m_inputSize || y->cols() != m_inputSize) {
			throw NetworkException("invalid loss input size");
		}

		// Check that the y dimensions match.
		if (y->rows() != input->rows() || input->cols() != y->cols()) {
			throw NetworkException("invalid loss y dimensions");
		}

		// Check that the output matrix dimensions match.
		if (outputLoss->rows() != input->rows() || outputLoss->cols() != 1) {
			throw NetworkException("invalid loss output matrix dimensions");
		}

		m_inputCache = input->copy();
		
		// Calculate (y - y_hat)^2.
		Matrix* temp = input->copy();
		temp->neg(device);
		temp->add(y, device);
		temp->pow(2.0, device);

		// Average the loss.
		sum(temp, outputLoss, SumOverRows, device);
		outputLoss->div((float)m_inputSize, device);

		// Calculate the total averaged loss.
		Matrix* total = new Matrix(1, 1);
		sum(outputLoss, total, SumOverColumns, device);
		float avgLoss = total->buf()[0];
		
		delete temp;
		delete total;

		return avgLoss;
	}

	void MeanSquaredLoss::backward(Matrix* y, Matrix* prevGrad, int device) {
		if (!m_trainable) {
			throw NetworkException("layer not initialized for training");
		}
		if (!m_inputCache) {
			throw NetworkException("forward pass has not been run");
		}

		// Check that the prevGrad size is correct.
		if (prevGrad->cols() != m_inputSize || y->cols() != m_inputSize) {
			throw NetworkException("invalid loss input size");
		}

		// Check that the y dimensions match.
		if (y->rows() != prevGrad->rows() || prevGrad->cols() != y->cols()) {
			throw NetworkException("invalid loss y dimensions");
		}

		// Calculate -2 * (y - y_hat), average and normalize.
		memcpy(prevGrad->buf(), m_inputCache->buf(), m_inputCache->rows() * m_inputCache->cols() * sizeof(float));
		prevGrad->neg(device);
		prevGrad->add(y, device);
		prevGrad->mul(-2.0 / ((float)m_inputSize) / ((float)m_inputCache->rows()), device);
		
		delete m_inputCache;
		m_inputCache = NULL;
	}
}