// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// SGDOptimizer.cpp
// The stochastic gradient descent (SGD) optimizer.

#include "Optimizer.h"
#include "Matrix.h"
#include "Network.h"

namespace Oliver {
	SGDOptimizer::SGDOptimizer(Matrix* x, Matrix* xGrad, const float learningRate) : m_x(x), m_xGrad(xGrad), m_learningRate(learningRate), m_currentRate(learningRate) {
		if (x->rows() != xGrad->rows() || x->cols() != xGrad->cols()) {
			throw NetworkException("matrix sizes must match for optimizer");
		}
	}

	void SGDOptimizer::update(int device) {
		// TODO: momentum, decay, etc.
		Matrix* deltaX = m_x->copy();
		deltaX->mul(m_learningRate, device);
		m_xGrad->sub(deltaX, device);
		delete deltaX;
	}
}
