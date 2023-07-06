// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Optimizer.h
// Base optimizer and settings class.

#pragma once

#include "Matrix.h"

namespace Oliver {
	class Optimizer {
	public:
		virtual ~Optimizer() = 0;
		virtual void update(int device) = 0;
	};

	class OptimizerSettings {
	public:
		virtual Optimizer* create(Matrix* x, Matrix* xGrad) = 0;
	};

	class SGDOptimizerSettings : public OptimizerSettings {
	public:
		SGDOptimizerSettings(const float learningRate);
		Optimizer* create(Matrix* x, Matrix* xGrad);
	private:
		const float m_learningRate;
	};

	// Stochastic gradient descent (SGD) optimizer.
	class SGDOptimizer : public virtual Optimizer {
	public:
		SGDOptimizer(Matrix* x, Matrix* xGrad, const float learningRate);

		void update(int device);
	private:
		Matrix* m_x;
		Matrix* m_xGrad;

		const float m_learningRate;
		float m_currentRate;
	};
}