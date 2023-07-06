// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Optimizer.cpp
// Base optimizer and settings class.

#include <iostream>
#include "Optimizer.h"
#include "SGDOptimizer.h"

namespace Oliver {
	Optimizer::~Optimizer() {}

	SGDOptimizerSettings::SGDOptimizerSettings(const float learningRate) : m_learningRate(learningRate) {}

	Optimizer* SGDOptimizerSettings::create(Matrix* x, Matrix* xGrad) {
		Optimizer* opt = new SGDOptimizer(x, xGrad, m_learningRate);
		return opt;
	}
}