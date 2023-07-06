// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Initializer.h
// Weight/bias initializer classes.

#include <algorithm>
#include <random>
#include <chrono>
#include <cstdlib>

#include "Initializer.h"
#include "Matrix.h"

namespace Oliver {
	void ZerosInitializer::init(Matrix* m) {
		std::fill(m->buf(), &m->buf()[m->rows() * m->cols()], 0.0);
	}

	void OnesInitializer::init(Matrix* m) {
		std::fill(m->buf(), &m->buf()[m->rows() * m->cols()], 1.0);
	}

	void NormalInitializer::init(Matrix* m) {
		// Normal distribution randomness around 0, variance 1.
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<float> distribution(0.0, 1.0);

		for (int i = 0; i < m->rows() * m->cols(); ++i) {
			m->buf()[i] = distribution(generator);
		}
	}

	void HeInitializer::init(Matrix* m) {
		// Calculate the scale using the matrix size. We assume the number of inputs
		// is the same as the number of rows in the matrix.
		float std = sqrtf(2.0 / ((float)m->rows()));

		// Normal distribution randomness around 0, variance 1.
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<float> distribution(0.0, 1.0);

		for (int i = 0; i < m->rows() * m->cols(); ++i) {
			m->buf()[i] = distribution(generator) * std;
		}
	}
}