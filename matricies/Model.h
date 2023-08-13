// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Model.h
// Network model class.

#pragma once

#include <vector>
#include <memory>

#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"

namespace Oliver {
	// Network model class.
	class Model {
	public:
		Model();
		~Model();
		void initTraining(OptimizerSettings* opt);
		void addLayer(Layer* l);
		void finalize(Loss* l);
		void predict(Matrix* input, Matrix* output, int device);
		float forward(Matrix* input, Matrix* y, Matrix* loss, int device);
		void backward(Matrix* y, Matrix* loss, int device);
		void train(Matrix* input, Matrix* y, unsigned int sample_size, unsigned int epochs, int device);
	private:
		std::vector<Layer*> m_layers;
		Loss* m_loss;
		Optimizer* m_opt;

		bool m_finalized;
		bool m_trainable;
	};
}
