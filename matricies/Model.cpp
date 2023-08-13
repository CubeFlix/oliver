// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Model.cpp
// Network model class.

#include <vector>

#include "Model.h"
#include "Network.h"
#include "Layer.h"

namespace Oliver {
	Model::Model() : m_finalized(false), m_trainable(false), m_loss(NULL), m_opt(NULL) {

	}

	Model::~Model() {
		// Delete each layer.
		for (std::vector<Layer*>::iterator iter = m_layers.begin(); iter < m_layers.end(); iter++) {
			delete *iter;
		}
		m_layers.clear();

		// Delete the loss.
		if (m_loss) {
			delete m_loss;
		}

		// Delete the optimizer.
		if (m_opt) {
			delete m_opt;
		}
	}

	void Model::addLayer(Layer* l) {
		// If there are no layers yet, add the first one.
		if (m_layers.size() == 0) {
			m_layers.push_back(l);
			return;
		}

		// Ensure that the dimensions match.
		if (l->inputSize() != m_layers.back()->outputSize()) {
			throw NetworkException("layer input size must match previous layer output size");
		}
		m_layers.push_back(l);
	}

	void Model::finalize(Loss* l) {
		if (m_layers.size() == 0) {
			throw NetworkException("model must have at least one layer");
		}

		// Ensure that the loss dimension matches.
		if (l->inputSize() != m_layers.back()->outputSize()) {
			throw NetworkException("loss input size must match previous layer output size");
		}

		// Set the loss.
		m_loss = l;

		m_finalized = true;
	}

	void Model::initTraining(OptimizerSettings* opt) {
		if (!m_finalized) {
			throw NetworkException("model not finalized");
		}

		for (std::vector<Layer*>::iterator iter = m_layers.begin(); iter < m_layers.end(); iter++) {
			(*iter)->initTraining(opt);
		}

		m_trainable = true;
	}

	float Model::forward(Matrix* input, Matrix* y, Matrix* loss, int device) {
		if (!m_finalized) {
			throw NetworkException("model not finalized");
		}

		// Check that the number of samples match.
		if (input->rows() != y->rows()) {
			throw NetworkException("number of samples must match in input and y matricies");
		}
		if (input->rows() != loss->rows()) {
			throw NetworkException("number of samples must match in input and loss matricies");
		}

		// Check that the input and y dimensions are correct.
		if (input->cols() != m_layers[0]->inputSize()) {
			throw NetworkException("invalid input matrix size");
		}
		if (y->cols() != m_layers.back()->outputSize()) {
			throw NetworkException("invalid y matrix size");
		}

		// Check that the loss dimensions are correct.
		if (loss->cols() != m_layers.back()->outputSize()) {
			throw NetworkException("invalid loss matrix size");
		}

		unsigned int samples = input->rows();
		Matrix* current_input = new Matrix(samples, input->cols());

		// Perform the forward pass.
		for (std::vector<Layer*>::iterator iter = m_layers.begin(); iter < m_layers.end(); iter++) {
			// Create the output matrix and perform the forward pass.
			Matrix* current_output = new Matrix(samples, (*iter)->outputSize());
			(*iter)->forward(current_input, current_output, device);
			
			// Re-create the input matrix with the output matrix data.
			delete current_input;
			current_input = current_output->copy();
			delete current_output;
		}

		// TODO: handle cross-entropy loss with softmax

		// Perform the loss pass.
		float avg_loss = m_loss->forward(current_input, y, loss, device);
		delete current_input;
		return avg_loss;
	}
}