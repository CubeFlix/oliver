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
		m_loss->initTraining();

		m_trainable = true;
	}

	void Model::predict(Matrix* input, Matrix* output, int device) {
		if (!m_finalized) {
			throw NetworkException("model not finalized");
		}

		// Check that the number of samples match.
		if (input->rows() != output->rows()) {
			throw NetworkException("number of samples must match in input and output matricies");
		}

		// Check that the input and output dimensions are correct.
		if (input->cols() != m_layers[0]->inputSize()) {
			throw NetworkException("invalid input matrix size");
		}
		if (output->cols() != m_layers.back()->outputSize()) {
			throw NetworkException("invalid output matrix size");
		}

		unsigned int samples = input->rows();
		Matrix* current_input = input->copy();

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

		memcpy(output->buf(), current_input->buf(), output->rows() * output->cols() * sizeof(float));
		delete current_input;
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
		if (loss->cols() != 1) {
			throw NetworkException("invalid loss matrix size");
		}

		unsigned int samples = input->rows();
		Matrix* current_input = input->copy();

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

	void Model::backward(Matrix* y, int device) {
		if (!m_finalized) {
			throw NetworkException("model not finalized");
		}
		if (!m_trainable) {
			throw NetworkException("model not initialized for training");
		}

		// Check that the y dimensions are correct.
		if (y->cols() != m_layers.back()->outputSize()) {
			throw NetworkException("invalid y matrix size");
		}

		unsigned int samples = y->rows();
		Matrix* current_grad = new Matrix(samples, m_loss->inputSize());

		// Perform the loss pass.
		m_loss->backward(y, current_grad, device);

		// TODO: handle cross-entropy loss with softmax

		// Perform the backward pass.
		for (std::vector<Layer*>::reverse_iterator iter = m_layers.rbegin(); iter != m_layers.rend(); iter++) {
			// Create the new gradient matrix and perform the backward pass.
			Matrix* current_new_grad = new Matrix(samples, (*iter)->inputSize());
			(*iter)->backward(current_grad, current_new_grad, device);

			// Re-create the gradient matrix with the new gradient data.
			delete current_grad;
			current_grad = current_new_grad->copy();
			delete current_new_grad;
		}

		delete current_grad;
	}

	void Model::train(Matrix* input, Matrix* y, unsigned int sample_size, unsigned int epochs, std::ostream* log, int device) {
		if (!m_finalized) {
			throw NetworkException("model not finalized");
		}
		if (!m_trainable) {
			throw NetworkException("model not initialized for training");
		}
		
		// Check that the number of samples match.
		unsigned int samples = input->rows();
		if (y->rows() != samples) {
			throw NetworkException("number of samples must match in input and y matricies");
		}

		// Check that the input and output dimensions are correct.
		if (input->cols() != m_layers[0]->inputSize()) {
			throw NetworkException("invalid input matrix size");
		}
		if (y->cols() != m_layers.back()->outputSize()) {
			throw NetworkException("invalid output matrix size");
		}

		// Start the training process.
		for (int epoch = 0; epoch < epochs; epoch++) {
			// Create the matricies used for the training.
			Matrix* current_input = new Matrix(sample_size, input->cols());
			Matrix* current_y = new Matrix(sample_size, y->cols());
			Matrix* current_loss = new Matrix(sample_size, 1);

			if (log) {
				*log << "Epoch " << epoch << "\n";
			}

			float acc_loss = 0.0;
			int acc_samples = 0;
			for (int current_sample = 0; current_sample < samples; current_sample += sample_size) {
				// If the buffer is too big to take in the remaining samples, create a smaller buffer.
				if (sample_size > samples - current_sample) {
					delete current_input;
					delete current_y;
					delete current_loss;
					current_input = new Matrix(samples - current_sample, input->cols());
					current_y = new Matrix(samples - current_sample, y->cols());
					current_loss = new Matrix(samples - current_sample, 1);
				}

				// Copy in the current input and y samples.
				memcpy(current_input->buf(), &input->buf()[input->cols() * current_sample], current_input->rows() * input->cols() * sizeof(float));
				memcpy(current_y->buf(), &y->buf()[y->cols() * current_sample], current_y->rows() * y->cols() * sizeof(float));

				if (log) {
					*log << "Sample " << current_sample << "/" << samples << "\n";
				}

				// Forward and backward pass.
				float loss = forward(current_input, current_y, current_loss, device);
				acc_loss += loss;
				acc_samples++;
				backward(current_y, device);

				// Optimize.
				for (std::vector<Layer*>::iterator iter = m_layers.begin(); iter < m_layers.end(); iter++) {
					(*iter)->update(device);
				}
			}

			if (log) {
				*log << "Current loss: " << acc_loss / (float)(acc_samples) << "\n";
			}

			delete current_input;
			delete current_y;
			delete current_loss;
		}
	}
}