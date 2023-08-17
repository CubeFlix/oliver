#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "DenseLayer.h"
#include "Initializer.h"
#include "MeanSquaredLoss.h"
#include "Network.h"
#include "Model.h"
#include "SGDOptimizer.h"

#define SIZE 1
#define INPUTSIZE 5
#define OUTPUTSIZE 5

using namespace Oliver;

int main() {
	try {
		/* Oliver::Matrix* a = new Oliver::Matrix(SIZE, SIZE);
		Oliver::Matrix* b = new Oliver::Matrix(SIZE-1, SIZE);
		for (int i = 0; i < SIZE * (SIZE - 1); i++) {
			a->buf()[i] = i;
			b->buf()[i] = 2;
		}
		// Oliver::Matrix* b = a->copy();
		std::cout << "beginning gpu calculation...\n";
		Oliver::Matrix* c = a->dot(b, 0);
		// a->sub(1.0, 0);
		std::cout << "finished gpu calculation\n";

		for (int i = 0; i < SIZE * SIZE; i++) {
			std::cout << c->buf()[i] << " ";
		}

		float delta = 0;
		for (int i = 0; i < SIZE * SIZE; i++) {
			delta += abs(a->buf()[i] - (float)(i*2));
		}
		std::cout << "delta: " << delta << " " << a->buf()[24] << "\n";
		delete a;
		delete b;
		delete c;*/

		/*HeInitializer weightInit = HeInitializer();
		ZerosInitializer biasInit = ZerosInitializer();
		SGDOptimizerSettings optSettings = SGDOptimizerSettings(0.05);
		Layer* l = new DenseLayer(INPUTSIZE, OUTPUTSIZE, &weightInit, &biasInit);
		Loss* mse = new MeanSquaredLoss(OUTPUTSIZE);

		l->initTraining(&optSettings);
		mse->initTraining();

		Matrix* input = new Matrix(SIZE, INPUTSIZE);
		input->sub(input, 0);
		Matrix* output = new Matrix(SIZE, OUTPUTSIZE);
		Matrix* outGrad = new Matrix(SIZE, OUTPUTSIZE);
		Matrix* inGrad = new Matrix(SIZE, INPUTSIZE);
		Matrix* loss = new Matrix(SIZE, 1);
		Matrix* y = new Matrix(1, OUTPUTSIZE);

		input->buf()[0] = 1;
		input->buf()[1] = 2;
		input->buf()[2] = 3;
		input->buf()[3] = 4;
		input->buf()[4] = 5;

		y->buf()[0] = 2;
		y->buf()[1] = 4;
		y->buf()[2] = 6;
		y->buf()[3] = 8;
		y->buf()[4] = 10;

		l->forward(input, output, 0);
		std::cout << "forward layer done" << std::endl;
		mse->forward(output, y, loss, 0);
		std::cout << "forward done" << std::endl;
		mse->backward(y, outGrad, 0);
		l->backward(outGrad, inGrad, 0);
		l->update(0);

		std::cout << "loss: " << loss->buf()[0] << std::endl;
		std::cout << "out[0]: " << output->buf()[0] << std::endl;
		std::cout << "y[0]: " << y->buf()[0] << std::endl;

		l->forward(input, output, 0);
		std::cout << "forward layer done" << std::endl;
		mse->forward(output, y, loss, 0);
		std::cout << "forward done" << std::endl;
		mse->backward(y, outGrad, 0);
		l->backward(outGrad, inGrad, 0);
		l->update(0);

		std::cout << "loss: " << loss->buf()[0] << std::endl;
		std::cout << "out[0]: " << output->buf()[0] << std::endl;
		std::cout << "y[0]: " << y->buf()[0] << std::endl;

		l->forward(input, output, 0);
		std::cout << "forward layer done" << std::endl;
		mse->forward(output, y, loss, 0);
		std::cout << "forward done" << std::endl;
		mse->backward(y, outGrad, 0);
		l->backward(outGrad, inGrad, 0);
		l->update(0);

		std::cout << "loss: " << loss->buf()[0] << std::endl;
		std::cout << "out[0]: " << output->buf()[0] << std::endl;
		std::cout << "y[0]: " << y->buf()[0] << std::endl;

		delete input;
		delete output;
		delete outGrad;
		delete inGrad;
		delete loss;
		delete y;
		delete l;
		delete mse;*/

		HeInitializer weightInit = HeInitializer();
		ZerosInitializer biasInit = ZerosInitializer();

		Model m = Model();
		DenseLayer* l1 = new DenseLayer(1, 5, &weightInit, &biasInit);
		DenseLayer* l2 = new DenseLayer(5, 2, &weightInit, &biasInit);
		m.addLayer(l1);
		m.addLayer(l2);

		MeanSquaredLoss* loss = new MeanSquaredLoss(2);
		m.finalize(loss);

		OptimizerSettings* opt = &SGDOptimizerSettings(0.01);
		m.initTraining(opt);

		Matrix* input = new Matrix(5, 1);
		Matrix* y = new Matrix(5, 2);
		Matrix* outloss = new Matrix(5, 1);
		Matrix* output = new Matrix(5, 2);
		input->sub(input, 0);
		y->sub(y, 0);
		float avgl = m.forward(input, y, outloss, 0);
		m.backward(y, 0);
		m.predict(input, output, 0);

		std::cout << avgl;

		delete input;
		delete y;
		delete outloss;
		delete output;
	}
	catch (NetworkException e) {
		std::cout << e.what() << std::endl;
		std::cout << typeid(e).name() << std::endl;
	}
	return 0;
}