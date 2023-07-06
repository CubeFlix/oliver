#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "DenseLayer.h"
#include "Initializer.h"
#include "MeanSquaredLoss.h"
#include "Network.h"

#define SIZE 100
#define INPUTSIZE 10
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

		HeInitializer weightInit = HeInitializer();
		ZerosInitializer biasInit = ZerosInitializer();
		SGDOptimizerSettings optSettings = SGDOptimizerSettings(0.01);
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
		y->sub(y, 0);

		l->forward(input, output, 0);
		std::cout << "forward layer done" << std::endl;
		mse->forward(output, y, loss, 0);
		std::cout << "forward done" << std::endl;
		mse->backward(y, outGrad, 0);
		l->backward(outGrad, inGrad, 0);
		l->update(0);

		std::cout << loss->buf()[0] << " " << y->buf()[0];

		delete input;
		delete output;
		delete outGrad;
		delete inGrad;
		delete loss;
		delete y;
		delete l;
		delete mse;
	}
	catch (NetworkException e) {
		std::cout << e.what() << std::endl;
		std::cout << typeid(e).name() << std::endl;
	}
	return 0;
}