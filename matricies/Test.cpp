#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "DenseLayer.h"
#include "Initializer.h"
#include "MeanSquaredLoss.h"
#include "Network.h"
#include "Model.h"
#include "SGDOptimizer.h"
#include "RELULayer.h"
#include "MatrixKernels.cuh"

#define SIZE 10
#define INPUTSIZE 1
#define OUTPUTSIZE 1

using namespace Oliver;

void test_1() {
	HeInitializer weightInit = HeInitializer();
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
	Matrix* y = new Matrix(SIZE, OUTPUTSIZE);

	input->buf()[0] = 1;
	input->buf()[2] = 2;
	input->buf()[4] = 3;
	input->buf()[6] = 4;
	input->buf()[8] = 5;
	input->buf()[1] = 1;
	input->buf()[3] = 2;
	input->buf()[5] = 3;
	input->buf()[7] = 4;
	input->buf()[9] = 5;

	y->buf()[0] = 2;
	y->buf()[2] = 4;
	y->buf()[4] = 6;
	y->buf()[6] = 8;
	y->buf()[8] = 10;
	y->buf()[1] = 2;
	y->buf()[3] = 4;
	y->buf()[5] = 6;
	y->buf()[7] = 8;
	y->buf()[9] = 10;

	l->forward(input, output, 0);
	std::cout << "forward layer done" << std::endl;
	float ls = mse->forward(output, y, loss, 0);
	std::cout << "forward done" << std::endl;
	mse->backward(y, outGrad, 0);
	l->backward(outGrad, inGrad, 0);
	l->update(0);

	std::cout << "loss: " << ls << std::endl;
	std::cout << "out[0]: " << output->buf()[0] << std::endl;
	std::cout << "y[0]: " << y->buf()[0] << std::endl;

	l->forward(input, output, 0);
	std::cout << "forward layer done" << std::endl;
	ls = mse->forward(output, y, loss, 0);
	std::cout << "forward done" << std::endl;
	mse->backward(y, outGrad, 0);
	l->backward(outGrad, inGrad, 0);
	l->update(0);

	std::cout << "loss: " << ls << std::endl;
	std::cout << "out[0]: " << output->buf()[0] << std::endl;
	std::cout << "y[0]: " << y->buf()[0] << std::endl;

	l->forward(input, output, 0);
	std::cout << "forward layer done" << std::endl;
	mse->forward(output, y, loss, 0);
	std::cout << "forward done" << std::endl;
	mse->backward(y, outGrad, 0);
	l->backward(outGrad, inGrad, 0);
	l->update(0);

	std::cout << "loss: " << ls << std::endl;
	std::cout << "out[0]: " << output->buf()[0] << std::endl;
	std::cout << "y[0]: " << y->buf()[0] << std::endl;

	delete input;
	delete output;
	delete outGrad;
	delete inGrad;
	delete loss;
	delete y;
	delete l;
	delete mse;
}

void test_2() {
	HeInitializer weightInit = HeInitializer();
	ZerosInitializer biasInit = ZerosInitializer();

	Model m = Model();
	DenseLayer* l1 = new DenseLayer(INPUTSIZE, 1, &weightInit, &biasInit);
	// RELULayer* l2 = new RELULayer(1);
	// DenseLayer* l3 = new DenseLayer(1, OUTPUTSIZE, &weightInit, &biasInit);
	m.addLayer(l1);
	// m.addLayer(l2);
	// m.addLayer(l3);

	MeanSquaredLoss* loss = new MeanSquaredLoss(OUTPUTSIZE);
	m.finalize(loss);

	OptimizerSettings* opt = &SGDOptimizerSettings(0.05);
	m.initTraining(opt);

	Matrix* input = new Matrix(SIZE, INPUTSIZE);
	Matrix* y = new Matrix(SIZE, OUTPUTSIZE);
	Matrix* output = new Matrix(SIZE, OUTPUTSIZE);
	// Matrix* outLoss = new Matrix(SIZE, 1);
	input->buf()[0] = 1;
	input->buf()[2] = 2;
	input->buf()[4] = 3;
	input->buf()[6] = 4;
	input->buf()[8] = 5;
	input->buf()[1] = 1;
	input->buf()[3] = 2;
	input->buf()[5] = 3;
	input->buf()[7] = 4;
	input->buf()[9] = 5;

	y->buf()[0] = 2;
	y->buf()[2] = 4;
	y->buf()[4] = 6;
	y->buf()[6] = 8;
	y->buf()[8] = 10;
	y->buf()[1] = 2;
	y->buf()[3] = 4;
	y->buf()[5] = 6;
	y->buf()[7] = 8;
	y->buf()[9] = 10;
	/*float avgl = m.forward(input, y, outloss, 0);
	m.backward(y, 0);
	m.predict(input, output, 0);*/
	m.train(input, y, 10, 5, &std::cout, 0);
	m.predict(input, output, 0);
	std::cout << output->buf()[0] << " " << output->buf()[2] << " " << output->buf()[4] << " " << output->buf()[6] << std::endl;

	std::cout << l1->m_weights->buf()[0];
	std::cout << l1->m_biases->buf()[0];

	delete input;
	delete y;
	// delete outLoss;
	delete output;
}

void test_3() {
	HeInitializer weightInit = HeInitializer();
	ZerosInitializer biasInit = ZerosInitializer();

	Model m = Model();
	DenseLayer* l1 = new DenseLayer(INPUTSIZE, 2, &weightInit, &biasInit);
	RELULayer* l2 = new RELULayer(2);
	DenseLayer* l3 = new DenseLayer(2, OUTPUTSIZE, &weightInit, &biasInit);
	m.addLayer(l1);
	m.addLayer(l2);
	m.addLayer(l3);

	MeanSquaredLoss* loss = new MeanSquaredLoss(OUTPUTSIZE);
	m.finalize(loss);

	OptimizerSettings* opt = &SGDOptimizerSettings(0.01);
	m.initTraining(opt);

	Matrix* input = new Matrix(SIZE, INPUTSIZE);
	Matrix* y = new Matrix(SIZE, OUTPUTSIZE);
	Matrix* output = new Matrix(SIZE, OUTPUTSIZE);
	// Matrix* outLoss = new Matrix(SIZE, 1);
	input->buf()[0] = 1;
	input->buf()[2] = 2;
	input->buf()[4] = 3;
	input->buf()[6] = 4;
	input->buf()[8] = 5;
	input->buf()[1] = -1;
	input->buf()[3] = -2;
	input->buf()[5] = -3;
	input->buf()[7] = -4;
	input->buf()[9] = -5;

	y->buf()[0] = 2;
	y->buf()[2] = 4;
	y->buf()[4] = 6;
	y->buf()[6] = 8;
	y->buf()[8] = 10;
	y->buf()[1] = 2;
	y->buf()[3] = 4;
	y->buf()[5] = 6;
	y->buf()[7] = 8;
	y->buf()[9] = 10;
	/*float avgl = m.forward(input, y, outloss, 0);
	m.backward(y, 0);
	m.predict(input, output, 0);*/
	m.train(input, y, 10, 30, &std::cout, 0);
	m.predict(input, output, 0);
	std::cout << output->buf()[0] << " " << output->buf()[2] << " " << output->buf()[4] << " " << output->buf()[6] << std::endl;

	std::cout << l1->m_weights->buf()[0] << std::endl;
	std::cout << l1->m_biases->buf()[0];

	delete input;
	delete y;
	// delete outLoss;
	delete output;
}

int main() {
	try {
		/*Oliver::Matrix* a = new Oliver::Matrix(SIZE, SIZE);
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
		Matrix* y = new Matrix(SIZE, OUTPUTSIZE);

		input->buf()[0] = 0.1;
		input->buf()[2] = 0.2;
		input->buf()[4] = 0.3;
		input->buf()[6] = 0.4;
		input->buf()[8] = 0.5;
		input->buf()[1] = 0.1;
		input->buf()[3] = 0.2;
		input->buf()[5] = 0.3;
		input->buf()[7] = 0.4;
		input->buf()[9] = 0.5;

		y->buf()[0] = 0.2;
		y->buf()[2] = 0.4;
		y->buf()[4] = 0.6;
		y->buf()[6] = 0.8;
		y->buf()[8] = 1.0;
		y->buf()[1] = 0.2;
		y->buf()[3] = 0.4;
		y->buf()[5] = 0.6;
		y->buf()[7] = 0.8;
		y->buf()[9] = 1.0;

		l->forward(input, output, 0);
		std::cout << "forward layer done" << std::endl;
		float ls = mse->forward(output, y, loss, 0);
		std::cout << "forward done" << std::endl;
		mse->backward(y, outGrad, 0);
		l->backward(outGrad, inGrad, 0);
		l->update(0);

		std::cout << "loss: " << loss->buf()[0] << " " << ls << std::endl;
		std::cout << "out[0]: " << output->buf()[0] << std::endl;
		std::cout << "y[0]: " << y->buf()[0] << std::endl;

		l->forward(input, output, 0);
		std::cout << "forward layer done" << std::endl;
		ls = mse->forward(output, y, loss, 0);
		std::cout << "forward done" << std::endl;
		mse->backward(y, outGrad, 0);
		l->backward(outGrad, inGrad, 0);
		l->update(0);

		std::cout << "loss: " << loss->buf()[0] << " " << ls << std::endl;
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

		// test_1();
		// test_2();
		test_3();

		/*RELULayer relu = RELULayer(1);
		relu.initTraining(NULL);
		Matrix in = Matrix(5, 1);
		Matrix out = Matrix(5, 1);
		Matrix din = Matrix(5, 1);
		in.buf()[0] = 1.0;
		in.buf()[1] = 2.0;
		in.buf()[2] = -1.0;
		in.buf()[3] = -5.0;
		in.buf()[4] = 0.0;
		relu.forward(&in, &out, 0);
		relu.backward(&in, &din, 0);
		std::cout << din.buf()[0];*/
	}
	catch (NetworkException e) {
		std::cout << e.what() << std::endl;
		std::cout << typeid(e).name() << std::endl;
	}
	return 0;
}