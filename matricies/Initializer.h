// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Initializer.h
// Weight/bias initializer classes.

#pragma once

#include "Matrix.h"

namespace Oliver {
	class Initializer {
	public:
		virtual void init(Matrix* x) = 0;
	};

	class ZerosInitializer : public Initializer {
	public:
		void init(Matrix* x);
	};

	class OnesInitializer : public Initializer {
	public:
		void init(Matrix* x);
	};

	class NormalInitializer : public Initializer {
	public:
		void init(Matrix* x);
	};

	class HeInitializer : public Initializer {
	public:
		void init(Matrix* x);
	};
}