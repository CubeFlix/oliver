// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Network.cpp
// Neural network.

#include "Network.h"

namespace Oliver {
	char* NetworkException::what() {
		return m_message;
	}
}