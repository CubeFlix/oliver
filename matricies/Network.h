// Oliver - machine learning library.
// written by cubeflix - https://github.com/cubeflix/oliver
// 
// Network.h
// Neural network.

#pragma once

#include <exception>

namespace Oliver {
	class NetworkException : public std::exception {
	public:
		NetworkException(char* msg) : m_message(msg) {}
		char* what();
	private:
		char* m_message;
	};
}