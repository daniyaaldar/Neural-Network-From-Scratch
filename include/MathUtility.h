#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <random>

#include "Matrix.h"

class MathUtility 
{
public:
	enum class ActivationFunction
	{
		RELU,
		SIGMOID
	};

	static double getRandomData(double min = -1, double max = 1);
	static std::vector<double> getRandomData(size_t size, double min = -1, double max = 1);
	static double sigmoid(double x);
	static double ReLu(double x);
	static double sigmoidDerivative(double x);
	static std::vector<double> softmax(const std::vector<double>& values);
	static double dot(const std::vector<double>& a, const std::vector<double>& b);
	static double cost(const std::vector<double>& outputs, const std::vector<double>& target);
	static std::function<double(double)> getActivationFunc(ActivationFunction activationFunc);

private:

};