#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <random>

class MathUtility 
{
public:
	static std::vector < double > getRandomData(size_t size, double min = -1, double max = 1);
	static double sigmoid(double x);
	static double sigmoidDerivative(double x);
	static double dot(const std::vector<double>& a, const std::vector<double>& b);
	static double meanSquaredError(const std::vector<double>& output, const std::vector<double>& target);
	static std::vector<double> softmax(const std::vector<double>& values);

private:

};