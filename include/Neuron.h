#pragma once

#include <vector>
#include "MathUtility.h"

class Neuron {

public:
	Neuron(size_t numOfOutputs);

	void setValue(double value);

	//std::vector < double > getWeights() const { return m_weights; };
	double getBias() const { return m_bias; };

	double activate(const std::vector<double>& inputs) const;
	double activateDerivative(double x) const;

private:
	std::vector < double > m_weights;
	double m_bias;
	double m_output;
	double m_delta;
};