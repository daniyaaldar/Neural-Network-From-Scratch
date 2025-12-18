#pragma once

#include <vector>
#include "MathUtility.h"

class Neuron 
{
public:
    Neuron(size_t inputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate);

    void print() const;
    void setOutput(double output) { m_output = output; };
    void setBias(double bias) { m_bias = bias; };
    void setDelta(double delta) { m_delta = delta; };
    void setNumOfInputs(size_t inputsPerNeuron) { m_weights = MathUtility::getRandomData(inputsPerNeuron, -1.0, 1.0); }
    void setWeights(const std::vector<double>& weights) { m_weights = weights; };
    
    double getOutput() const { return m_output; };
    double getBias() const { return m_bias; };
    double getDelta() const { return m_delta; };
    size_t getNumOfInputs() const { return m_weights.size(); };
    std::vector<double> getWeights() const { return m_weights; };

    void activate(const std::vector<double>& inputs);
    double activateDerivative(double x) const;
    std::vector<double> backwardsPropagate(double target);
    void calculateDelta(double target);

private:
    std::vector<double> m_weights; // input weights
    double m_bias;
    double m_output;
    double m_delta;
    double m_learningRate;
    std::function<double(double)> m_activationFunc;
    std::function<double(double)> m_activationDerivativeFunc;
};