#include "Neuron.h"
#include <random>
#include <iostream>

Neuron::Neuron(size_t inputsPerNeuron, size_t outputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate, bool initialiseRandomData)
    : 
    m_output(0.0),
    m_delta(0.0),
    m_learningRate(learningRate),
    m_activationFunc(MathUtility::getActivationFunc(activationFunc))
{
    m_inputsPerNeuron = inputsPerNeuron;

    if (initialiseRandomData)
    {
        double limit = std::sqrt(6.0 / (inputsPerNeuron + outputsPerNeuron)); // Xavier initialization
        m_weights = MathUtility::getRandomData(inputsPerNeuron, -limit, limit);
        m_bias = MathUtility::getRandomData(-limit, limit);
    }
    else
    {
        m_weights.assign(inputsPerNeuron, 0.0);
        m_bias = 0.0;
    }
}

void Neuron::print() const
{
    std::cout << "OUTPUT: " << m_output << std::endl;
}

void Neuron::activate(const std::vector<double>& inputs)
{
    double total = m_bias;

    if (inputs.size() == m_weights.size())
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            total += inputs[i] * m_weights[i];
        }
    }

    m_output = m_activationFunc(total);
}

double Neuron::activateDerivative(double x) const
{
    return x;
}

std::vector<double> Neuron::backwardsPropagate(double target)
{
    std::vector<double> res;
    return res;
}

void Neuron::calculateDelta(double target)
{
    m_delta = m_output - target;
}