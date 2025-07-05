#include "Neuron.h"
#include <random>

Neuron::Neuron(size_t numOfOutputs) 
{
    m_weights = MathUtility::getRandomData(numOfOutputs, -1.0, 1.0);
    m_bias = MathUtility::getRandomData(1, -1.0, 1.0)[0];
    m_output = 0.0;
    m_delta = 0.0;
}

void Neuron::setValue(double value) 
{
    m_output = value;
}

double Neuron::activate(const std::vector<double>& inputs) const
{
    double total = m_bias;

    if (inputs.size() == m_weights.size())
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            total += inputs[i] * m_weights[i];
        }
    }

    return MathUtility::sigmoid(total);
}

double Neuron::activateDerivative(double x) const
{
    return x;
}
