#include "Neuron.h"
#include "Layer.h"
#include <random>
#include <iostream>

Neuron::Neuron(size_t neuronIdx, size_t numOfInputs, size_t numOfOutputs, MathUtility::ActivationFunction activationFunc, double learningRate, bool initialiseRandomWeights)
    : 
    m_neuronIdx(neuronIdx),
    m_numOfInputs(numOfInputs),
    m_numOfOutputs(numOfOutputs),
    m_bias(0.0),
    m_output(0.0),
    m_delta(0.0),
    m_learningRate(learningRate),
    m_activationFunc(MathUtility::getActivationFunc(activationFunc)),
    m_activationDerivativeFunc(MathUtility::getActivationDerivativeFunc(activationFunc))
{
    if (initialiseRandomWeights)
    {
        double limit = std::sqrt(6.0 / (numOfInputs + numOfOutputs)); // Xavier initialization
        m_weights = MathUtility::getRandomData(numOfInputs, -limit, limit);
    }
    else
    {
        m_weights.assign(numOfInputs, 0.0);
    }
}

void Neuron::print() const
{
    std::cout << "OUTPUT: " << m_output << std::endl;
}

void Neuron::activate(const std::vector<double>& inputs)
{
    if (inputs.size() != m_weights.size())
    {
        throw std::invalid_argument("Input size does not match weight size");
    }

    const double* inPtr = inputs.data();
    const double* wPtr = m_weights.data();
    const size_t n = m_weights.size();

    double total = m_bias;
    for (size_t i = 0; i < n; ++i)
    {
        total += inPtr[i] * wPtr[i];
    }

    m_output = m_activationFunc(total);
}

void Neuron::calculateOutputGradient(double target)
{
    m_delta = (m_output - target) * m_activationDerivativeFunc(m_output);
}

double Neuron::sumDerivativesOfWeights(const Layer& nextLayer) const
{
    double sum = 0.0;
    for (size_t i = 0; i < nextLayer.getNumOfNeurons(); i++)
    {
        const Neuron& neuron = nextLayer.getNeuron(i);
        sum += neuron.getWeight(m_neuronIdx) * neuron.getDelta();
    }

    return sum;
}

void Neuron::calculateHiddenGradient(const Layer& nextLayer)
{
    double sumDow = sumDerivativesOfWeights(nextLayer);
    m_delta = sumDow * m_activationDerivativeFunc(m_output);
}

void Neuron::updateWeights(Layer& nextLayer)
{
    // Update the weights of neurons in the next layer that connect from this neurons output
    size_t numOfNeurons = nextLayer.getNumOfNeurons();
    for (size_t i = 0; i < numOfNeurons; i++)
    {
        Neuron& nextNeuron = nextLayer.getNeuron(i);

        // Access delta and weight vector directly to avoid repeated calls
        double delta = nextNeuron.getDelta();
        std::vector<double>& weightsRef = nextNeuron.getWeightsRef();

        weightsRef[m_neuronIdx] -= nextNeuron.getLearningRate() * delta * m_output;
    }
}

void Neuron::updateBias()
{
    m_bias -= m_learningRate * m_delta;
}