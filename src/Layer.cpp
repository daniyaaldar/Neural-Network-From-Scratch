#include "Layer.h"
#include <iostream>
#include <string>

Layer::Layer(size_t layerIdx, size_t numOfNeurons, size_t inputsPerNeuron, size_t outputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate, bool initialiseRandomWeights)
    : 
    m_layerIdx(layerIdx),
    m_inputsPerNeuron(inputsPerNeuron),
    m_outputs()
{
    m_neurons.reserve(numOfNeurons);
    m_outputs.reserve(numOfNeurons);

    for (size_t i = 0; i < numOfNeurons; i++)
    {
        m_neurons.push_back(std::make_unique<Neuron>(i, inputsPerNeuron, outputsPerNeuron, activationFunc, learningRate, initialiseRandomWeights));
        m_outputs.push_back(0.0);
    }
}

void Layer::setOutputs(const std::vector<double>& outputs)
{
    if (m_neurons.size() != outputs.size())
    {
        throw std::out_of_range("Output count mismatch (received " + std::to_string(outputs.size()) + ", expected " + std::to_string(m_neurons.size()) + ")");
    }

    for (size_t neuronIdx = 0; neuronIdx < m_neurons.size(); neuronIdx++)
    {
        m_neurons[neuronIdx]->setOutput(outputs[neuronIdx]);
        m_outputs[neuronIdx] = outputs[neuronIdx];
    }
}

void Layer::setOutput(size_t neuronIdx, double output)
{
    if (neuronIdx >= m_neurons.size())
    {
        throw std::out_of_range("Neuron index out of range (received " + std::to_string(neuronIdx) + ", valid range 0 to " + std::to_string(m_neurons.size() - 1) + ")");
    }

    m_neurons[neuronIdx]->setOutput(output);
    m_outputs[neuronIdx] = output;
}

void Layer::setWeights(const std::vector<std::vector<double>>& weights)
{
    if (m_neurons.size() != weights.size())
    {
        throw std::out_of_range("Neuron weight row count mismatch (received " + std::to_string(weights.size()) + ", expected " + std::to_string(m_neurons.size()) + ")");
    }

    for (size_t neuronIdx = 0; neuronIdx < weights.size(); neuronIdx++)
    {
        setWeights(neuronIdx, weights[neuronIdx]);
    }
}

void Layer::setWeights(size_t neuronIdx, const std::vector<double>& weights)
{
    if (m_neurons[neuronIdx]->getNumOfInputs() != weights.size())
    {
        throw std::out_of_range("Weight count mismatch for neuron " + std::to_string(neuronIdx) + " (received " + std::to_string(weights.size()) + ", expected " + std::to_string(m_neurons[neuronIdx]->getNumOfInputs()) + ")");
    }

    m_neurons[neuronIdx]->setWeights(weights);
}

void Layer::setBiases(const std::vector<double>& biases)
{
    if (m_neurons.size() != biases.size())
    {
        throw std::out_of_range("Bias count mismatch (received " + std::to_string(biases.size()) + ", expected " + std::to_string(m_neurons.size()) + ")");
    }

    for (size_t neuronIdx = 0; neuronIdx < biases.size(); neuronIdx++)
    {
        setBias(neuronIdx, biases[neuronIdx]);
    }
}

void Layer::setBias(size_t neuronIdx, double bias)
{
    m_neurons[neuronIdx]->setBias(bias);
}

const std::vector<double>& Layer::getOutputs() const
{
    for (size_t idx = 0; idx < m_neurons.size(); ++idx)
    {
        m_outputs[idx] = m_neurons[idx]->getOutput();
    }

    return m_outputs;
}

double Layer::getOutput(size_t neuronIdx) const
{
    if (neuronIdx >= m_neurons.size())
    {
        throw std::out_of_range("Neuron index out of range (received " + std::to_string(neuronIdx) + ", valid range 0 to " + std::to_string(m_neurons.size() - 1) + ")");
    }

    return m_neurons[neuronIdx]->getOutput();
}

std::vector<std::vector<double>> Layer::getWeights() const
{
    std::vector<std::vector<double>> weights;

    for (const std::unique_ptr<Neuron>& n : m_neurons)
    {
        weights.push_back(n->getWeights());
    }

    return weights;
}

std::vector<double> Layer::getWeights(size_t neuronIdx) const
{
    if (neuronIdx >= m_neurons.size())
    {
        throw std::out_of_range("Neuron index out of range (received " + std::to_string(neuronIdx) + ", valid range 0 to " + std::to_string(m_neurons.size() - 1) + ")");
    }

    return m_neurons[neuronIdx]->getWeights();
}

double Layer::getWeight(size_t neuronIdx, size_t weightIdx) const
{
    if (neuronIdx >= m_neurons.size())
    {
        throw std::out_of_range("Neuron index out of range (received " + std::to_string(neuronIdx) + ", valid range 0 to " + std::to_string(m_neurons.size() - 1) + ")");
    }

    if (weightIdx >= m_neurons[neuronIdx]->getNumOfInputs())
    {
        throw std::out_of_range("Neuron weight index out of range (received " + std::to_string(weightIdx) + ", valid range 0 to " + std::to_string(m_neurons[neuronIdx]->getNumOfInputs() - 1) + ")");
    }

    return m_neurons[neuronIdx]->getWeight(weightIdx);
}

Matrix Layer::getWeightsAsMatrix() const
{
    size_t numNeurons = m_neurons.size();
    size_t numInputs = m_inputsPerNeuron;

    Matrix matrix(numNeurons, numInputs);

    for (size_t i = 0; i < numNeurons; i++)
    {
        for (size_t j = 0; j < numInputs; j++)
        {
            matrix.SetValue(i, j, m_neurons[i]->getWeight(j));
        }
    }

    return matrix;
}

std::vector<double> Layer::getBiases() const
{
    std::vector<double> biases;

    for (const std::unique_ptr<Neuron>& n : m_neurons)
    {
        biases.push_back(n->getBias());
    }

    return biases;
}

double Layer::getBias(size_t neuronIdx) const
{
    if (neuronIdx >= m_neurons.size())
    {
        throw std::out_of_range("Neuron index out of range (received " + std::to_string(neuronIdx) + ", valid range 0 to " + std::to_string(m_neurons.size() - 1) + ")");
    }

    return m_neurons[neuronIdx]->getBias();
}

void Layer::feedForward(const std::vector<double>& inputs)
{
    if (inputs.size() != m_neurons[0]->getNumOfInputs())
    {
        throw std::out_of_range("Input count mismatch (received " + std::to_string(inputs.size()) + ", expected " + std::to_string(m_neurons[0]->getNumOfInputs()) + ")");
    }

    for (size_t idx = 0; idx < m_neurons.size(); ++idx)
    {
        m_neurons[idx]->activate(inputs);
    }
}

void Layer::calculateOutputGradients(const std::vector<double>& targets)
{
    size_t numOfNeurons = getNumOfNeurons();
    for (size_t i = 0; i < numOfNeurons; i++)
    {
        getNeuron(i).calculateOutputGradient(targets[i]);
    }
}

void Layer::calculateHiddenGradients(const Layer& nextLayer)
{
    size_t numOfNeurons = getNumOfNeurons();
    for (size_t i = 0; i < numOfNeurons; i++)
    {
        getNeuron(i).calculateHiddenGradient(nextLayer);
    }
}

void Layer::updateWeights(Layer& nextLayer)
{
    size_t numOfNeurons = getNumOfNeurons();
    for (size_t i = 0; i < numOfNeurons; i++)
    {
        getNeuron(i).updateWeights(nextLayer);
    }
}

void Layer::updateBiases()
{
    size_t numOfNeurons = getNumOfNeurons();
    for (size_t i = 0; i < numOfNeurons; i++)
    {
        getNeuron(i).updateBias();
    }
}

void Layer::print() const
{
    size_t count = 1;
    for (const std::unique_ptr<Neuron>& neuron : m_neurons)
    {
        std::cout << "NEURON " << count++ << ": ";
        neuron->print();
    }
}