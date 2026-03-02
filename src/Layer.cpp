#include "Layer.h"
#include <iostream>
#include <cassert>
#include <string>

Layer::Layer(size_t layerIdx, size_t numOfNeurons, size_t inputsPerNeuron, size_t outputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate, bool initialiseRandomWeights)
{
    m_layerIdx = layerIdx;
    m_inputsPerNeuron = inputsPerNeuron;

    for (size_t i = 0; i < numOfNeurons; i++)
    {
        m_neurons.push_back(std::make_unique<Neuron>(inputsPerNeuron, outputsPerNeuron, activationFunc, learningRate, initialiseRandomWeights));
    }
}

Layer::~Layer()
{
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
    }
}

void Layer::setOutput(size_t neuronIdx, double output)
{
    if (neuronIdx >= m_neurons.size())
    {
        throw std::out_of_range("Neuron index out of range (received " + std::to_string(neuronIdx) + ", valid range 0 to " + std::to_string(m_neurons.size() - 1) + ")");
    }

    m_neurons[neuronIdx]->setOutput(output);
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

std::vector<double> Layer::getOutputs() const
{
    std::vector<double> outputs;

    for (const std::unique_ptr<Neuron>& n : m_neurons)
    {
        outputs.push_back(n->getOutput());
    }

    return outputs;
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

    for (std::unique_ptr<Neuron>& n : m_neurons)
    {
        n->activate(inputs);
    }
}

void Layer::print() const
{
    int count = 1;
    for (const std::unique_ptr<Neuron>& neuron : m_neurons)
    {
        std::cout << "NEURON " << count++ << ": ";
        neuron->print();
    }
}