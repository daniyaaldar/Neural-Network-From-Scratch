#include "NeuralNetwork.h"
#include <iostream>
#include <stdexcept>
#include <string>

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, MathUtility::ActivationFunction activationFunc, double learningRate)
    : 
    m_activationFunc(activationFunc),
    m_learningRate(learningRate)
{
    if (learningRate <= 0.0)
    {
        throw std::invalid_argument("Learning rate must be positive");
    }

    if (topology.size() < 2)
    {
        throw std::invalid_argument("Network must have at least input and output layer");
    }

    for (size_t i = 0; i < topology.size(); ++i)
    {
        if (topology[i] == 0)
        {
            throw std::invalid_argument("Layer " + std::to_string(i) + " has zero neurons");
        }
    }

    m_layers.reserve(topology.size());

    for (size_t i = 0; i < topology.size(); ++i)
    {
        size_t inputsPerNeuron = (i == 0) ? 0 : topology[i - 1];
        size_t outputsPerNeuron = (i == topology.size() - 1) ? 0 : topology[i + 1];

        m_layers.push_back(std::make_unique<Layer>(i, topology[i], inputsPerNeuron, outputsPerNeuron, activationFunc, learningRate));
    }
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::print() const
{
    for (size_t i = 0; i < m_layers.size(); ++i)
    {
        std::cout << "LAYER " << i << std::endl;
        m_layers[i]->print();
        std::cout << "\n";
    }
}

void NeuralNetwork::feedForward(const std::vector<double>& inputs)
{
    if (inputs.size() != m_layers.front()->getNumOfNeurons())
    {
        throw std::invalid_argument("Input size does not match number of neurons in input layer (Has " + std::to_string(inputs.size()) + ", expecting " + std::to_string(m_layers.front()->getNumOfNeurons()) + ")");
    }

    m_layers[0]->setOutputs(inputs);
    
    for (size_t i = 1; i < m_layers.size(); ++i)
    {
        m_layers[i]->feedForward(m_layers[i - 1]->getOutputs());
    }
}

void NeuralNetwork::backwardsPropagate(const std::vector<double>& targets)
{
    if (targets.size() != m_layers.back()->getNumOfNeurons())
    {
        throw std::invalid_argument("Target size does not match number of neurons in output layer (Has " + std::to_string(targets.size()) + ", expecting " + std::to_string(m_layers.back()->getNumOfNeurons()) + ")");
    }

    // TODO
}

void NeuralNetwork::setWeights(const std::vector<Matrix>& weights)
{
    if (weights.size() != m_layers.size() - 1)
    {
        throw std::invalid_argument("Incorrect number of weight matrices");
    }

    for (size_t layerIdx = 0; layerIdx < weights.size(); ++layerIdx)
    {
        setWeights(layerIdx + 1, weights[layerIdx]);
    }
}

void NeuralNetwork::setWeights(const std::vector<std::vector<std::vector<double>>>& weights)
{
    if (weights.size() != m_layers.size() - 1)
    {
        throw std::invalid_argument("Number of weights given does not match number of layers with weights (Has " + std::to_string(weights.size()) + ", expecting " + std::to_string(m_layers.size() - 1) + ")");
    }

    for (size_t layerIdx = 0; layerIdx < weights.size(); ++layerIdx)
    {
        setWeights(layerIdx + 1, weights[layerIdx]);
    }
}

void NeuralNetwork::setWeights(size_t layerIdx, const Matrix& weights)
{
    if (layerIdx == 0 || layerIdx >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index for weights");
    }

    size_t expectedRows = m_layers[layerIdx]->getNumOfNeurons();
    size_t expectedCols = m_layers[layerIdx]->getNumInputsPerNeuron();

    if (weights.GetRows() != expectedRows)
    {
        throw std::invalid_argument("Weight row count mismatch");
    }

    if (weights.GetCols() != expectedCols)
    {
        throw std::invalid_argument("Weight column count mismatch");
    }

    for (size_t neuronIdx = 0; neuronIdx < expectedRows; ++neuronIdx)
    {
        m_layers[layerIdx]->setWeights(neuronIdx, weights.getRow(neuronIdx));
    }
}

void NeuralNetwork::setWeights(size_t layerIdx, const std::vector<std::vector<double>>& weights)
{
    if (layerIdx == 0 || layerIdx >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index for weights");
    }

    if (weights.size() != m_layers[layerIdx]->getNumOfNeurons())
    {
        throw std::invalid_argument("Weight neuron count mismatch");
    }

    for (size_t neuronIdx = 0; neuronIdx < weights.size(); ++neuronIdx)
    {
        if (weights[neuronIdx].size() != m_layers[layerIdx]->getNumInputsPerNeuron())
        {
            throw std::invalid_argument("Weight input count mismatch");
        }

        setWeights(layerIdx, neuronIdx, weights[neuronIdx]);
    }
}

void NeuralNetwork::setWeights(size_t layerIdx, size_t neuronIdx, const std::vector<double>& weights)
{
    if (layerIdx == 0 || layerIdx >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index");
    }

    if (neuronIdx >= m_layers[layerIdx]->getNumOfNeurons())
    {
        throw std::out_of_range("Invalid neuron index");
    }

    if (weights.size() != m_layers[layerIdx]->getNumInputsPerNeuron())
    {
        throw std::invalid_argument("Weight size mismatch");
    }

    m_layers[layerIdx]->setWeights(neuronIdx, weights);
}

void NeuralNetwork::setBiases(const std::vector<std::vector<double>>& biases)
{
    if (biases.size() != m_layers.size() - 1)
    {
        throw std::invalid_argument("Incorrect number of bias layers");
    }

    for (size_t layerIdx = 0; layerIdx < biases.size(); ++layerIdx)
    {
        setBiases(layerIdx + 1, biases[layerIdx]);
    }
}

void NeuralNetwork::setBiases(size_t layerIdx, const std::vector<double>& biases)
{
    if (layerIdx == 0 || layerIdx >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index for biases");
    }

    if (biases.size() != m_layers[layerIdx]->getNumOfNeurons())
    {
        throw std::invalid_argument("Bias count mismatch");
    }

    for (size_t neuronIdx = 0; neuronIdx < biases.size(); ++neuronIdx)
    {
        setBias(layerIdx, neuronIdx, biases[neuronIdx]);
    }
}

void NeuralNetwork::setBias(size_t layerIdx, size_t neuronIdx, double bias)
{
    if (layerIdx == 0 || layerIdx >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index");
    }

    if (neuronIdx >= m_layers[layerIdx]->getNumOfNeurons())
    {
        throw std::out_of_range("Invalid neuron index");
    }

    m_layers[layerIdx]->setBias(neuronIdx, bias);
}
