#include "NeuralNetwork.h"
#include <iostream>
#include <stdexcept>
#include <string>

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology, MathUtility::ActivationFunction activationFunc, double learningRate)
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
        throw std::invalid_argument("Target size does not match number of neurons in output layer (Has " + std::to_string(targets.size()) + ", expecting " + std::to_string(m_layers.front()->getNumOfNeurons()) + ")");
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

void NeuralNetwork::setWeights(size_t layerNum, const Matrix& weights)
{
    if (layerNum == 0 || layerNum >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index for weights");
    }

    size_t expectedRows = m_layers[layerNum]->getNumOfNeurons();
    size_t expectedCols = m_layers[layerNum]->getNumInputsPerNeuron();

    if (weights.GetNumRows() != expectedRows)
    {
        throw std::invalid_argument("Weight row count mismatch");
    }

    if (weights.GetNumCols() != expectedCols)
    {
        throw std::invalid_argument("Weight column count mismatch");
    }

    for (size_t neuronNum = 0; neuronNum < expectedRows; ++neuronNum)
    {
        m_layers[layerNum]->setWeights(neuronNum, weights.getRow(neuronNum));
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

void NeuralNetwork::setWeights(size_t layerNum, const std::vector<std::vector<double>>& weights)
{
    if (layerNum == 0 || layerNum >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index for weights");
    }

    if (weights.size() != m_layers[layerNum]->getNumOfNeurons())
    {
        throw std::invalid_argument("Weight neuron count mismatch");
    }

    for (size_t neuronNum = 0; neuronNum < weights.size(); ++neuronNum)
    {
        if (weights[neuronNum].size() != m_layers[layerNum]->getNumInputsPerNeuron())
        {
            throw std::invalid_argument("Weight input count mismatch");
        }

        setWeights(layerNum, neuronNum, weights[neuronNum]);
    }
}

void NeuralNetwork::setWeights(size_t layerNum, size_t neuronNum, const std::vector<double>& weights)
{
    if (layerNum == 0 || layerNum >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index");
    }

    if (neuronNum >= m_layers[layerNum]->getNumOfNeurons())
    {
        throw std::out_of_range("Invalid neuron index");
    }

    if (weights.size() != m_layers[layerNum]->getNumInputsPerNeuron())
    {
        throw std::invalid_argument("Weight size mismatch");
    }

    m_layers[layerNum]->setWeights(neuronNum, weights);
}

void NeuralNetwork::setBiases(const std::vector<Matrix>& biases)
{
    if (biases.size() != m_layers.size() - 1)
    {
        throw std::invalid_argument("Incorrect number of bias matrices");
    }

    for (size_t layerIdx = 0; layerIdx < biases.size(); ++layerIdx)
    {
        setBiases(layerIdx + 1, biases[layerIdx]);
    }
}

void NeuralNetwork::setBiases(size_t layerNum, const Matrix& biasMatrix)
{
    if (layerNum == 0 || layerNum >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index for biases");
    }

    size_t expectedRows = m_layers[layerNum]->getNumOfNeurons();

    if (biasMatrix.GetNumRows() != expectedRows)
    {
        throw std::invalid_argument("Bias row count mismatch");
    }

    if (biasMatrix.GetNumCols() != 1)
    {
        throw std::invalid_argument("Bias matrix must be column vector");
    }

    for (size_t neuronNum = 0; neuronNum < expectedRows; ++neuronNum)
    {
        m_layers[layerNum]->setBias(neuronNum, biasMatrix.getRow(neuronNum)[0]);
    }
}

void NeuralNetwork::setBiases(const std::vector<std::vector<double>>& biases)
{
    if (biases.size() != m_layers.size() - 1)
    {
        throw std::invalid_argument("Incorrect number of bias layers");
    }

    for (size_t layerNum = 0; layerNum < biases.size(); ++layerNum)
    {
        setBiases(layerNum + 1, biases[layerNum]);
    }
}

void NeuralNetwork::setBiases(size_t layerNum, const std::vector<double>& biases)
{
    if (layerNum == 0 || layerNum >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index for biases");
    }

    if (biases.size() != m_layers[layerNum]->getNumOfNeurons())
    {
        throw std::invalid_argument("Bias count mismatch");
    }

    for (size_t neuronNum = 0; neuronNum < biases.size(); ++neuronNum)
    {
        setBias(layerNum, neuronNum, biases[neuronNum]);
    }
}

void NeuralNetwork::setBias(size_t layerNum, size_t neuronNum, double bias)
{
    if (layerNum == 0 || layerNum >= m_layers.size())
    {
        throw std::out_of_range("Invalid layer index");
    }

    if (neuronNum >= m_layers[layerNum]->getNumOfNeurons())
    {
        throw std::out_of_range("Invalid neuron index");
    }

    m_layers[layerNum]->setBias(neuronNum, bias);
}
