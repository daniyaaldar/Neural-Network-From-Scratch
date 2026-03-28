#include "NeuralNetwork.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology,
    MathUtility::ActivationFunction activationFunc,
    MathUtility::ErrorFunction errorFunc,
    double learningRate,
    bool initialiseRandomWeights)
    : 
    NeuralNetwork(topology, activationFunc, MathUtility::ActivationFunction::LINEAR, errorFunc, learningRate, initialiseRandomWeights)
{
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& topology,
    MathUtility::ActivationFunction activationFunc,
    MathUtility::ActivationFunction outputActivationFunc,
    MathUtility::ErrorFunction errorFunc,
    double learningRate,
    bool initialiseRandomWeights)
{
    const size_t numLayers = topology.size();

    if (numLayers < 2)
    {
        throw std::invalid_argument("Network must have at least input and output layer");
    }

    if (learningRate <= 0.0)
    {
        throw std::invalid_argument("Learning rate must be positive");
    }

    for (size_t i = 0; i < numLayers; ++i)
    {
        if (topology[i] == 0)
        {
            throw std::invalid_argument("Layer " + std::to_string(i) + " has zero neurons");
        }
    }

    m_error = 0.0;
    m_activationFunc = activationFunc;
    m_outputActivationFunc = outputActivationFunc;
    m_errorFunc = MathUtility::getErrorFunc(errorFunc);
    m_learningRate = learningRate;

    m_layers.reserve(numLayers);
    for (size_t i = 0; i < numLayers; ++i)
    {
        const bool isInputLayer = (i == 0);
        const bool isOutputLayer = (i == numLayers - 1);

        const size_t inputsPerNeuron = isInputLayer ? 0 : topology[i - 1];
        const size_t outputsPerNeuron = isOutputLayer ? 0 : topology[i + 1];

        const MathUtility::ActivationFunction layerActivationFunc = isOutputLayer ? m_outputActivationFunc : m_activationFunc;

        m_layers.push_back(std::make_unique<Layer>(i, topology[i], inputsPerNeuron, outputsPerNeuron, layerActivationFunc, m_learningRate, initialiseRandomWeights));
    }
}

double NeuralNetwork::getError(const std::vector<double>& targets) const
{
    return m_errorFunc(m_layers.back()->getOutputs(), targets);
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

std::vector<std::vector<std::vector<double>>> NeuralNetwork::getWeights() const
{
    std::vector<std::vector<std::vector<double>>> weights;
    weights.reserve(m_layers.size());

    for (const std::unique_ptr<Layer>& layer : m_layers)
    {
        weights.push_back(layer->getWeights());
    }
    
    return weights;
}

std::vector<std::vector<double>> NeuralNetwork::getWeights(size_t layerIdx) const
{
    return m_layers[layerIdx]->getWeights();
}

std::vector<double> NeuralNetwork::getWeights(size_t layerIdx, size_t neuronIdx) const
{
    return m_layers[layerIdx]->getNeuron(neuronIdx).getWeights();
}

double NeuralNetwork::getWeight(size_t layerIdx, size_t neuronIdx, size_t weightIdx) const
{
    return m_layers[layerIdx]->getNeuron(neuronIdx).getWeight(weightIdx);
}

std::vector<Matrix> NeuralNetwork::getWeightsAsMatrices() const
{
    std::vector<Matrix> weightMatricies;

    for (const std::unique_ptr<Layer>& layer : m_layers)
    {
        weightMatricies.push_back(layer->getWeightsAsMatrix());
    }

    return weightMatricies;
}

Matrix NeuralNetwork::getLayerWeightsAsMatrix(size_t layerIdx) const
{
    return m_layers[layerIdx]->getWeightsAsMatrix();
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

void NeuralNetwork::learn(size_t epochs, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, size_t logInterval)
{
    if (inputs.size() != targets.size())
    {
        throw std::invalid_argument("Input and target count mismatch");
    }

    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng{ std::random_device{}() };

    for (size_t iteration = 0; iteration < epochs; iteration++)
    {
        std::shuffle(indices.begin(), indices.end(), rng);

        int correct = 0;
        double epochError = 0.0;
        auto epochStart = std::chrono::steady_clock::now();

        for (size_t sampleIdx = 0; sampleIdx < inputs.size(); sampleIdx++)
        {
            size_t idx = indices[sampleIdx];
            feedForward(inputs[idx]);
            backwardsPropagate(targets[idx]);
            epochError += m_error;

            const std::vector<double>& outputs = getOutput();
            int predicted = std::max_element(outputs.begin(), outputs.end()) - outputs.begin();
            int expected = std::max_element(targets[idx].begin(), targets[idx].end()) - targets[idx].begin();
            if (predicted == expected)
                ++correct;
        }

        auto epochEnd = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(epochEnd - epochStart).count();

        if (logInterval > 0 && iteration % logInterval == 0)
        {
            double accuracy = 100.0 * correct / inputs.size();
            std::cout 
                << "Epoch " << iteration
                << " | Error: " << epochError / inputs.size()
                << " | Accuracy: " << accuracy << "%"
                << " | Correct: " << correct
                << " | Wrong: " <<  inputs.size() - correct
                << " | Time Taken: " <<  elapsed
                << "\n";
        }
    }
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputs)
{
    if (inputs.size() != m_layers.front()->getNumOfNeurons())
    {
        throw std::invalid_argument("Input size does not match number of neurons in input layer (Has " + std::to_string(inputs.size()) + ", expecting " + std::to_string(m_layers.front()->getNumOfNeurons()) + ")");
    }

    m_layers[0]->setOutputs(inputs);

    size_t numLayers = m_layers.size();
    for (size_t i = 1; i < numLayers; ++i)
    {
        const std::vector<double>& lastLayerOutput = m_layers[i - 1]->getOutputs();
        m_layers[i]->feedForward(lastLayerOutput);
    }

    std::vector<double> outputs = getOutput();

    return outputs;
}

void NeuralNetwork::backwardsPropagate(const std::vector<double>& targets)
{
    const std::unique_ptr<Layer>& outputLayer = m_layers.back();

    if (targets.size() != outputLayer->getNumOfNeurons())
    {
        throw std::invalid_argument("Target size does not match number of neurons in output layer (Has " +
            std::to_string(targets.size()) + ", expecting " +
            std::to_string(m_layers.back()->getNumOfNeurons()) + ")");
    }

    // Calculate output layer error
    m_error = m_errorFunc(outputLayer->getOutputs(), targets);

    // Calculate output layer gradients
    outputLayer->calculateOutputGradients(targets);

    // Calculate gradients on hidden layers
    for (size_t i = m_layers.size() - 2; i > 0; i--)
    {
        std::unique_ptr<Layer>& currentLayer = m_layers[i];
        std::unique_ptr<Layer>& nextLayer = m_layers[i + 1];

        currentLayer->calculateHiddenGradients(*nextLayer);
    }

    // Update weights
    for (size_t i = 0; i < m_layers.size() - 1; i++)
    {
        m_layers[i]->updateWeights(*m_layers[i + 1]);
    }

    // Update biases (skip input layer)
    for (size_t i = 1; i < m_layers.size(); i++) 
    {
        m_layers[i]->updateBiases();
    }
}
