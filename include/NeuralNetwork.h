#pragma once

#include "MathUtility.h"
#include "Layer.h"

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<size_t>& topology, MathUtility::ActivationFunction activationFunc, double learningRate);
    ~NeuralNetwork();

    void print() const;
    void feedForward(const std::vector<double>& inputs);
    void backwardsPropagate(const std::vector<double>& targets);

    void setWeights(const std::vector<Matrix>& weights);
    void setWeights(size_t layerIdx, const Matrix& weights);
    void setWeights(const std::vector<std::vector<std::vector<double>>>& weights);
    void setWeights(size_t layerIdx, const std::vector<std::vector<double>>& weights);
    void setWeights(size_t layerIdx, size_t neuronIdx, const std::vector<double>& weights);

    void setBiases(const std::vector<Matrix>& biases);
    void setBiases(size_t layerIdx, const Matrix& biases);
    void setBiases(const std::vector<std::vector<double>>& biases);
    void setBiases(size_t layerIdx, const std::vector<double>& biases);
    void setBias(size_t layerIdx, size_t neuronIdx, double bias);

    std::vector<double> getOutput() const { return m_layers.back()->getOutputs(); }

private:
    std::vector<std::unique_ptr<Layer>>  m_layers;
};
