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
    void setWeights(size_t layerNum, const Matrix& weights);
    void setWeights(const std::vector<std::vector<std::vector<double>>>& weights);
    void setWeights(size_t layerNum, const std::vector<std::vector<double>>& weights);
    void setWeights(size_t layerNum, size_t neuronNum, const std::vector<double>& weights);

    void setBiases(const std::vector<Matrix>& biases);
    void setBiases(size_t layerNum, const Matrix& biases);
    void setBiases(const std::vector<std::vector<double>>& biases);
    void setBiases(size_t layerNum, const std::vector<double>& biases);
    void setBias(size_t layerNum, size_t neuronNum, double bias);

    std::vector<double> getOutput() const { return m_layers.back()->getOutputs(); }

private:
    std::vector<std::unique_ptr<Layer>>  m_layers;
};
