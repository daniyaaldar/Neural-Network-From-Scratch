#pragma once

#include "MathUtility.h"
#include "Layer.h"

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<size_t>& numOfNeuronsPerLayer, MathUtility::ActivationFunction activationFunc, double learningRate);
    ~NeuralNetwork();

    void print() const;
    void feedForward(const std::vector<double>& inputs);
    void backwardsPropagate(const std::vector<double>& targets);
    bool setWeights(size_t layerNum, const std::vector<std::vector<double>>& weights);
    bool setWeights(size_t layerNum, size_t neuronNum, const std::vector<double>& weights);
    std::vector<double> getOutput() const { return m_layers.back()->getOutputs(); }

private:
    std::vector<Layer*>  m_layers;
};
