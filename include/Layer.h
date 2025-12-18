#pragma once

#include "MathUtility.h"
#include "Neuron.h"

class Layer 
{
public:
    Layer(size_t layerNum, size_t numOfNeurons, size_t numInputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate);
    ~Layer();
    void setOutputs(const std::vector<double>& outputs);
    std::vector<double> getOutputs();
    void feedForward(const std::vector<double>& inputs);
    void print() const;
    bool setWeights(const std::vector<std::vector<double>>& weights);

private:
    size_t m_layerNum;
    std::vector<Neuron*> m_neurons;
};