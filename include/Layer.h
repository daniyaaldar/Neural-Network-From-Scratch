#pragma once

#include "MathUtility.h"
#include "Neuron.h"

class Layer 
{
public:
    Layer(size_t layerIdx, size_t numOfNeurons, size_t inputsPerNeuron, size_t outputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate, bool initialiseRandomData = false);
    ~Layer();

    void setOutputs(const std::vector<double>& outputs);
    void setOutput(size_t neuronIdx, double output);

    void setWeights(const std::vector<std::vector<double>>& weights);
    void setWeights(size_t neuronIdx, const std::vector<double>& weights);

    void setBiases(const std::vector<double>& biases);
    void setBias(size_t neuronIdx, double bias);

    std::vector<double> getOutputs() const;
    double getOutput(size_t neuronIdx) const;

    std::vector<std::vector<double>> getWeights() const;
    std::vector<double> getWeights(size_t neuronIdx) const;

    std::vector<double> getBiases() const;
    double getBias(size_t neuronIdx) const;

    void feedForward(const std::vector<double>& inputs);
    void print() const;

    size_t getNumOfNeurons() const { return m_neurons.size(); }
    size_t getNumInputsPerNeuron() const { return m_inputsPerNeuron; }
private:
    size_t m_layerIdx;
    size_t m_inputsPerNeuron;
    std::vector<std::unique_ptr<Neuron>> m_neurons;
};