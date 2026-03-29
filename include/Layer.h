#pragma once

#include <memory>
#include <vector>
#include "MathUtility.h"
#include "Neuron.h"

class Layer 
{
public:
    Layer(size_t layerIdx, size_t numOfNeurons, size_t inputsPerNeuron, size_t outputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate, bool initialiseRandomWeights = true);

    void setOutputs(const std::vector<double>& outputs);
    void setOutput(size_t neuronIdx, double output);
    void setWeights(const std::vector<std::vector<double>>& weights);
    void setWeights(size_t neuronIdx, const std::vector<double>& weights);
    void setBiases(const std::vector<double>& biases);
    void setBias(size_t neuronIdx, double bias);

    const std::vector<double>& getOutputs() const;
    double getOutput(size_t neuronIdx) const;
    std::vector<std::vector<double>> getWeights() const;
    std::vector<double> getWeights(size_t neuronIdx) const;
    double getWeight(size_t neuronIdx, size_t weightIdx) const;
    Matrix getWeightsAsMatrix() const;
    std::vector<double> getBiases() const;
    double getBias(size_t neuronIdx) const;
    Neuron& getNeuron(size_t neuronIdx) { return m_neurons[neuronIdx]; }
    const Neuron& getNeuron(size_t neuronIdx) const { return m_neurons[neuronIdx]; }
    size_t getNumOfNeurons() const { return m_neurons.size(); }
    size_t getNumInputsPerNeuron() const { return m_inputsPerNeuron; }

    void feedForward(const std::vector<double>& inputs);
    void calculateOutputGradients(const std::vector<double>& targets);
    void calculateHiddenGradients(const Layer& nextLayer);
    void updateWeights(Layer& nextLayer);
    void updateBiases();
    void print() const;

private:
    size_t m_layerIdx;
    size_t m_inputsPerNeuron;
    std::vector<Neuron> m_neurons;
    mutable std::vector<double> m_outputs; // cached outputs to avoid reallocations
};