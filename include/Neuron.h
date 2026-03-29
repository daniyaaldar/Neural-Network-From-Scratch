#pragma once

#include <vector>
#include "MathUtility.h"

class Layer;

class Neuron 
{
public:
    Neuron(size_t neuronIdx, size_t numOfInputs, size_t numOfOutputs, MathUtility::ActivationFunction activationFunc, double learningRate, bool initialiseRandomWeights = true);

    void print() const;
    void setOutput(double output) { m_output = output; }
    void setBias(double bias) { m_bias = bias; }
    void setDelta(double delta) { m_delta = delta; }
    void setWeights(const std::vector<double>& weights) { m_weights = weights; }
    void setWeight(size_t weightIdx, double weight) { m_weights[weightIdx] = weight; } // TODO add error checking
    
    double getOutput() const { return m_output; }
    double getBias() const { return m_bias; }
    double getDelta() const { return m_delta; }
    size_t getNumOfInputs() const { return m_weights.size(); }
    std::vector<double> getWeights() const { return m_weights; }
    const std::vector<double>& getWeightsRef() const { return m_weights; }
    std::vector<double>& getWeightsRef() { return m_weights; }
    double getWeight(size_t idx) const { return m_weights[idx]; }
    double getLearningRate() const { return m_learningRate; }

    void activate(const std::vector<double>& inputs);
    void calculateOutputGradient(double target);
    void calculateHiddenGradient(const Layer& nextLayer);
    void updateWeights(Layer& nextLayer);
    void updateBias();

private:    
    double sumDerivativesOfWeights(const Layer& nextLayer) const;

    size_t m_neuronIdx = 0;
    size_t m_numOfInputs = 0;
    size_t m_numOfOutputs = 0;
    std::vector<double> m_weights; // input weights
    double m_bias;
    double m_output;
    double m_delta;
    double m_learningRate;
    std::function<double(double)> m_activationFunc;
    std::function<double(double)> m_activationDerivativeFunc;
};