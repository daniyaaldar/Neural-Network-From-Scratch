#pragma once

#include <vector>
#include <memory>
#include "MathUtility.h"
#include "Layer.h"
#include "Matrix.h"

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<size_t>& topology,
        MathUtility::ActivationFunction activationFunc,
        MathUtility::ErrorFunction errorFunc,
        double learningRate,
        bool initialiseRandomWeights = true);

    NeuralNetwork(const std::vector<size_t>& topology,
        MathUtility::ActivationFunction activationFunc,
        MathUtility::ActivationFunction outputActivationFunc,
        MathUtility::ErrorFunction errorFunc,
        double learningRate,
        bool initialiseRandomWeights = true);

    void                                          setWeights(const std::vector<Matrix>& weights);
    void                                          setWeights(const std::vector<std::vector<std::vector<double>>>& weights);
    void                                          setWeights(size_t layerIdx, const Matrix& weights);
    void                                          setWeights(size_t layerIdx, const std::vector<std::vector<double>>& weights);
    void                                          setWeights(size_t layerIdx, size_t neuronIdx, const std::vector<double>& weights);
    void                                          setBiases(const std::vector<std::vector<double>>& biases);
    void                                          setBiases(size_t layerIdx, const std::vector<double>& biases);
    void                                          setBias(size_t layerIdx, size_t neuronIdx, double bias);

    std::vector<std::vector<std::vector<double>>> getWeights() const;
    std::vector<std::vector<double>>              getWeights(size_t layerIdx) const;
    std::vector<double>                           getWeights(size_t layerIdx, size_t neuronIdx) const;
    double                                        getWeight(size_t layerIdx, size_t neuronIdx, size_t weightIdx) const;
    std::vector<Matrix>                           getWeightsAsMatrices() const;
    Matrix                                        getLayerWeightsAsMatrix(size_t layerIdx) const;
    std::vector<std::vector<double>>              getBiases() const;
    std::vector<double>                           getBiases(size_t layerIdx) const;
    double                                        getBias(size_t layerIdx, size_t neuronIdx) const;
    const std::vector<double>&                    getOutput() const { return m_layers.back()->getOutputs(); }
    double                                        getError(const std::vector<double>& targets) const;
    double                                        getDelta(size_t layerIdx, size_t neuronIdx) const { return m_layers[layerIdx]->getNeuron(neuronIdx).getDelta(); } // TODO add error check

    void                                          print() const;
    void                                          learn(size_t epochs, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, size_t logInterval = 0);
    const std::vector<double>&                    feedForward(const std::vector<double>& inputs);
    void                                          backwardsPropagate(const std::vector<double>& targets);

private:
    std::vector<std::unique_ptr<Layer>>  m_layers;
    MathUtility::ActivationFunction m_activationFunc;
    MathUtility::ActivationFunction m_outputActivationFunc;
    std::function<double(const std::vector<double>&, const std::vector<double>&)> m_errorFunc;
    double m_error;
    double m_learningRate;
};
