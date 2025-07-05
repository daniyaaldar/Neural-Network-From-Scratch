#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector < Layer > layers) {
    m_layers = layers;
}

std::vector<double> NeuralNetwork::forwardPropagate(std::vector < double >& inputs) const
{
    std::vector<double> currentInputs;

    for (const Layer& layer : m_layers) {

        currentInputs = layer.forwardPropagate(currentInputs);
    }

    return currentInputs;
}