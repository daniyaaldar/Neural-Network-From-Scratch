#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& numOfNeuronsPerLayer, MathUtility::ActivationFunction activationFunc, double learningRate)
{
    for (size_t i = 0; i < numOfNeuronsPerLayer.size(); i++)
    {
        if (i == 0)
        {
            m_layers.push_back(new Layer(i, numOfNeuronsPerLayer[i], 0, activationFunc, learningRate));
        }
        else
        {
            m_layers.push_back(new Layer(i, numOfNeuronsPerLayer[i], numOfNeuronsPerLayer[i - 1], activationFunc, learningRate));
        }
    }
}

NeuralNetwork::~NeuralNetwork()
{
    for (Layer* l : m_layers)
        delete l;
}

void NeuralNetwork::print() const
{
    int count = 1;
    for (const Layer* layer : m_layers)
    {
        std::cout << "LAYER " << count++ << std::endl;
        layer->print();
        std::cout << "\n\n";
    }
}

void NeuralNetwork::feedForward(const std::vector<double>& inputs)
{
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        if (i == 0)
        {
            m_layers[i]->setOutputs(inputs);
        }
        else
        {
            m_layers[i]->feedForward(m_layers[i - 1]->getOutputs());
        }
    }
}

void NeuralNetwork::backwardsPropagate(const std::vector < double >& targets)
{
}

bool NeuralNetwork::setWeights(size_t layerNum, const std::vector<std::vector<double>>& weights)
{
    if (layerNum < m_layers.size())
    {
        return m_layers[layerNum]->setWeights(weights);
    }

    return false;
}
