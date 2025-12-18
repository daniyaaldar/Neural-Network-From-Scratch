#include "Layer.h"
#include <iostream>
#include <cassert>

Layer::Layer(size_t layerNum, size_t numOfNeurons, size_t numInputsPerNeuron, MathUtility::ActivationFunction activationFunc, double learningRate)
{
    m_layerNum = layerNum;

    for (size_t i = 0; i < numOfNeurons; i++)
    {
        m_neurons.push_back(new Neuron(numInputsPerNeuron, activationFunc, learningRate));
    }
}

Layer::~Layer()
{
    for (Neuron* n : m_neurons)
        delete n;
}
void Layer::setOutputs(const std::vector<double>& outputs)
{
    if (m_neurons.size() != outputs.size())
        throw std::runtime_error("Number of outputs and number of neurons in layer do not match");

    for (size_t i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i]->setOutput(outputs[i]);
    }
}


std::vector<double> Layer::getOutputs()
{
    std::vector<double> outputs;

    for (const Neuron* n : m_neurons)
    {
        outputs.push_back(n->getOutput());
    }

    return outputs;
}

void Layer::feedForward(const std::vector<double>& inputs)
{
    for (Neuron* n : m_neurons)
    {
        n->activate(inputs);
    }
}

void Layer::print() const
{
    //int count = 1;
    //for (const Neuron& neuron : m_neurons)
    //{
    //    std::cout << "NEURON " << count++ << ": ";
    //    neuron.print();
    //}
}

bool Layer::setWeights(const std::vector<std::vector<double>>& weights)
{
    if (m_neurons.size() != weights.size())
        return false;

    for (size_t i = 0; i < weights.size(); i++)
    {
        if (weights[i].size() != m_neurons[i]->getNumOfInputs())
            return false;

        m_neurons[i]->setWeights(weights[i]);
    }

    return true;
}
