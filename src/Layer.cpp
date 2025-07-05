#include "Layer.h"

Layer::Layer(size_t numOfNeurons, size_t numOfInputsPerNeuron) 
{
    for (size_t idx = 0; idx < numOfNeurons; idx++) {
        Neuron neuron(numOfInputsPerNeuron);
        m_neurons.push_back(neuron);
    }
}

std::vector < double > Layer::forwardPropagate(const std::vector<double>& inputs) const
{
    std::vector < double > outputs;

    for (const Neuron& neuron : m_neurons)
    {
        outputs.push_back(neuron.activate(inputs));
    }

    return outputs;
}
