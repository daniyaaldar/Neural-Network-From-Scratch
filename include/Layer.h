#pragma once

#include "MathUtility.h"
#include "Neuron.h"

class Layer {
public:
    Layer(size_t numOfNeurons, size_t numOfinputsPerNeuron);
    std::vector < Neuron > getNeurons() const { return m_neurons; };
    std::vector < double > forwardPropagate(const std::vector<double>& inputs) const;

private:
    std::vector < Neuron > m_neurons;
};