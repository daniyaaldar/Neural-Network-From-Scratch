#pragma once

#include "MathUtility.h"
#include "Layer.h"

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(std::vector < Layer > layers);

    std::vector<double> forwardPropagate(std::vector < double >& inputs) const;

private:
    std::vector < Layer > m_layers;
};
