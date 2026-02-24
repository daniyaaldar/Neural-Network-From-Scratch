#include <gtest/gtest.h>
#include "Neuron.h"

TEST(NeuronTest, Construction)
{
    Neuron n(3, MathUtility::ActivationFunction::RELU, 0.0001);
    EXPECT_EQ(n.getNumOfInputs(), 3);
}