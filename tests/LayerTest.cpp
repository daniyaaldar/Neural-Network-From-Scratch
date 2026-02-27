#include <gtest/gtest.h>
#include "Layer.h"
#include "MathUtility.h"

class LayerTest : public ::testing::Test
{
protected:
    size_t layerIdx = 1;
    size_t neurons = 2;
    size_t inputsPerNeuron = 3;
    size_t outputsPerNeuron = 3;
    MathUtility::ActivationFunction activationFunc = MathUtility::ActivationFunction::LINEAR;
    double learningRate = 0.1;
    bool initialiseRandomData = false; // deterministic (biases and weights are set to zero)

    Layer layer{
        layerIdx,
        neurons,
        inputsPerNeuron,
        outputsPerNeuron,
        activationFunc,
        learningRate,
        initialiseRandomData
    };
};

//
// Constructor
//

TEST_F(LayerTest, ConstructorCreatesCorrectNeuronCount)
{
    auto outputs = layer.getOutputs();
    EXPECT_EQ(outputs.size(), neurons);
}

TEST(LayerConstructorTest, ZeroNeuronsAllowedButProducesEmptyLayer)
{
    Layer l(0, 0, 3, 3, MathUtility::ActivationFunction::LINEAR, 0.1, false);
    EXPECT_EQ(l.getOutputs().size(), 0);
}

//
// setOutputs / getOutputs
//

TEST_F(LayerTest, SetOutputsValid)
{
    std::vector<double> values = { 1.0, 2.0 };
    layer.setOutputs(values);

    auto outputs = layer.getOutputs();
    EXPECT_DOUBLE_EQ(outputs[0], 1.0);
    EXPECT_DOUBLE_EQ(outputs[1], 2.0);
}

TEST_F(LayerTest, SetOutputsWrongSizeThrows)
{
    std::vector<double> values = { 1.0 };
    EXPECT_THROW(layer.setOutputs(values), std::out_of_range);
}

//
// setOutput / getOutput
//

TEST_F(LayerTest, SetSingleOutputValid)
{
    layer.setOutput(0, 5.0);
    EXPECT_DOUBLE_EQ(layer.getOutput(0), 5.0);
}

TEST_F(LayerTest, SetSingleOutputOutOfBoundsThrows)
{
    EXPECT_THROW(layer.setOutput(999, 1.0), std::out_of_range);
}

TEST_F(LayerTest, GetOutputOutOfBoundsShouldThrow)
{
    EXPECT_THROW(layer.getOutput(999), std::out_of_range);
}

//
// setWeights / getWeights
//

TEST_F(LayerTest, SetWeightsValid)
{
    std::vector<std::vector<double>> weights =
    {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0}
    };

    layer.setWeights(weights);

    auto returned = layer.getWeights();

    EXPECT_EQ(returned.size(), 2);
    EXPECT_EQ(returned[0].size(), 3);
    EXPECT_DOUBLE_EQ(returned[0][0], 1.0);
    EXPECT_DOUBLE_EQ(returned[1][0], 2.0);
}

TEST_F(LayerTest, SetWeightsWrongNeuronCountThrows)
{
    std::vector<std::vector<double>> weights = { {1.0, 1.0, 1.0} };
    EXPECT_THROW(layer.setWeights(weights), std::out_of_range);
}

TEST_F(LayerTest, SetWeightsWrongInputSizeThrows)
{
    std::vector<double> badWeights = { 1.0, 2.0 };
    EXPECT_THROW(layer.setWeights(0, badWeights), std::out_of_range);
}

TEST_F(LayerTest, GetWeightsOutOfBoundsShouldThrow)
{
    EXPECT_THROW(layer.getWeights(999), std::out_of_range);
}

//
// Bias Tests
//

TEST_F(LayerTest, SetAndGetBiasesValid)
{
    std::vector<double> biases = { 0.5, -0.5 };
    layer.setBiases(biases);

    auto returned = layer.getBiases();

    EXPECT_DOUBLE_EQ(returned[0], 0.5);
    EXPECT_DOUBLE_EQ(returned[1], -0.5);
}

TEST_F(LayerTest, SetBiasesWrongSizeThrows)
{
    std::vector<double> biases = { 0.5 };
    EXPECT_THROW(layer.setBiases(biases), std::out_of_range);
}

TEST_F(LayerTest, SetSingleBiasValid)
{
    layer.setBias(0, 3.14);
    EXPECT_DOUBLE_EQ(layer.getBias(0), 3.14);
}

TEST_F(LayerTest, GetBiasOutOfBoundsShouldThrow)
{
    EXPECT_THROW(layer.getBias(999), std::out_of_range);
}

//
// feedForward
//

TEST_F(LayerTest, FeedForwardLinearNoBias)
{
    std::vector<std::vector<double>> weights =
    {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0}
    };

    std::vector<double> biases = { 0.0, 0.0 };

    layer.setWeights(weights);
    layer.setBiases(biases);

    std::vector<double> inputs = { 1.0, 1.0, 1.0 };

    layer.feedForward(inputs);

    auto outputs = layer.getOutputs();

    EXPECT_DOUBLE_EQ(outputs[0], 3.0);
    EXPECT_DOUBLE_EQ(outputs[1], 6.0);
}

TEST_F(LayerTest, FeedForwardWithBias)
{
    std::vector<std::vector<double>> weights =
    {
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0}
    };

    std::vector<double> biases = { 1.0, -1.0 };

    layer.setWeights(weights);
    layer.setBiases(biases);

    std::vector<double> inputs = { 1.0, 1.0, 1.0 };

    layer.feedForward(inputs);

    auto outputs = layer.getOutputs();

    EXPECT_DOUBLE_EQ(outputs[0], 4.0);  // 3 + 1
    EXPECT_DOUBLE_EQ(outputs[1], 2.0);  // 3 - 1
}

TEST_F(LayerTest, FeedForwardWrongInputSizeShouldThrow)
{
    std::vector<double> badInputs = { 1.0 };
    EXPECT_THROW(layer.feedForward(badInputs), std::out_of_range);
}

//
// Stability / Consistency Tests
//

TEST_F(LayerTest, RepeatedFeedForwardProducesSameResult)
{
    std::vector<std::vector<double>> weights =
    {
        {1.0, 2.0, 3.0},
        {3.0, 2.0, 1.0}
    };

    layer.setWeights(weights);
    layer.setBiases({ 0.0, 0.0 });

    std::vector<double> inputs = { 1.0, 1.0, 1.0 };

    layer.feedForward(inputs);
    auto first = layer.getOutputs();

    layer.feedForward(inputs);
    auto second = layer.getOutputs();

    EXPECT_DOUBLE_EQ(first[0], second[0]);
    EXPECT_DOUBLE_EQ(first[1], second[1]);
}