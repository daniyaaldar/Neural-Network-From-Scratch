#include <gtest/gtest.h>
#include "NeuralNetwork.h"
#include "MathUtility.h"

TEST(NeuralNetworkTest, RejectEmptyLayout)
{
    std::vector<size_t> layout = {};
    EXPECT_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.1), std::invalid_argument);
}

TEST(NeuralNetworkTest, RejectSingleLayerNetwork)
{
    std::vector<size_t> layout = { 5 };
    EXPECT_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.1), std::invalid_argument);
}

TEST(NeuralNetworkTest, RejectZeroNeuronsInLayer)
{
    std::vector<size_t> layout = { 4, 0, 3 };
    EXPECT_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.1), std::invalid_argument);
}

TEST(NeuralNetworkTest, RejectZeroLearningRate)
{
    std::vector<size_t> layout = { 4, 5, 3 };
    EXPECT_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.0), std::invalid_argument);
}

TEST(NeuralNetworkTest, AcceptValidNetworkSingleActivation)
{
    std::vector<size_t> layout = { 4, 5, 3 };
    EXPECT_NO_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::RELU, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.01));
}

TEST(NeuralNetworkTest, AcceptValidNetworkDualActivation)
{
    std::vector<size_t> layout = { 4, 5, 3 };
    EXPECT_NO_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::RELU, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.01));
}

TEST(NeuralNetworkTest, AcceptDeepNetworkStructureSingleActivation)
{
    std::vector<size_t> layout = { 128, 256, 128, 64, 10 };
    EXPECT_NO_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::RELU, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.001));
}

TEST(NeuralNetworkTest, AcceptDeepNetworkStructureDualActivation)
{
    std::vector<size_t> layout = { 128, 256, 128, 64, 10 };
    EXPECT_NO_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::RELU, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.001));
}

TEST(NeuralNetworkTest, ConstructorDoesNotThrowOnModerateNetworkSingleActivation)
{
    std::vector<size_t> layout = { 32, 64, 32 };
    EXPECT_NO_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.05));
}

TEST(NeuralNetworkTest, ConstructorDoesNotThrowOnModerateNetworkDualActivation)
{
    std::vector<size_t> layout = { 32, 64, 32 };
    EXPECT_NO_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ActivationFunction::LINEAR, MathUtility::ErrorFunction::ROOT_MEAN_SQUARED, 0.05));
}

TEST(NeuralNetworkTest, FeedForward)
{
    std::vector<size_t> layout = { 2, 2, 1 };
    NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::MEAN_SQUARED, 0.1);

    std::vector<double> inputs = { 0.5, 1.5 };

    std::vector<Matrix> weights =
    {
        Matrix(2, 2,
        {
            0.1, 0.2,
            0.3, 0.4
        }),
        Matrix(1, 2,
        {
            0.5, 0.6
        }),
    };

    std::vector<std::vector<double>> biases = {
        { -1.0, 2.0 },
        { -5.0 }
    };

    nn.setWeights(weights);
    nn.setBiases(biases);

    EXPECT_NO_THROW(nn.feedForward(inputs));
    EXPECT_NEAR(nn.getOutput()[0], 0.0138632, 1e-6);
}

TEST(NeuralNetworkTest, FeedForwardWithMatrix)
{
    // EXAMPLE FROM: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    std::vector<size_t> layout = { 2, 2, 2 };
    NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::MEAN_SQUARED, 0.1);

    std::vector<double> inputs = { 0.05, 0.1 };
    std::vector<double> target = { 0.01, 0.99 };

    std::vector<Matrix> weights =
    {
        Matrix(2, 2,
        {
            0.15, 0.20, // Layer 2 Neuron 1
            0.25, 0.30  // Layer 2 Neuron 2
        }),

        Matrix(2, 2,
        {
            0.40, 0.45, // Layer 3 Neuron 1
            0.50, 0.55  // Layer 3 Neuron 2
        }),
    };

    std::vector<std::vector<double>> biases = {
        { 0.35, 0.35 },
        { 0.6, 0.6 }
    };

    nn.setWeights(weights);
    nn.setBiases(biases);

    EXPECT_NO_THROW(nn.feedForward(inputs));
    EXPECT_NEAR(nn.getOutput()[0], 0.75136507, 1e-8);
    EXPECT_NEAR(nn.getOutput()[1], 0.772928465, 1e-8);
    EXPECT_NEAR(nn.getError(target), 0.298371109, 1e-8);
}

TEST(NeuralNetworkTest, BackpropGradients)
{
    std::vector<size_t> layout = { 2, 2, 2 };
    NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, MathUtility::ActivationFunction::SIGMOID, MathUtility::ErrorFunction::MEAN_SQUARED, 0.1);

    std::vector<double> inputs = { 0.05, 0.1 };
    std::vector<double> target = { 0.01, 0.99 };

    std::vector<Matrix> weights =
    {
        Matrix(2, 2,
        {
            0.15, 0.20, // Layer 2 Neuron 1
            0.25, 0.30  // Layer 2 Neuron 2
        }),

        Matrix(2, 2,
        {
            0.40, 0.45, // Layer 3 Neuron 1
            0.50, 0.55  // Layer 3 Neuron 2
        }),
    };

    std::vector<std::vector<double>> biases = {
        { 0.35, 0.35 },
        { 0.6, 0.6 }
    };

    nn.setWeights(weights);
    nn.setBiases(biases);
    nn.feedForward(inputs);
    nn.backwardsPropagate(target);

    // Output layer deltas
    EXPECT_NEAR(nn.getDelta(2, 0), 0.13849856, 1e-6);
    EXPECT_NEAR(nn.getDelta(2, 1), -0.03809824, 1e-6);

    // Hidden layer deltas
    EXPECT_NEAR(nn.getDelta(1, 0), 0.00877139, 1e-6);
    EXPECT_NEAR(nn.getDelta(1, 1), 0.00995425, 1e-6);
}

TEST(NeuralNetworkTest, BackpropWeightUpdate)
{
    std::vector<size_t> layout = { 2, 2, 2 };
    NeuralNetwork nn(layout,
        MathUtility::ActivationFunction::SIGMOID,
        MathUtility::ActivationFunction::SIGMOID,
        MathUtility::ErrorFunction::MEAN_SQUARED,
        0.5);

    std::vector<double> inputs = { 0.05, 0.1 };
    std::vector<double> target = { 0.01, 0.99 };
    std::vector<Matrix> weights =
    {
        Matrix(2, 2,
        {
            0.15, 0.20,
            0.25, 0.30
        }),
        Matrix(2, 2,
        {
            0.40, 0.45,
            0.50, 0.55
        }),
    };
    std::vector<std::vector<double>> biases = {
        { 0.35, 0.35 },
        { 0.6, 0.6 }
    };
    nn.setWeights(weights);
    nn.setBiases(biases);
    nn.feedForward(inputs);
    nn.backwardsPropagate(target);

    // hidden to output weights
    EXPECT_NEAR(nn.getWeight(2, 0, 0), 0.35891648, 1e-6);
    EXPECT_NEAR(nn.getWeight(2, 0, 1), 0.408666186, 1e-6);
    EXPECT_NEAR(nn.getWeight(2, 1, 0), 0.511301270, 1e-6);
    EXPECT_NEAR(nn.getWeight(2, 1, 1), 0.561370121, 1e-6);

    // input to hidden weights
    EXPECT_NEAR(nn.getWeight(1, 0, 0), 0.149780716, 1e-6);
    EXPECT_NEAR(nn.getWeight(1, 0, 1), 0.199561432, 1e-6);
    EXPECT_NEAR(nn.getWeight(1, 1, 0), 0.249751244, 1e-6);
    EXPECT_NEAR(nn.getWeight(1, 1, 1), 0.299502488, 1e-6);
}

TEST(NeuralNetworkTest, XORLearning)
{
    std::vector<size_t> layout = { 2, 4, 1 };
    NeuralNetwork nn(layout,
        MathUtility::ActivationFunction::SIGMOID,
        MathUtility::ActivationFunction::SIGMOID,
        MathUtility::ErrorFunction::MEAN_SQUARED,
        0.5,
        false);

    std::vector<std::vector<double>> inputs = {
        { 0.0, 0.0 },
        { 0.0, 1.0 },
        { 1.0, 0.0 },
        { 1.0, 1.0 }
    };

    std::vector<std::vector<double>> targets = {
        { 0.0 },
        { 1.0 },
        { 1.0 },
        { 0.0 }
    };

    std::vector<Matrix> weights =
    {
        Matrix(4, 2,
        {
            0.15, 0.20,
            0.25, 0.30,
            0.35, 0.40,
            0.45, 0.50
        }),
        Matrix(1, 4,
        {
            0.55, 0.60,
            0.65, 0.70
        }),
    };

    std::vector<std::vector<double>> biases = {
        { 0.35, 0.35, 0.35, 0.35 },
        { 0.6 }
    };

    nn.setWeights(weights);
    nn.setBiases(biases);
    nn.learn(10000, inputs, targets, 0);

    nn.feedForward({ 0.0, 0.0 });
    EXPECT_NEAR(nn.getOutput()[0], 0.0, 0.1);

    nn.feedForward({ 0.0, 1.0 });
    EXPECT_NEAR(nn.getOutput()[0], 1.0, 0.1);

    nn.feedForward({ 1.0, 0.0 });
    EXPECT_NEAR(nn.getOutput()[0], 1.0, 0.1);

    nn.feedForward({ 1.0, 1.0 });
    EXPECT_NEAR(nn.getOutput()[0], 0.0, 0.1);
}