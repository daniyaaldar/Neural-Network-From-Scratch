#include <gtest/gtest.h>
#include "NeuralNetwork.h"
#include "MathUtility.h"

//
// Helper function
//

static std::vector<size_t> MakeValidLayout()
{
    return { 4, 5, 3 };
}

//
// Constructor tests
//

TEST(NeuralNetworkTest, RejectEmptyLayout)
{
    std::vector<size_t> layout = {};
    EXPECT_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.1), std::invalid_argument);
}

TEST(NeuralNetworkTest, RejectSingleLayerNetwork)
{
    std::vector<size_t> layout = { 5 };
    EXPECT_THROW( NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.1), std::invalid_argument);
}

TEST(NeuralNetworkTest, RejectZeroNeuronsInLayer)
{
    std::vector<size_t> layout = { 4, 0, 3 };
    EXPECT_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.1), std::invalid_argument);
}

TEST(NeuralNetworkTest, RejectZeroLearningRate)
{
    std::vector<size_t> layout = MakeValidLayout();
    EXPECT_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.0), std::invalid_argument);
}

TEST(NeuralNetworkTest, AcceptValidNetwork)
{
    std::vector<size_t> layout = MakeValidLayout();
    EXPECT_NO_THROW(NeuralNetwork nn(layout, MathUtility::ActivationFunction::RELU, 0.01));
}

TEST(NeuralNetworkTest, AcceptDeepNetworkStructure)
{
    std::vector<size_t> layout = { 128, 256, 128, 64, 10 };
    EXPECT_NO_THROW( NeuralNetwork nn(layout, MathUtility::ActivationFunction::RELU, 0.001));
}

TEST(NeuralNetworkTest, ConstructorDoesNotThrowOnModerateNetwork)
{
    std::vector<size_t> layout = { 32, 64, 32 };
    EXPECT_NO_THROW( NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.05));
}

//
// Forward pass safety test
//

TEST(NeuralNetworkTest, FeedFoward)
{
    std::vector<size_t> layout = { 2, 2, 1 };
    NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.1);

    std::vector<double> inputs = { 0.5, 1.5 };

    std::vector<std::vector<std::vector<double>>> weights = {
        { { { 0.1, 0.2 }, { 0.3, 0.4 } } },
        { { { 0.5, 0.6 } } }
    };

    std::vector<std::vector<double>> biases = {
        { { -1.0, 2.0 } },
        { { -5.0 } }
    };

    nn.setWeights(weights);
    nn.setBiases(biases);

    EXPECT_NO_THROW(nn.feedForward(inputs));
    EXPECT_NEAR(nn.getOutput()[0], 0.0138632, 1e-6);
}

TEST(NeuralNetworkTest, FeedForwardAndCalculateMSE)
{
    std::vector<size_t> layout = { 2, 2, 1 };
    NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.1);

    std::vector<double> inputs = { 0.5, 1.5 };
    std::vector<double> targetOutput = { 1.0 };

    std::vector<std::vector<std::vector<double>>> weights = {
        { { { 0.1, 0.2 }, { 0.3, 0.4 } } },
        { { { 0.5, 0.6 } } }
    };

    std::vector<std::vector<double>> biases = {
        { { -1.0, 2.0 } },
        { { -5.0 } }
    };

    nn.setWeights(weights);
    nn.setBiases(biases);

    EXPECT_NO_THROW(nn.feedForward(inputs));
    EXPECT_NEAR(MathUtility::cost(nn.getOutput(), targetOutput), 0.9724657, 1e-6);
}

TEST(NeuralNetworkTest, FeedFowardWithMatrix)
{
    std::vector<size_t> layout = { 2, 2, 1 };
    NeuralNetwork nn(layout, MathUtility::ActivationFunction::SIGMOID, 0.1);

    std::vector<double> inputs = { 0.5, 1.5 };
    std::vector<double> targetOutput = { 1.0 };

    std::vector<Matrix> weights = 
    { 
        Matrix(2, 2,
        {
            0.1, 0.2, // Layer 2 Neuron 1
            0.3, 0.4  // Layer 2 Neuron 2
        }),

        Matrix(1, 2,
        {
            0.5, 0.6  // Layer 3 Neuron 1
        })
    };

    std::vector<std::vector<double>> biases = {
        { { -1.0, 2.0 } },
        { { -5.0 } }
    };

    nn.setWeights(weights);
    nn.setBiases(biases);

    EXPECT_NO_THROW(nn.feedForward(inputs));
    EXPECT_NEAR(nn.getOutput()[0], 0.0138632, 1e-6);
}