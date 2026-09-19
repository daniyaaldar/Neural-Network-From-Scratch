#include <iostream>
#include <string>

#include "NeuralNetwork.h"
#include "MathUtility.h"
#include "MNIST_dataloader.h"

#define NUM_OF_INPUTS 28 * 28
#define NUM_OF_OUTPUTS 10
#define BATCH_SIZE 1000
#define NUM_EPOCHS 10

int main()
{
    std::cout << "Loading MNIST dataset...\n";

    int rows = 0;
    int cols = 0;

    std::vector<std::vector<double>> trainImages;
    std::vector<std::vector<double>> testImages;
    std::vector<uint8_t> trainLabels;
    std::vector<uint8_t> testLabels;

    try
    {
        trainImages = MNISTLoader::load_mnist_images("../../../../data/train-images.idx3-ubyte", rows, cols);
        trainLabels = MNISTLoader::load_mnist_labels("../../../../data/train-labels.idx1-ubyte");

        testImages = MNISTLoader::load_mnist_images("../../../../data/t10k-images.idx3-ubyte", rows, cols);
        testLabels = MNISTLoader::load_mnist_labels("../../../../data/t10k-labels.idx1-ubyte");
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to load MNIST dataset: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Dataset loaded. Training on " << trainImages.size() << " samples.\n";

    // Build  targets for training
    std::vector<std::vector<double>> trainTargets;
    for (uint8_t label : trainLabels)
    {
        std::vector<double> target(10, 0.0);
        target[label] = 1.0;
        trainTargets.push_back(target);
    }
    std::vector<size_t> topology = { 784, 16, 16, 10 };

    NeuralNetwork nn(topology, MathUtility::ActivationFunction::SIGMOID,
        MathUtility::ActivationFunction::LINEAR,
        MathUtility::ErrorFunction::MEAN_SQUARED, 0.001);

    std::cout << "Training neural network...\n";

    nn.learn(40, trainImages, trainTargets, 1);

    // Evaluate on test set
    int correct = 0;
    for (size_t i = 0; i < testImages.size(); i++)
    {
        const std::vector<double>& outputs = nn.feedForward(testImages[i]);
        int predicted = std::max_element(outputs.begin(), outputs.end()) - outputs.begin();
        if (predicted == testLabels[i])
            ++correct;
    }

    std::cout << "Test Accuracy: " << 100.0 * correct / testImages.size() << "%\n";
}