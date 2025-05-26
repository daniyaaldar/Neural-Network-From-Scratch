#include "MathUtility.h"
#include <random>

std::vector<double> MathUtility::getRandomData(size_t size, double min, double max)
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);

    std::vector<double> result(size);
    for (auto& val : result)
        val = dist(gen);

    return result;
}

double MathUtility::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double MathUtility::sigmoidDerivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double MathUtility::dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double sum = 0.0;

    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

double MathUtility::meanSquaredError(const std::vector<double>& output, const std::vector<double>& target)
{
    double error = 0;

    for (size_t  i = 0; i < output.size(); i++)
    {
        error += std::pow((target[i] - output[i]), 2);
    }

    return error;
}

std::vector<double> MathUtility::softmax(const std::vector<double>& values)
{
    std::vector<double> result(values.size());
    double total = 0.0;

    double maxVal = *std::max_element(values.begin(), values.end()); // Sub max val to find relative difference

    for (size_t i = 0; i < values.size(); i++) {
        total += std::exp(values[i] - maxVal);
    }

    for (size_t i = 0; i < values.size(); i++) {
        result[i] = std::exp(values[i] - maxVal) / total;
    }

    return result;
}
