#include "MathUtility.h"
#include <random>

double MathUtility::getRandomData(double min, double max)
{
    if (min > max)
        throw std::invalid_argument("min must be <= max");

    std::uniform_real_distribution<double> dist(min, max);

    return dist(getRNG());
}

std::vector<double> MathUtility::getRandomData(size_t size, double min, double max)
{
    if (min > max)
        throw std::invalid_argument("min must be <= max");

    std::uniform_real_distribution<double> dist(min, max);
    std::vector<double> result(size);

    for (double& val : result)
    {
        val = dist(getRNG());
    }

    return result;
}

double MathUtility::dot(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("Dot product requires equal-length vectors");

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

double MathUtility::linear(double x)
{
    return x;
}

double MathUtility::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double MathUtility::ReLu(double x)
{
    return std::max(0.0, x);
}

std::vector<double> MathUtility::softmax(const std::vector<double>& values)
{
    if (values.empty())
        throw std::invalid_argument("Empty values");

    std::vector<double> result(values.size());

    double total = 0.0;
    double maxVal = *std::max_element(values.begin(), values.end()); // Sub max val to find relative difference

    for (size_t i = 0; i < values.size(); i++)
    {
        total += std::exp(values[i] - maxVal);
    }

    for (size_t i = 0; i < values.size(); i++)
    {
        result[i] = std::exp(values[i] - maxVal) / total;
    }

    return result;
}

double MathUtility::linearDerivative(double x)
{
    return 1.0;
}

double MathUtility::sigmoidDerivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double MathUtility::ReLuDerivative(double x)
{
    return x > 0.0 ? 1.0 : 0.0;
}

double MathUtility::meanSquaredError(const std::vector<double>& outputs, const std::vector<double>& target)
{
    double error = 0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        error += 0.5 * std::pow(target[i] - outputs[i], 2);
    }

    return error;
}

double MathUtility::rootMeanSquaredError(const std::vector<double>& outputs, const std::vector<double>& target)
{
    if (outputs.empty())
        throw std::invalid_argument("Empty outputs");
    if (outputs.size() != target.size())
        throw std::invalid_argument("outputs and target must be the same size");

    double error = 0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        error += std::pow(target[i] - outputs[i], 2);
    }

    error /= outputs.size();
    return std::sqrt(error);
}

double MathUtility::crossEntropyError(const std::vector<double>& outputs, const std::vector<double>& target)
{
    if (outputs.empty())
        throw std::invalid_argument("Empty outputs");
    if (outputs.size() != target.size())
        throw std::invalid_argument("outputs and target must be the same size");

    double error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        error -= target[i] * std::log(outputs[i] + 1e-10);
    }

    return error;
}

double MathUtility::binaryCrossEntropyError(const std::vector<double>& outputs, const std::vector<double>& target)
{
    if (outputs.empty())
        throw std::invalid_argument("Empty outputs");
    if (outputs.size() != target.size())
        throw std::invalid_argument("outputs and target must be the same size");

    double error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        error -= target[i] * std::log(outputs[i] + 1e-10) + (1.0 - target[i]) * std::log(1.0 - outputs[i] + 1e-10);
    }

    return error / outputs.size();
}

double MathUtility::meanAbsoluteError(const std::vector<double>& outputs, const std::vector<double>& target)
{
    if (outputs.empty())
        throw std::invalid_argument("Empty outputs");
    if (outputs.size() != target.size())
        throw std::invalid_argument("outputs and target must be the same size");

    double error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        error += std::abs(target[i] - outputs[i]);
    }

    return error / outputs.size();
}

std::function<double(double)> MathUtility::getActivationFunc(ActivationFunction activationFunc)
{
    switch (activationFunc)
    {
    case ActivationFunction::LINEAR:
        return linear;
    case ActivationFunction::RELU:
        return ReLu;
    case ActivationFunction::SIGMOID:
        return sigmoid;
    default:
        throw std::invalid_argument("Unhandled activation function");
    }
}

std::function<double(double)> MathUtility::getActivationDerivativeFunc(ActivationFunction activationFunc)
{
    switch (activationFunc)
    {
    case MathUtility::ActivationFunction::LINEAR:
        return linearDerivative;
    case MathUtility::ActivationFunction::RELU:
        return ReLuDerivative;
    case MathUtility::ActivationFunction::SIGMOID:
        return sigmoidDerivative;
    default:
        throw std::invalid_argument("Unhandled activation function");
    }
}

std::function<double(const std::vector<double>&, const std::vector<double>&)> MathUtility::getErrorFunc(ErrorFunction errorFunction)
{
    switch (errorFunction)
    {
    case ErrorFunction::MEAN_SQUARED:
        return meanSquaredError;
    case ErrorFunction::ROOT_MEAN_SQUARED:
        return rootMeanSquaredError;
    case ErrorFunction::CROSS_ENTROPY:
        return crossEntropyError;
    case ErrorFunction::BINARY_CROSS_ENTROPY:
        return binaryCrossEntropyError;
    case ErrorFunction::MEAN_ABSOLUTE:
        return meanAbsoluteError;
    default:
        throw std::invalid_argument("Unhandled error function");
    }
}

std::mt19937& MathUtility::getRNG()
{
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}