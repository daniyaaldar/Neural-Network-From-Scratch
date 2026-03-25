#pragma once
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include "Matrix.h"

class MathUtility
{
public:
    enum class ActivationFunction
    {
        LINEAR,
        RELU,
        SIGMOID
    };

    enum class ErrorFunction
    {
        MEAN_SQUARED,
        ROOT_MEAN_SQUARED,
        CROSS_ENTROPY,
        BINARY_CROSS_ENTROPY,
        MEAN_ABSOLUTE
    };

    static double               getRandomData(double min = -1, double max = 1);
    static std::vector<double>  getRandomData(size_t size, double min = -1, double max = 1);

    static double               dot(const std::vector<double>& a, const std::vector<double>& b);

    static double               linear(double x);
    static double               sigmoid(double x);
    static double               ReLu(double x);
    static std::vector<double>  softmax(const std::vector<double>& values);

    static double               linearDerivative(double x);
    static double               sigmoidDerivative(double x);
    static double               ReLuDerivative(double x);

    static double               meanSquaredError(const std::vector<double>& outputs, const std::vector<double>& target);
    static double               rootMeanSquaredError(const std::vector<double>& outputs, const std::vector<double>& target);
    static double               crossEntropyError(const std::vector<double>& outputs, const std::vector<double>& target);
    static double               binaryCrossEntropyError(const std::vector<double>& outputs, const std::vector<double>& target);
    static double               meanAbsoluteError(const std::vector<double>& outputs, const std::vector<double>& target);

    static std::function<double(double)>                                                        getActivationFunc(ActivationFunction activationFunc);
    static std::function<double(double)>                                                        getActivationDerivativeFunc(ActivationFunction activationFunc);
    static std::function<double(const std::vector<double>&, const std::vector<double>&)>        getErrorFunc(ErrorFunction errorFunction);

private:
    static std::mt19937& getRNG();
};