#pragma once

#include <vector>
#include <string>

struct Image
{
    std::vector < double > data;
    uint8_t rowSize;
    uint8_t colSize;
};

class MNISTLoader
{
public:
    static std::vector<std::vector<double>> load_mnist_images(const std::string& filename, int& rows, int& cols);
    static std::vector<uint8_t> load_mnist_labels(const std::string& filename);
    static void print_image(const std::vector<double>& img, int rows, int cols);
    static void print_image(const std::vector<size_t>& img, int rows, int cols);

private:
    static int32_t readInt32(std::ifstream& file);
};