#include "MNIST_dataloader.h"

#include <iostream>
#include <fstream>

int32_t MNISTLoader::readInt32(std::ifstream& file) 
{
    uint8_t bytes[4] = {0};
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]);
}

std::vector<std::vector<double>> MNISTLoader::load_mnist_images(const std::string& filename, int& rows, int& cols)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    int32_t magic = readInt32(file);
    if (magic != 2051) throw std::runtime_error("Invalid image file magic number.");

    int num_images = readInt32(file);
    rows = readInt32(file);
    cols = readInt32(file);

    std::vector<std::vector<double>> images(num_images, std::vector<double>(rows * cols));
    std::vector<unsigned char> buffer(rows * cols);

    for (int i = 0; i < num_images; ++i) 
    {
        file.read(reinterpret_cast<char*>(buffer.data()), rows * cols);
        for (int j = 0; j < rows * cols; ++j) 
        {
            images[i][j] = static_cast<double>(buffer[j]) / 255.0;
        }
    }

    return images;
}

std::vector<uint8_t> MNISTLoader::load_mnist_labels(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open label file: " + filename);

    int32_t magic = readInt32(file);
    if (magic != 2049) throw std::runtime_error("Invalid label file magic number.");

    int num_labels = readInt32(file);
    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    return labels;
}

void MNISTLoader::print_image(const std::vector<double>& img, int rows, int cols)
{
    for (int i = 0; i < rows* cols; ++i)
    {
        std::cout << (img[i] > 0.5 ? '#' : ' ');
        if ((i + 1) % cols == 0) std::cout << '\n';
    }
}

void MNISTLoader::print_image(const std::vector<size_t>& img, int rows, int cols)
{
    for (int i = 0; i < rows* cols; ++i)
    {
        std::cout << (img[i] > 128 ? '#' : ' ');
        if ((i + 1) % cols == 0) std::cout << '\n';
    }
}