#pragma once

#include <functional>
#include <random>

class Matrix
{
public:
    Matrix(size_t numRows, size_t numCols);
    Matrix(size_t numRows, size_t numCols, double fillValue);
    Matrix(size_t numRows, size_t numCols, const std::vector<double>& data);
    Matrix(const Matrix& m);

    void print() const;

    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix& operator*=(const Matrix& other);
    Matrix& operator=(const Matrix& other);
private:
    size_t m_numRows;
    size_t m_numCols;
    std::vector<double> m_data;
};