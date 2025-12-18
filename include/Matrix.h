#pragma once

#include <functional>
#include <random>

class Matrix
{
public:
    Matrix();
    Matrix(size_t numRows, size_t numCols);
    Matrix(size_t numRows, size_t numCols, double fillValue);
    Matrix(size_t numRows, size_t numCols, const std::vector<double>& data);
    Matrix(const Matrix& m);

    void SetValue(size_t row, size_t col, double value) { m_data[row * m_numCols + col] = value; }

    size_t GetNumRows() const { return m_numRows; }
    size_t GetNumCols() const { return m_numCols; }
    double GetValue(size_t row, size_t col) const { return m_data[row * m_numCols + col]; }


    Matrix Transpose() const;
    double dot(const Matrix& other) const;
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