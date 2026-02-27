#pragma once

#include <functional>
#include <random>

class Matrix
{
public:
    Matrix();
    Matrix(size_t rows, size_t cols, double fillValue = 0.0);
    Matrix(size_t rows, size_t cols, const std::vector<double>& data);
    Matrix(const Matrix& m);

    void SetValue(size_t row, size_t col, double value) { m_data[row * m_cols + col] = value; }

    size_t GetRows() const { return m_rows; }
    size_t GetCols() const { return m_cols; }
    double GetValue(size_t row, size_t col) const { return m_data[row * m_cols + col]; }

    std::vector<double> getRow(size_t row) const;
    std::vector<double> getCol(size_t col) const;

    Matrix Transpose() const;
    double dot(const Matrix& other) const;
    void print() const;

    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix& operator*=(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    bool operator==(const Matrix& other) const;
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

private:
    size_t m_rows;
    size_t m_cols;
    std::vector<double> m_data;
};