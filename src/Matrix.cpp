#include "Matrix.h"
#include "MathUtility.h"
#include <iostream>
#include <string>

Matrix::Matrix()
    : Matrix(0, 0, 0.0)
{
}

Matrix::Matrix(size_t rows, size_t cols, double fillValue)
    : Matrix(rows, cols, std::vector<double>(rows * cols, fillValue))
{
}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& data)
    : 
    m_rows(rows),
    m_cols(cols),
    m_data(data)
{
    if (m_data.size() != m_rows * m_cols)
    {
        throw std::out_of_range("Data size does not match matrix dimensions");
    }
}

Matrix::Matrix(const Matrix& m)
    : Matrix(m.m_rows, m.m_cols, m.m_data)
{
}

std::vector<double> Matrix::getRow(size_t row) const
{
    if (row >= m_rows)
    {
        throw std::out_of_range("Matrix row index out of range");
    }

    std::vector<double> result;
    result.reserve(m_cols);

    for (size_t c = 0; c < m_cols; ++c)
    {
        result.push_back(m_data[row * m_cols + c]);
    }

    return result;
}

std::vector<double> Matrix::getCol(size_t col) const
{
    if (col >= m_cols)
    {
        throw std::out_of_range("Matrix col index out of range");
    }

    std::vector<double> result;
    result.reserve(m_rows);

    for (size_t r = 0; r < m_rows; ++r)
    {
        result.push_back(m_data[r * m_cols + col]);
    }

    return result;
}

Matrix Matrix::Transpose() const
{
    Matrix m(m_cols, m_rows);

    for (size_t i = 0; i < m_rows; i++)
    {
        for (size_t j = 0; j < m_cols; j++)
        {
            m.SetValue(j, i, GetValue(i, j));
        }
    }

    return m;
}

double Matrix::dot(const Matrix& other) const
{
    if (m_rows * m_cols != other.m_rows * other.m_cols)
    {
        throw std::invalid_argument("Dot product requires same number of elements");
    }

    const size_t n = m_rows * m_cols;
    double sum = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        sum += m_data[i] * other.m_data[i];
    }

    return sum;
}

void Matrix::print() const
{
    for (size_t i = 0; i < m_rows; i++)
    {
        for (size_t j = 0; j < m_cols; j++)
        {
            std::cout << GetValue(i, j) << ", ";
        }
        std::cout << "\n";
    }        
    std::cout << "\n";
}

Matrix Matrix::operator+(const Matrix& other) const
{
    if (this->m_cols != other.m_cols || this->m_rows != other.m_rows)
    {
        throw std::invalid_argument("Matrix dimensions do not allow addition");
    }

    Matrix m = *this;

    for (size_t idx = 0; idx < other.m_data.size(); idx++)
    {
        m.m_data[idx] += other.m_data[idx];
    }

    return m;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (this->m_cols != other.m_rows)
    {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication");
    }

    Matrix m(this->m_rows, other.m_cols, 0.0);

    for (size_t i = 0; i < this->m_rows; i++)
    {
        for (size_t j = 0; j < other.m_cols; j++)
        {
            for (size_t k = 0; k < this->m_cols; k++)
            {
                m.m_data[i * m.m_cols + j] += this->m_data[i * m_cols + k] * other.m_data[k * other.m_cols + j];
            }
        }
    }

    return m;
}

Matrix& Matrix::operator+=(const Matrix& other)
{
    Matrix m = *this + other;
    *this = m;
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other)
{
    Matrix m = *this * other;
    *this = m;
    return *this;
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this != &other)
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = other.m_data;
    }

    return *this;
}

bool Matrix::operator==(const Matrix& other) const
{
    return m_rows == other.m_rows && 
           m_cols == other.m_cols && 
           m_data == other.m_data;
}

double& Matrix::operator()(size_t row, size_t col)
{
    if (row >= m_rows || col >= m_cols)
    {
        throw std::out_of_range("Matrix index out of range");
    }

    return m_data[row * m_cols + col];
}

const double& Matrix::operator()(size_t row, size_t col) const
{
    if (row >= m_rows || col >= m_cols)
    {
        throw std::out_of_range("Matrix index out of range");
    }

    return m_data[row * m_cols + col];
}