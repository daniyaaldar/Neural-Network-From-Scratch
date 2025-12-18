#include "Matrix.h"
#include "MathUtility.h"
#include <iostream>

Matrix::Matrix()
    : Matrix(0, 0, 0.0)
{
}

Matrix::Matrix(size_t numRows, size_t numCols)
    : Matrix(numRows, numCols, 0.0)
{
    for (double& v : m_data)
        v = MathUtility::getRandomData();
}

Matrix::Matrix(size_t numRows, size_t numCols, double fillValue)
    : Matrix(numRows, numCols, std::vector<double>(numRows * numCols, fillValue))
{
}

Matrix::Matrix(size_t numRows, size_t numCols, const std::vector<double>& data)
    : 
    m_numRows(numRows),
    m_numCols(numCols),
    m_data(data)
{
    if (m_data.size() != m_numRows * m_numCols)
        throw std::invalid_argument("Data size does not match matrix dimensions");
}

Matrix::Matrix(const Matrix& m)
    : Matrix(m.m_numRows, m.m_numCols, m.m_data)
{
}

Matrix Matrix::operator+(const Matrix& other) const
{
    if (this->m_numCols != other.m_numCols || this->m_numRows != other.m_numRows)
        throw std::invalid_argument("Matrix dimensions do not allow addition");

    Matrix m = *this;

    for (size_t idx = 0; idx < other.m_data.size(); idx++)
    {
        m.m_data[idx] += other.m_data[idx];
    }

    return m;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (this->m_numCols != other.m_numRows)
        throw std::invalid_argument("Matrix dimensions do not allow multiplication");

    Matrix m(this->m_numRows, other.m_numCols, 0.0);

    for (size_t i = 0; i < this->m_numRows; i++)
    {
        for (size_t j = 0; j < other.m_numCols; j++)
        {
            for (size_t k = 0; k < this->m_numCols; k++)
            {
                m.m_data[i * m.m_numCols + j] += this->m_data[i * m_numCols + k] * other.m_data[k * other.m_numCols + j];
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
        m_numRows = other.m_numRows;
        m_numCols = other.m_numCols;
        m_data = other.m_data;
    }

    return *this;
}

Matrix Matrix::Transpose() const
{
    Matrix m(m_numCols, m_numRows);

    for (size_t i = 0; i < m_numRows; i++)
    {
        for (size_t j = 0; j < m_numCols; j++)
        {
            m.SetValue(j, i, GetValue(i, j));
        }
    }

    return m;
}

double Matrix::dot(const Matrix& other) const
{
    if (m_numRows * m_numCols != other.m_numRows * other.m_numCols)
        throw std::invalid_argument("Dot product requires same number of elements");

    const size_t n = m_numRows * m_numCols;
    double sum = 0.0;

    for (size_t i = 0; i < n; ++i)
        sum += m_data[i] * other.m_data[i];

    return sum;
}

//double Matrix::dot(const Matrix& other) const
//{
//    return MathUtility::dot(m_data, other.m_data);
//}

void Matrix::print() const
{
    for (size_t i = 0; i < m_numRows; i++)
    {
        for (size_t j = 0; j < m_numCols; j++)
        {
            std::cout << GetValue(i, j) << ", ";
        }
        std::cout << "\n";
    }        
    std::cout << "\n";
}