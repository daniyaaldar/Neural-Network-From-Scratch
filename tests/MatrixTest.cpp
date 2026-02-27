#include <gtest/gtest.h>
#include "Matrix.h"

TEST(MatrixTest, ConstructorTest)
{
    EXPECT_NO_THROW(Matrix m1);
    EXPECT_NO_THROW(Matrix m2(2, 3, 0.0));
    EXPECT_NO_THROW(Matrix m3(2, 2, { 1, 2, 3, 4 }));

    EXPECT_THROW(Matrix m4(3, 2, { 1, 2, 3, 4 }), std::out_of_range);
    EXPECT_THROW(Matrix m5(2, 1, { 1, 2, 3, 4 }), std::out_of_range);
    EXPECT_THROW(Matrix m6(0, 2, { 1, 2 }), std::out_of_range);
    EXPECT_THROW(Matrix m7(2, 0, { 1, 2 }), std::out_of_range);
}

TEST(MatrixTest, CopyConstructor)
{
    Matrix m1(2, 2, { 1,2,3,4 });
    Matrix m2(m1);

    EXPECT_EQ(m1, m2);
}

TEST(MatrixTest, OperatorParenthesesOverload)
{
    Matrix m(2, 2, { 1,2,3,4 });

    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(1, 0), 3);
    EXPECT_EQ(m(1, 1), 4);

    m(0, 0) = 10;
    EXPECT_EQ(m(0, 0), 10);
}

TEST(MatrixTest, GetRowTest)
{
    Matrix m(2, 2, { 1,2,3,4 });

    auto col0 = m.getCol(0);
    EXPECT_EQ(col0.size(), 2);
    EXPECT_EQ(col0[0], 1);
    EXPECT_EQ(col0[1], 3);
}

TEST(MatrixTest, GetColTest)
{
    Matrix m(2, 2, { 1,2,3,4 });

    auto row0 = m.getRow(0);
    EXPECT_EQ(row0.size(), 2);
    EXPECT_EQ(row0[0], 1);
    EXPECT_EQ(row0[1], 2);
}

TEST(MatrixTest, AdditionTest)
{
    Matrix m1(2, 2, { 1,2,3,4 });
    Matrix m2(2, 2, { 1,1,1,1 });

    Matrix result = m1 + m2;

    EXPECT_EQ(result(0, 0), 2);
    EXPECT_EQ(result(0, 1), 3);
    EXPECT_EQ(result(1, 0), 4);
    EXPECT_EQ(result(1, 1), 5);
}

TEST(MatrixTest, MultiplicationTest)
{
    Matrix A(2, 2, {
        1,2,
        3,4
    });

    Matrix B(2, 2, {
        5,6,
        7,8
    });

    Matrix C = A * B;

    EXPECT_EQ(C(0, 0), 19);
    EXPECT_EQ(C(0, 1), 22);
    EXPECT_EQ(C(1, 0), 43);
    EXPECT_EQ(C(1, 1), 50);
}

TEST(MatrixTest, DotProductTest)
{
    Matrix m1(2, 2, { 1,2,3,4 });
    Matrix m2(2, 2, { 1,1,1,1 });

    EXPECT_DOUBLE_EQ(m1.dot(m2), 10);
}

TEST(MatrixTest, AssignmentOperatorTest)
{
    Matrix m1(2, 2, { 1,2,3,4 });
    Matrix m2;

    m2 = m1;

    EXPECT_EQ(m1, m2);
    EXPECT_EQ(m1.GetCols(), m2.GetCols());
    EXPECT_EQ(m1.GetRows(), m2.GetRows());
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(0, 1), 2);
    EXPECT_EQ(m2(1, 0), 3);
    EXPECT_EQ(m2(1, 1), 4);
}

TEST(MatrixTest, BoundsCheckingTest)
{
    Matrix m(2, 2, { 1,2,3,4 });

    EXPECT_THROW(m(2, 0), std::out_of_range);
    EXPECT_THROW(m(0, 2), std::out_of_range);
    EXPECT_THROW(m.getRow(-1), std::out_of_range);
    EXPECT_THROW(m.getCol(-1), std::out_of_range);
    EXPECT_THROW(m.getRow(5), std::out_of_range);
    EXPECT_THROW(m.getCol(5), std::out_of_range);
}

TEST(MatrixTest, DimensionMismatchTest)
{
    Matrix A(2, 2, { 1,2,3,4 });
    Matrix B(3, 3, { 1,1,1,1,1,1,1,1,1 });
    Matrix C(2, 3, { 1,1,1,1,1,1 });
    Matrix D(2, 2, { 1,1,1,1 });

    EXPECT_THROW(A + B, std::invalid_argument);
    EXPECT_THROW(C * D, std::invalid_argument);
}