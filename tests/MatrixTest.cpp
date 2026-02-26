#include <gtest/gtest.h>
#include "Matrix.h"

TEST(MatrixTest, Construction)
{
    Matrix m(3, 4);
    EXPECT_EQ(m.GetNumRows(), 3);
    EXPECT_EQ(m.GetNumCols(), 4);
}