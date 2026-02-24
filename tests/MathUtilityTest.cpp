#include <gtest/gtest.h>
#include "MathUtility.h"

TEST(MathUtility, Sigmoid)
{
    EXPECT_EQ(MathUtility::sigmoid(10), 1.0 / (1.0 + std::exp(-10)));
}