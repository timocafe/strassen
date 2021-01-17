// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "gtest/gtest.h"

#include "memory/matrix.h"

TEST(ConstructorTestMatrix, ExplicitConstructor) {
  {
    matrix<float> m;
    EXPECT_EQ(m.size(), 0);
    EXPECT_EQ(m.data(), nullptr);
  }
  {
    matrix<float> m(1, 1);
    EXPECT_EQ(m.size(), 1);
    EXPECT_NE(m.data(), nullptr);
    EXPECT_EQ(m(0, 0), 0);
  }
  {
    matrix<float> m(2, 2);
    EXPECT_EQ(m.size(), 4);
    EXPECT_NE(m.data(), nullptr);
    ASSERT_FLOAT_EQ(m(1, 1), 0.f);
  }
}

TEST(ConstructorTestMatrix, InitializerListConstructor) {
  {
    matrix<float> m(2, 2);
    m = {1., 2., 3., 4.};
    EXPECT_EQ(m.size(), 4);
    ASSERT_FLOAT_EQ(m(0, 0), 1.f);
    ASSERT_FLOAT_EQ(m(1, 0), 2.f);
    ASSERT_FLOAT_EQ(m(0, 1), 3.f);
    ASSERT_FLOAT_EQ(m(1, 1), 4.f);
  }
}

TEST(DataMembersTestMatrix, FunctionMatrix) {
  matrix<float> m(2, 2);
  m = {1., 2., 3., 4.};
  EXPECT_EQ(m.memory_allocated(), 4 * sizeof(float));
}

TEST(AlgebraTestMatrix, Addition) {
  matrix<float> m1(2, 2);
  m1 = {1., 2., 3., 4.};

  matrix<float> m2(2, 2);
  m2 = {2., 3., 4., 5.};

  m1 += m2;

  ASSERT_FLOAT_EQ(m1(0, 0), 3.f);
  ASSERT_FLOAT_EQ(m1(1, 0), 5.f);
  ASSERT_FLOAT_EQ(m1(0, 1), 7.f);
  ASSERT_FLOAT_EQ(m1(1, 1), 9.f);
}

TEST(AlgebraTestMatrix, MultiplicationGPU) {
  matrix<float> m1(2, 2);
  m1 = {1., 2., 3., 4.};

  matrix<float> m2(2, 2);
  m2 = {2., 3., 4., 5.};

  matrix<float> m3(2, 2);

  m3 = m1 * m2;

  ASSERT_FLOAT_EQ(m3(0, 0), 11.f);
  ASSERT_FLOAT_EQ(m3(1, 0), 16.f);
  ASSERT_FLOAT_EQ(m3(0, 1), 19.f);
  ASSERT_FLOAT_EQ(m3(1, 1), 28.f);
}

TEST(AlgebraTestMatrix, MultiplicationCPU) {
  matrix<float> m1(2, 2);
  m1 = {1., 2., 3., 4.};

  matrix<float> m2(2, 2);
  m2 = {2., 3., 4., 5.};

  matrix<float> m3(2, 2);

  mul_matrix_cpu(m3, m1, m2);

  ASSERT_FLOAT_EQ(m3(0, 0), 11.f);
  ASSERT_FLOAT_EQ(m3(1, 0), 16.f);
  ASSERT_FLOAT_EQ(m3(0, 1), 19.f);
  ASSERT_FLOAT_EQ(m3(1, 1), 28.f);
}
