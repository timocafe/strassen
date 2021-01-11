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

  mul_matrix_gpu(m3, m1, m2);

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

TEST(AlgebraTestMatrix, CopyBlock) {

  matrix<float> m(4, 4);
  m = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.};
  // quarter top left
  matrix<float> m_qtl(2, 2);
  // quarter top right
  matrix<float> m_qtr(2, 2);
  // quarter buttom right
  matrix<float> m_qbr(2, 2);
  // quarter buttom left
  matrix<float> m_qbl(2, 2);

  copy_block(m_qtl, m, 0, 0);
  copy_block(m_qtr, m, 0, 2);
  copy_block(m_qbl, m, 2, 0);
  copy_block(m_qbr, m, 2, 2);

  ASSERT_FLOAT_EQ(m(0, 0), m_qtl(0, 0));
  ASSERT_FLOAT_EQ(m(1, 0), m_qtl(1, 0));
  ASSERT_FLOAT_EQ(m(0, 1), m_qtl(0, 1));
  ASSERT_FLOAT_EQ(m(1, 1), m_qtl(1, 1));

  ASSERT_FLOAT_EQ(m(0, 2), m_qtr(0, 0));
  ASSERT_FLOAT_EQ(m(1, 2), m_qtr(1, 0));
  ASSERT_FLOAT_EQ(m(0, 3), m_qtr(0, 1));
  ASSERT_FLOAT_EQ(m(1, 3), m_qtr(1, 1));

  ASSERT_FLOAT_EQ(m(2, 0), m_qbl(0, 0));
  ASSERT_FLOAT_EQ(m(3, 0), m_qbl(1, 0));
  ASSERT_FLOAT_EQ(m(2, 1), m_qbl(0, 1));
  ASSERT_FLOAT_EQ(m(3, 1), m_qbl(1, 1));

  ASSERT_FLOAT_EQ(m(2, 2), m_qbr(0, 0));
  ASSERT_FLOAT_EQ(m(3, 2), m_qbr(1, 0));
  ASSERT_FLOAT_EQ(m(2, 3), m_qbr(0, 1));
  ASSERT_FLOAT_EQ(m(3, 3), m_qbr(1, 1));
}

TEST(AlgebraTestMatrix, CopyMatrix) {

  matrix<float> m(4, 4);
  // quarter top left
  matrix<float> m_qtl(2, 2);
  m_qtl = {1., 2., 3., 4.};
  // quarter top right
  matrix<float> m_qtr(2, 2);
  m_qtl = {5., 6., 7., 8.};
  // quarter buttom right
  matrix<float> m_qbr(2, 2);
  m_qtl = {9., 10., 11., 12.};
  // quarter buttom left
  matrix<float> m_qbl(2, 2);
  m_qtl = {13., 14., 15., 16.};

  copy_matrix(m, m_qtl, 0, 0);
  copy_matrix(m, m_qtr, 0, 2);
  copy_matrix(m, m_qbl, 2, 0);
  copy_matrix(m, m_qbr, 2, 2);

  ASSERT_FLOAT_EQ(m(0, 0), m_qtl(0, 0));
  ASSERT_FLOAT_EQ(m(1, 0), m_qtl(1, 0));
  ASSERT_FLOAT_EQ(m(0, 1), m_qtl(0, 1));
  ASSERT_FLOAT_EQ(m(1, 1), m_qtl(1, 1));

  ASSERT_FLOAT_EQ(m(0, 2), m_qtr(0, 0));
  ASSERT_FLOAT_EQ(m(1, 2), m_qtr(1, 0));
  ASSERT_FLOAT_EQ(m(0, 3), m_qtr(0, 1));
  ASSERT_FLOAT_EQ(m(1, 3), m_qtr(1, 1));

  ASSERT_FLOAT_EQ(m(2, 0), m_qbl(0, 0));
  ASSERT_FLOAT_EQ(m(3, 0), m_qbl(1, 0));
  ASSERT_FLOAT_EQ(m(2, 1), m_qbl(0, 1));
  ASSERT_FLOAT_EQ(m(3, 1), m_qbl(1, 1));

  ASSERT_FLOAT_EQ(m(2, 2), m_qbr(0, 0));
  ASSERT_FLOAT_EQ(m(3, 2), m_qbr(1, 0));
  ASSERT_FLOAT_EQ(m(2, 3), m_qbr(0, 1));
  ASSERT_FLOAT_EQ(m(3, 3), m_qbr(1, 1));
}
