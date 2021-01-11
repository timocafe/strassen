// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "gtest/gtest.h"

#include "memory/vector.h"

TEST(ConstructorTestVector, ExplicitConstructor) {
  {
    vector<float> v;
    EXPECT_EQ(v.size(), 0);
    EXPECT_EQ(v.data(), nullptr);
  }
  {
    vector<float> v(1);
    EXPECT_EQ(v.size(), 1);
    EXPECT_NE(v.data(), nullptr);
    EXPECT_EQ(v[0], 0);
  }
  {
    vector<float> v(1, 1.f);
    EXPECT_EQ(v.size(), 1);
    EXPECT_NE(v.data(), nullptr);
    ASSERT_FLOAT_EQ(v[0], 1.f);
  }
}

TEST(ConstructorTestVector, InitializerListConstructor) {
  {
    vector<float> v = {1., 2., 3.};
    EXPECT_EQ(v.size(), 3);
    ASSERT_FLOAT_EQ(v[0], 1.f);
    ASSERT_FLOAT_EQ(v[1], 2.f);
    ASSERT_FLOAT_EQ(v[2], 3.f);
  }
}

TEST(DataMembersTestVector, FunctionVector) {
  vector<float> v = {1., 2., 3.};
  EXPECT_EQ(v.memory_allocated(), 3 * sizeof(float));
}

TEST(AlgebraTestVector, AdditionGPU) {
  {
    vector<float> v = {1., 2.};
    vector<float> w = {2., 3.};
    v += w;
    ASSERT_FLOAT_EQ(v[0], 3.f);
    ASSERT_FLOAT_EQ(v[1], 5.f);
  }
}

TEST(AlgebraTestVector, AdditionCPU) {
  {
    vector<float> v = {1., 2.};
    vector<float> w = {2., 3.};
    add_vector_cpu(v, w);
    ASSERT_FLOAT_EQ(v[0], 3.f);
    ASSERT_FLOAT_EQ(v[1], 5.f);
  }
  {
    vector<float> v = {1., 2.};
    vector<float> w = {2., 3.};
    sub_vector_cpu(v, w);
    ASSERT_FLOAT_EQ(v[0], -1.f);
    ASSERT_FLOAT_EQ(v[1], -1.f);
  }
}
