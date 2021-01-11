// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "gtest/gtest.h"

#include "memory/matrix.h"
#include "memory/tile_matrix.h"

TEST(ConstructorTestTileMatrix, ExplicitConstructor) {
  {
    tile_matrix<float> m;
    EXPECT_EQ(m.size(), 0);
  }
  {
    tile_matrix<float> m(64, 64);
    EXPECT_EQ(m.size(), 1);
    EXPECT_EQ(m.cols(), 64);
    EXPECT_EQ(m.rows(), 64);
    EXPECT_EQ(m.tile_cols(), 1);
    EXPECT_EQ(m.tile_rows(), 1);
  }
  {
    tile_matrix<float> m(256, 256);
    EXPECT_EQ(m.size(), 16);
    EXPECT_EQ(m.cols(), 256);
    EXPECT_EQ(m.rows(), 256);
    EXPECT_EQ(m.tile_cols(), 4);
    EXPECT_EQ(m.tile_rows(), 4);
  }
}

TEST(bracketTestTileMatrix, MoveTile) {
  tile_matrix<float> A(4, 4, 2);
  A(0, 0) = 0;
  A(1, 0) = 1;
  A(2, 0) = 2;
  A(3, 0) = 3;
  A(0, 1) = 10;
  A(1, 1) = 11;
  A(2, 1) = 12;
  A(3, 1) = 13;
  A(0, 2) = 20;
  A(1, 2) = 21;
  A(2, 2) = 22;
  A(3, 2) = 23;
  A(0, 3) = 30;
  A(1, 3) = 31;
  A(2, 3) = 32;
  A(3, 3) = 33;

  matrix<float> B;
  auto original_pointer = A.tile(0, 0).data();
  B = std::move(A.tile(0, 0));
  // check the old tile block, should be dead
  EXPECT_EQ(A.tile(0, 0).rows(), 0);
  EXPECT_EQ(A.tile(0, 0).cols(), 0);
  EXPECT_EQ(A.tile(0, 0).data(), nullptr);
  EXPECT_EQ(B.data(), original_pointer);

  EXPECT_EQ(B.cols(), 2);
  EXPECT_EQ(B.rows(), 2);
  EXPECT_EQ(B(0, 0), 0);
  EXPECT_EQ(B(1, 0), 1);
  EXPECT_EQ(B(0, 1), 10);
  EXPECT_EQ(B(1, 1), 11);

  // reset new value is the tile matrix
  B(0, 0) = 130;
  B(1, 0) = 151;
  B(0, 1) = 171;
  B(1, 1) = 191;

  A.tile(0, 0) = std::move(B);
  EXPECT_EQ(A(0, 0), 130);
  EXPECT_EQ(A(1, 0), 151);
  EXPECT_EQ(A(0, 1), 171);
  EXPECT_EQ(A(1, 1), 191);
  EXPECT_EQ(A.tile(0, 0).data(), original_pointer);
}

TEST(AlgebraTestTileMatrix, MinusTileMatrix) {
  tile_matrix<float> A(4, 4, 2);
  A(0, 0) = 0;
  A(1, 0) = 1;
  A(2, 0) = 2;
  A(3, 0) = 3;
  A(0, 1) = 10;
  A(1, 1) = 11;
  A(2, 1) = 12;
  A(3, 1) = 13;
  A(0, 2) = 20;
  A(1, 2) = 21;
  A(2, 2) = 22;
  A(3, 2) = 23;
  A(0, 3) = 30;
  A(1, 3) = 31;
  A(2, 3) = 32;
  A(3, 3) = 33;

  tile_matrix<float> B(4, 4, 2);
  B(0, 0) = 1;
  B(1, 0) = 1;
  B(2, 0) = 1;
  B(3, 0) = 1;
  B(0, 1) = 2;
  B(1, 1) = 2;
  B(2, 1) = 2;
  B(3, 1) = 2;
  B(0, 2) = 3;
  B(1, 2) = 3;
  B(2, 2) = 3;
  B(3, 2) = 3;
  B(0, 3) = 4;
  B(1, 3) = 4;
  B(2, 3) = 4;
  B(3, 3) = 4;

  A -= B;

  EXPECT_EQ(A(0, 0), -1);
  EXPECT_EQ(A(1, 0), 0);
  EXPECT_EQ(A(2, 0), 1);
  EXPECT_EQ(A(3, 0), 2);
  EXPECT_EQ(A(0, 1), 8);
  EXPECT_EQ(A(1, 1), 9);
  EXPECT_EQ(A(2, 1), 10);
  EXPECT_EQ(A(3, 1), 11);
  EXPECT_EQ(A(0, 2), 17);
  EXPECT_EQ(A(1, 2), 18);
  EXPECT_EQ(A(2, 2), 19);
  EXPECT_EQ(A(3, 2), 20);
  EXPECT_EQ(A(0, 3), 26);
  EXPECT_EQ(A(1, 3), 27);
  EXPECT_EQ(A(2, 3), 28);
  EXPECT_EQ(A(3, 3), 29);
}

TEST(AlgebraTestTileMatrix, AddTileMatrix) {
  tile_matrix<float> A(4, 4, 2);
  A(0, 0) = 0;
  A(1, 0) = 1;
  A(2, 0) = 2;
  A(3, 0) = 3;
  A(0, 1) = 10;
  A(1, 1) = 11;
  A(2, 1) = 12;
  A(3, 1) = 13;
  A(0, 2) = 20;
  A(1, 2) = 21;
  A(2, 2) = 22;
  A(3, 2) = 23;
  A(0, 3) = 30;
  A(1, 3) = 31;
  A(2, 3) = 32;
  A(3, 3) = 33;

  tile_matrix<float> B(4, 4, 2);

  B(0, 0) = 1;
  B(1, 0) = 1;
  B(2, 0) = 1;
  B(3, 0) = 1;
  B(0, 1) = 2;
  B(1, 1) = 2;
  B(2, 1) = 2;
  B(3, 1) = 2;
  B(0, 2) = 3;
  B(1, 2) = 3;
  B(2, 2) = 3;
  B(3, 2) = 3;
  B(0, 3) = 4;
  B(1, 3) = 4;
  B(2, 3) = 4;
  B(3, 3) = 4;

  A += B;

  EXPECT_EQ(A(0, 0), 1);
  EXPECT_EQ(A(1, 0), 2);
  EXPECT_EQ(A(2, 0), 3);
  EXPECT_EQ(A(3, 0), 4);
  EXPECT_EQ(A(0, 1), 12);
  EXPECT_EQ(A(1, 1), 13);
  EXPECT_EQ(A(2, 1), 14);
  EXPECT_EQ(A(3, 1), 15);
  EXPECT_EQ(A(0, 2), 23);
  EXPECT_EQ(A(1, 2), 24);
  EXPECT_EQ(A(2, 2), 25);
  EXPECT_EQ(A(3, 2), 26);
  EXPECT_EQ(A(0, 3), 34);
  EXPECT_EQ(A(1, 3), 35);
  EXPECT_EQ(A(2, 3), 36);
  EXPECT_EQ(A(3, 3), 37);
}

TEST(CopyBlockAndMatrix, CopyBlock) {
  tile_matrix<float> A(4, 4, 2);
  A(0, 0) = 0;
  A(1, 0) = 1;
  A(2, 0) = 2;
  A(3, 0) = 3;
  A(0, 1) = 10;
  A(1, 1) = 11;
  A(2, 1) = 12;
  A(3, 1) = 13;
  A(0, 2) = 20;
  A(1, 2) = 21;
  A(2, 2) = 22;
  A(3, 2) = 23;
  A(0, 3) = 30;
  A(1, 3) = 31;
  A(2, 3) = 32;
  A(3, 3) = 33;

  // 0 no init so no allocation
  tile_matrix<float> B(2, 2, 2, 0);

  copy_block(B, A, 1, 1);
  // A is dead now
  EXPECT_EQ(B(0, 0), 22);
  EXPECT_EQ(B(1, 0), 23);
  EXPECT_EQ(B(0, 1), 32);
  EXPECT_EQ(B(1, 1), 33);
}

TEST(CopyBlockAndMatrix, CopyMatrix) {
  tile_matrix<float> A(2, 2, 2);
  A(0, 0) = 0;
  A(1, 0) = 1;
  A(0, 1) = 2;
  A(1, 1) = 3;

  // 0 no init so no allocation
  tile_matrix<float> B(4, 4, 2, 0);

  copy_matrix(B, A, 0, 0);
  // A is dead now
  EXPECT_EQ(B(0, 0), 0);
  EXPECT_EQ(B(1, 0), 1);
  EXPECT_EQ(B(0, 1), 2);
  EXPECT_EQ(B(1, 1), 3);
}
