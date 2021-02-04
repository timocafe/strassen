// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "memory/tile_matrix.h"
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

////
//// \brief classical algo with notation wikipedia,
////
template <class T>
auto classic(const tile_matrix<T> &A, const tile_matrix<T> &B) {
  const uint32_t n = A.rows();
  const uint32_t lbs = A.tile();
  tile_matrix<T> C(n, n, lbs);
  parallel_for(tbb::blocked_range<size_t>(0, A.tile_rows()),
               [&](const tbb::blocked_range<size_t> &r) {
                 for (size_t i = r.begin(); i != r.end(); ++i)
                   for (int j = 0; j < B.tile_cols(); ++j)
                     for (int k = 0; k < A.tile_cols(); ++k)
                       fma(A.tile(i, k), B.tile(k, j), C.tile(i, j));
               });
  return std::move(C);
}

////
//// \brief classical algo with notation wikipedia, lbs indicate the limit block
//// size to stop the reccursive algo
////
template <class T>
auto rclassic(const tile_matrix<T> &A, const tile_matrix<T> &B,
              const uint32_t lbs = 64) {
  const uint32_t n = A.rows();
  const uint32_t k = A.rows() / 2;
  if (lbs == n) // limit blocksize
    return std::move(mul(A, B));
  tile_matrix<T> C(n, n, lbs);

  // allocate nothing careful size tile = 0
  tile_matrix<T> A11(k, k, lbs, false);
  tile_matrix<T> A12(k, k, lbs, false);
  tile_matrix<T> A21(k, k, lbs, false);
  tile_matrix<T> A22(k, k, lbs, false);

  tile_matrix<T> B11(k, k, lbs, false);
  tile_matrix<T> B12(k, k, lbs, false);
  tile_matrix<T> B21(k, k, lbs, false);
  tile_matrix<T> B22(k, k, lbs, false);

  // middle tile
  const uint32_t mt = A.tile_rows() / 2;

  copy_block(A11, A, 0, 0);
  copy_block(A12, A, 0, mt);
  copy_block(A21, A, mt, 0);
  copy_block(A22, A, mt, mt);

  copy_block(B11, B, 0, 0);
  copy_block(B12, B, 0, mt);
  copy_block(B21, B, mt, 0);
  copy_block(B22, B, mt, mt);

  tbb::task_group g;

  g.run([&] {
    tile_add_matrix(C, classic(A11, B11, lbs), 0, 0);
    tile_add_matrix(C, classic(A12, B21, lbs), 0, 0);
  });
  g.run([&] {
    tile_add_matrix(C, classic(A11, B12, lbs), 0, mt);
    tile_add_matrix(C, classic(A12, B22, lbs), 0, mt);
  });
  g.run([&] {
    tile_add_matrix(C, classic(A21, B11, lbs), mt, 0);
    tile_add_matrix(C, classic(A22, B21, lbs), mt, 0);
  });
  g.run([&] {
    tile_add_matrix(C, classic(A21, B12, lbs), mt, mt);
    tile_add_matrix(C, classic(A22, B22, lbs), mt, mt);
  });

  g.wait();

  return std::move(C);
}
