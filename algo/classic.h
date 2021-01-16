// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "memory/tile_matrix.h"
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

//
// \brief strassen algo with notation wikipedia, lbs indicate the limit block
// size to stop the reccursive algo
//

template <class T>
auto classic(const tile_matrix<T> &A, const tile_matrix<T> &B,
             const uint32_t lbs = 64) {
  const uint32_t n = A.rows();
  const uint32_t k = A.rows() / 2;
  if (lbs == n) // limit blocksize
    return std::move(mul(A, B));
  // allocate nothing careful size tile = 0
  tile_matrix<T> A11(k, k, lbs);
  tile_matrix<T> A12(k, k, lbs);
  tile_matrix<T> A21(k, k, lbs);
  tile_matrix<T> A22(k, k, lbs);

  tile_matrix<T> B11(k, k, lbs);
  tile_matrix<T> B12(k, k, lbs);
  tile_matrix<T> B21(k, k, lbs);
  tile_matrix<T> B22(k, k, lbs);

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

  tile_matrix<T> M1(k, k, lbs);
  tile_matrix<T> M2(k, k, lbs);
  tile_matrix<T> M3(k, k, lbs);
  tile_matrix<T> M4(k, k, lbs);
  tile_matrix<T> M5(k, k, lbs);
  tile_matrix<T> M6(k, k, lbs);
  tile_matrix<T> M7(k, k, lbs);
  tile_matrix<T> M8(k, k, lbs);

  tbb::task_group g;

  g.run([&] { M1 = classic(A11, B11, lbs); });
  g.run([&] { M2 = classic(A11, B12, lbs); });
  g.run([&] { M3 = classic(A21, B11, lbs); });
  g.run([&] { M4 = classic(A21, B12, lbs); });

  g.run([&] { M5 = classic(A12, B21, lbs); });
  g.run([&] { M6 = classic(A12, B22, lbs); });
  g.run([&] { M7 = classic(A22, B21, lbs); });
  g.run([&] { M8 = classic(A22, B22, lbs); });

  g.wait();

  M1 += M5;
  M2 += M6;
  M3 += M7;
  M4 += M8;

  tile_matrix<T> C(n, n, lbs);

  copy_matrix(C, M1, 0, 0);
  copy_matrix(C, M2, 0, mt);
  copy_matrix(C, M3, mt, 0);
  copy_matrix(C, M4, mt, mt);

  return std::move(C);
}
