// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "algo/classic.h"
#include "memory/tile_matrix.h"
#include <tbb/task_group.h>

//
// \brief strassen algo with notation wikipedia, lbs indicate the limit block
// size to stop the reccursive algo
//

template <class T>
auto strassen(const tile_matrix<T> &A, const tile_matrix<T> &B,
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

  tile_matrix<T> M1(k, k, lbs, false);
  tile_matrix<T> M2(k, k, lbs, false);
  tile_matrix<T> M3(k, k, lbs, false);
  tile_matrix<T> M4(k, k, lbs, false);
  tile_matrix<T> M5(k, k, lbs, false);
  tile_matrix<T> M6(k, k, lbs, false);
  tile_matrix<T> M7(k, k, lbs, false);

  tbb::task_group g;

  g.run([&] { M1 = strassen(A11 + A22, B11 + B22, lbs); });
  g.run([&] { M2 = strassen(A21 + A22, B11, lbs); });
  g.run([&] { M3 = strassen(A11, B12 - B22, lbs); });
  g.run([&] { M4 = strassen(A22, B21 - B11, lbs); });
  g.run([&] { M5 = strassen(A11 + A12, B22, lbs); });
  g.run([&] { M6 = strassen(A21 - A11, B11 + B12, lbs); });
  g.run([&] { M7 = strassen(A12 - A22, B21 + B22, lbs); });

  g.wait();

  g.run([&] {
    tile_add_matrix(C, M1, 0, 0);
    tile_add_matrix(C, M4, 0, 0);
    tile_add_matrix(C, M7, 0, 0);
    tile_sub_matrix(C, M5, 0, 0);
  });

  g.run([&] {
    tile_add_matrix(C, M3, 0, mt);
    tile_add_matrix(C, M5, 0, mt);
  });

  g.run([&] {
    tile_add_matrix(C, M2, mt, 0);
    tile_add_matrix(C, M4, mt, 0);
  });

  g.run([&] {
    tile_add_matrix(C, M1, mt, mt);
    tile_sub_matrix(C, M2, mt, mt);
    tile_sub_matrix(C, M3, mt, mt);
    tile_sub_matrix(C, M6, mt, mt);
  });

  g.wait();

  return std::move(C);
}
