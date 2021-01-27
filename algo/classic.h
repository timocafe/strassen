// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "memory/tile_matrix.h"
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

// template <class T> class classic_task_t : public tbb::task {
// public:
//  const tile_matrix<T> &A_, &B_;
//  const uint32_t lbs_;
//  tile_matrix<T> &C_;
//  classic_task_t(const tile_matrix<T> &A, const tile_matrix<T> &B,
//                 tile_matrix<T> &C, const uint32_t lbs)
//      : A_(A), B_(B), C_(C), lbs_(lbs) {}
//
//  task *execute() {
//    const uint32_t n = A_.rows();
//
//    if (lbs_ == n) { // limit blocksize
//      C_ = std::move(mul(A_, B_));
//    } else {
//      const uint32_t k = A_.rows() / 2;
//
//      // allocate nothing careful size tile = 0
//      tile_matrix<T> A11(k, k, lbs_);
//      tile_matrix<T> A12(k, k, lbs_);
//      tile_matrix<T> A21(k, k, lbs_);
//      tile_matrix<T> A22(k, k, lbs_);
//
//      tile_matrix<T> B11(k, k, lbs_);
//      tile_matrix<T> B12(k, k, lbs_);
//      tile_matrix<T> B21(k, k, lbs_);
//      tile_matrix<T> B22(k, k, lbs_);
//
//      // middle tile
//      const uint32_t mt = A_.tile_rows() / 2;
//
//      copy_block(A11, A_, 0, 0);
//      copy_block(A12, A_, 0, mt);
//      copy_block(A21, A_, mt, 0);
//      copy_block(A22, A_, mt, mt);
//
//      copy_block(B11, B_, 0, 0);
//      copy_block(B12, B_, 0, mt);
//      copy_block(B21, B_, mt, 0);
//      copy_block(B22, B_, mt, mt);
//
//      tile_matrix<T> M1(k, k, lbs_);
//      tile_matrix<T> M2(k, k, lbs_);
//      tile_matrix<T> M3(k, k, lbs_);
//      tile_matrix<T> M4(k, k, lbs_);
//      tile_matrix<T> M5(k, k, lbs_);
//      tile_matrix<T> M6(k, k, lbs_);
//      tile_matrix<T> M7(k, k, lbs_);
//      tile_matrix<T> M8(k, k, lbs_);
//
//      classic_task_t &child1 =
//          *new (allocate_child()) classic_task_t(A11, B11, M1, lbs_);
//      classic_task_t &child2 =
//          *new (allocate_child()) classic_task_t(A11, B12, M2, lbs_);
//      classic_task_t &child3 =
//          *new (allocate_child()) classic_task_t(A21, B11, M3, lbs_);
//      classic_task_t &child4 =
//          *new (allocate_child()) classic_task_t(A21, B12, M4, lbs_);
//
//      classic_task_t &child5 =
//          *new (allocate_child()) classic_task_t(A12, B21, M5, lbs_);
//      classic_task_t &child6 =
//          *new (allocate_child()) classic_task_t(A12, B22, M6, lbs_);
//      classic_task_t &child7 =
//          *new (allocate_child()) classic_task_t(A22, B21, M7, lbs_);
//      classic_task_t &child8 =
//          *new (allocate_child()) classic_task_t(A22, B22, M8, lbs_);
//
//      set_ref_count(9);
//      spawn(child1);
//      spawn(child2);
//      spawn(child3);
//      spawn(child4);
//      spawn(child5);
//      spawn(child6);
//      spawn(child7);
//
//      spawn_and_wait_for_all(child8);
//
//      M1 += M5;
//      M2 += M6;
//      M3 += M7;
//      M4 += M8;
//
//      copy_matrix(C_, M1, 0, 0);
//      copy_matrix(C_, M2, 0, mt);
//      copy_matrix(C_, M3, mt, 0);
//      copy_matrix(C_, M4, mt, mt);
//    }
//    return nullptr;
//  }
//};
//
// template <class T>
// auto classic(const tile_matrix<T> &A, const tile_matrix<T> &B,
//             const uint32_t lbs = 64) {
//  const uint32_t n = A.rows();
//  // result
//  tile_matrix<T> C(n, n, lbs);
//  classic_task_t<T> &root =
//      *new (tbb::task::allocate_root()) classic_task_t<T>(A, B, C, lbs);
//  tbb::task::spawn_root_and_wait(root);
//  return std::move(C);
//}

//
// \brief classical algo with notation wikipedia, lbs indicate the limit block
// size to stop the reccursive algo
//
template <class T>
auto classic(const tile_matrix<T> &A, const tile_matrix<T> &B,
             const uint32_t lbs = 64) {
  const uint32_t n = A.rows();
  const uint32_t k = A.rows() / 2;
  if (lbs == n) // limit blocksize
    return std::move(mul(A, B));
  tile_matrix<T> C(n, n, lbs);

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

  g.run([&] {
    M1 = classic(A11, B11, lbs);
    M5 = classic(A12, B21, lbs);
    M1 += M5;
    copy_matrix(C, M1, 0, 0);
  });
  g.run([&] {
    M2 = classic(A11, B12, lbs);
    M6 = classic(A12, B22, lbs);
    M2 += M6;
    copy_matrix(C, M2, 0, mt);
  });
  g.run([&] {
    M3 = classic(A21, B11, lbs);
    M7 = classic(A22, B21, lbs);
    M3 += M7;
    copy_matrix(C, M3, mt, 0);
  });
  g.run([&] {
    M4 = classic(A21, B12, lbs);
    M8 = classic(A22, B22, lbs);
    M4 += M8;
    copy_matrix(C, M4, mt, mt);
  });

  //  g.run([&] { M5 = classic(A12, B21, lbs);});
  //  g.run([&] { M6 = classic(A12, B22, lbs); });
  //  g.run([&] { M7 = classic(A22, B21, lbs); });
  //  g.run([&] { M8 = classic(A22, B22, lbs); });

  g.wait();

  //  g.run([&] {
  //
  //  });
  //
  //  g.run([&] {
  //    M2 += M6;
  //    copy_matrix(C, M2, 0, mt);
  //  });
  //
  //  g.run([&] {
  //    M3 += M7;
  //    copy_matrix(C, M3, mt, 0);
  //  });
  //  g.run([&] {
  //    M4 += M8;
  //    copy_matrix(C, M4, mt, mt);
  //  });
  //
  //  g.wait();

  return std::move(C);
}
