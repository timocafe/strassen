// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "algo/classic.h"
#include "algo/strassen.h"

#include "memory/tile_matrix.h"

static const float GIGA = 1000000000;

float classical_flops(float n) { return 2 * n * n * n - n * n; }

float gflops(float a, float time) { return (a / time) / GIGA; }

int main(int argc, char *argv[]) {
  std::cout << std::fixed << std::setprecision(2);

  const int size = std::atoi(argv[1]);

  matrix<float> A(size, size);
  matrix<float> B(size, size);
  matrix<float> C(size, size);

  random(A);
  random(B);

  auto start = std::chrono::system_clock::now();
  C = B * A;
  auto end = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration<float, std::chrono::seconds::period>(end - start);
  std::cout << " Time to solution, " << elapsed.count() << "[s], "
            << gflops(classical_flops(size), elapsed.count())
            << ", [GFlop/s], eigen sgemm multi thread" << std::endl;
}
