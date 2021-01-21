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

float strassen_flops(float n) {
  return 7 * std::pow(7, std::log2(n)) - 6 * std::pow(4, std::log2(n));
}

float gflops(float a, float time) { return (a / time) / GIGA; }

int main(int argc, char *argv[]) {
  int j = std::atoi(argv[1]); // compute the size with 2^j
  std::cout << std::fixed << std::setprecision(2);
  std::vector<std::pair<float, float>> v_cpu;
  std::vector<std::pair<float, float>> v_gpu;

  int size = std::pow(2, j);
  matrix<float> A(size, size);
  matrix<float> B(size, size);
  matrix<float> C(size, size);

  random(A);
  random(B);

  {
    auto start = std::chrono::system_clock::now();
    mul_matrix_cpu(C, A, B);
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<float, std::chrono::seconds::period>(end - start);
    v_cpu.push_back(std::pair<float, float>(
        elapsed.count(), gflops(classical_flops(size), elapsed.count())));
  }

  {
    auto start = std::chrono::system_clock::now();
    mul_matrix_gpu(C, A, B);
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<float, std::chrono::seconds::period>(end - start);
    v_gpu.push_back(std::pair<float, float>(
        elapsed.count(), gflops(classical_flops(size), elapsed.count())));
  }

  int size2 = v_cpu.size();

  for (int i = 0; i < size2; ++i)
    std::cout << size << ", " << v_cpu[i].first << ", " << v_cpu[i].second
              << ", " << v_gpu[i].first << ", " << v_gpu[i].second << "\n";
}
