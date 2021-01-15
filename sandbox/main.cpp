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

#include <tbb/task_arena.h>
#include <tbb/task_group.h>

static const float GIGA = 1000000000;

float classical_flops(float n) { return 2 * n * n * n - n * n; }

// https://mathworld.wolfram.com/StrassenFormulas.html
// log2 !
float strassen_flops(float n) {
  return 7 * std::pow(7, std::log2(n)) - 6 * std::pow(4, std::log2(n));
}

float gflops(float a, float time) { return (a / time) / GIGA; }

int main(int argc, char *argv[]) {
  std::cout << std::fixed << std::setprecision(2);

  int size = std::atoi(argv[1]);
  int bls = std::atoi(argv[2]);

  tile_matrix<float> A(size, size, bls);
  tile_matrix<float> B(size, size, bls);
  tile_matrix<float> C(size, size, bls);

  random(A);
  random(B);

  auto AgreA = aggregate(A);
  auto AgreB = aggregate(B);
  auto AgreC = aggregate(C);

  auto start = std::chrono::system_clock::now();
  AgreC = AgreA * AgreB;
  auto end = std::chrono::system_clock::now();

  auto elapsed =
      std::chrono::duration<float, std::chrono::seconds::period>(end - start);
  std::cout << " Time to solution, " << elapsed.count() << "[s], "
            << gflops(classical_flops(size), elapsed.count())
            << ", [Flop/s], Classical " << std::endl;

  start = std::chrono::system_clock::now();
  C = strassen(A, B, bls);
  end = std::chrono::system_clock::now();
  elapsed =
      std::chrono::duration<float, std::chrono::seconds::period>(end - start);
  std::cout << " Time to solution, " << elapsed.count() << "[s], "
            << gflops(strassen_flops(size), elapsed.count())
            << ", [Flop/s], Strassen " << std::endl;

  start = std::chrono::system_clock::now();
  auto Cclassic = strassen(A, B, bls);
  end = std::chrono::system_clock::now();
  elapsed =
      std::chrono::duration<float, std::chrono::seconds::period>(end - start);
  std::cout << " Time to solution, " << elapsed.count() << "[s], "
            << gflops(classical_flops(size), elapsed.count())
            << ", [Flop/s], Classic " << std::endl;

  auto DD = aggregate(C);
  auto Dclassic = aggregate(Cclassic);

  bool b = (DD == AgreC);
  if (b)
    std::cout << " Strassen works !\n";
  else
    std::cout << " It does not works !\n";

  b = (DD == Dclassic);
  if (b)
    std::cout << " Classic works !\n";
  else
    std::cout << " It does not works !\n";

  nmul_cpu--; // case of the classical
  std::cout << " Stats CPU " << nmul_cpu << " *,  " << nadd_cpu << " + "
            << std::endl;
  std::cout << " Stats GPU " << nmul_gpu << " *,  " << nadd_gpu << " + "
            << std::endl;
}
