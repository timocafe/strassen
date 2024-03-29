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

#include <tbb/global_control.h>

static const float GIGA = 1000000000;

float classical_flops(float n) { return 2 * n * n * n - n * n; }

// https://mathworld.wolfram.com/StrassenFormulas.html
// log2 !
float strassen_flops(float n) {
  return 7 * std::pow(7, std::log2(n)) - 6 * std::pow(4, std::log2(n));
}

float gflops(float a, float time) { return (a / time) / GIGA; }

void reset_counter() {
  nmul_cpu = 0;
  nadd_cpu = 0;
  nfma_cpu = 0;
  nmul_gpu = 0;
  nadd_gpu = 0;
  nfma_gpu = 0;
  nfma_gpu = 0;
}

void print_counter(std::string algo) {
  std::cout << " Algorithm sgemm: " << algo << std::endl;
  std::cout << " Stats CPU " << nmul_cpu << " *,  " << nadd_cpu << " +, "
            << nfma_cpu << " fma " << std::endl;
  std::cout << " Stats GPU " << nmul_gpu << " *,  " << nadd_gpu << " +, "
            << nfma_gpu << " fma " << std::endl;
}

int main(int argc, char *argv[]) {
  std::cout << std::fixed << std::setprecision(2);
  // size of the matrix
  int size = std::atoi(argv[1]);
  // size of the tile
  int bls = std::atoi(argv[2]);
  // number of thread per task
  int thread = std::atoi(argv[3]);
  tbb::global_control c(tbb::global_control::max_allowed_parallelism, thread);

  tile_matrix<float> A(size, size, bls);
  tile_matrix<float> B(size, size, bls);
  tile_matrix<float> C(size, size, bls);

  random(A);
  random(B);
  
    auto AgreA = aggregate(A);
    auto AgreB = aggregate(B);
    auto AgreC = aggregate(C);

    AgreC = AgreA * AgreB;
  
  auto start = std::chrono::system_clock::now();
  C = strassen(A, B, bls);
  auto end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration<float, std::milli>(end - start);
  std::cout << " Time to solution, " << elapsed.count() << "[ms], "
            << gflops(strassen_flops(size), elapsed.count() / 1000)
            << ", [Flop/s], Strassen " << std::endl;

  print_counter("Strassen");
  reset_counter();

  start = std::chrono::system_clock::now();
  auto Cclassic = classic(A, B);
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration<float, std::milli>(end - start);
  std::cout << " Time to solution, " << elapsed.count() << "[ms], "
            << gflops(classical_flops(size), elapsed.count() / 1000)
            << ", [Flop/s], Classic " << std::endl;

  print_counter("Classic");
  reset_counter();


  start = std::chrono::system_clock::now();
  auto Crclassic = rclassic(A, B,bls);
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration<float, std::milli>(end - start);
  std::cout << " Time to solution, " << elapsed.count() << "[ms], "
            << gflops(classical_flops(size), elapsed.count() / 1000)
            << ", [Flop/s], reccursive Classic " << std::endl;

  print_counter("reccursive Classic");
  reset_counter();


  
    auto DD = aggregate(C);
    auto Dclassic = aggregate(Cclassic);
    auto Drclassic = aggregate(Crclassic);


    bool b = (DD == AgreC);
    if (b)
      std::cout << " Strassen works !\n";
    else
      std::cout << " Strassen does not works !\n";

    b = (Dclassic == AgreC);
    if (b)
      std::cout << " Classic works !\n";
    else
      std::cout << " Classic does not works !\n";

    b = (Drclassic == AgreC);
    if (b)
      std::cout << " Recursive Classic works !\n";
    else
      std::cout << " Recursive Classic does not works !\n";


  // std::cout << C << std::endl;
  // std::cout << Dclassic << std::endl;
  // std::cout << AgreC << std::endl;
}
