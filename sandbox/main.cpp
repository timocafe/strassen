// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <chrono>
#include <iostream>

#include "algo/strassen.h"
#include "memory/matrix.h"
#include "memory/vector.h"

int main(int argc, char *argv[]) {
  int size = std::atoi(argv[1]);
  int bls = std::atoi(argv[2]);
  matrix<float> A(size, size);
  matrix<float> B(size, size);
  matrix<float> C(size, size);
  random(A);
  random(B);
 /* 
  {
    auto start = std::chrono::system_clock::now();
    C = A * B;
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<float, std::chrono::seconds::period>(end - start);
    std::cout << " time to solution Classical: " << elapsed.count()
              << std::endl;
  }
*/
  {
    auto start = std::chrono::system_clock::now();
    C = strassen(A, B, bls);
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<float, std::chrono::seconds::period>(end - start);
    std::cout << " time to solution Strassen: " << elapsed.count() << std::endl;
  }
 
}
