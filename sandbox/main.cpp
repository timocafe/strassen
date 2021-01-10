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
#include "memory/tile_matrix.h"
#include "memory/vector.h"

int main(int argc, char *argv[]) {
  int size = std::atoi(argv[1]);
  int bls = std::atoi(argv[2]);
  tile_matrix<float> A(size, size, bls);
  // random(A);
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
  print(A);

  /*
  matrix<float> A(size, size);
  matrix<float> B(size, size);
   {
     auto start = std::chrono::system_clock::now();
     C = A * B;
     auto end = std::chrono::system_clock::now();
     auto elapsed =
         std::chrono::duration<float, std::chrono::seconds::period>(end -
   start); std::cout << " time to solution Classical: " << elapsed.count()
               << std::endl;
   }
  {
    auto start = std::chrono::system_clock::now();
    C = strassen(A, B, bls);
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<float, std::chrono::seconds::period>(end - start);
    std::cout << " time to solution Strassen: " << elapsed.count() << std::endl;
  }
 */
}
