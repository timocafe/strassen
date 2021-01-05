// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <chrono>
#include <iostream>

#include "memory/matrix.h"
#include "memory/vector.h"

int main(int argc, char *argv[]) {
  typedef float value_type;
  matrix<value_type> m(3, 4);
  random(m);
  cudaDeviceSynchronize();
  std::cout << m << std::endl;
}
