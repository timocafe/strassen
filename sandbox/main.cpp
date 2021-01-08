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
  matrix<float> m1(2, 2);
  m1 = {1., 2., 3., 4.};

  matrix<float> m2(2, 2);
  m2 = {2., 3., 4., 5.};

  matrix<float> m3(2, 2);
  m3 = m1 * m2;
  // mul_matrix_cpu(m3, m1, m2);
}
