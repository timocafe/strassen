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
  /*
    matrix<value_type> A(2, 3);
    A = {1., 2., 3., 4., 5., 6.};
    matrix<value_type> B(3, 4);
    B = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    matrix<value_type> C = A * B;
    print(A);
    print(B);
    print(C);
  */

  matrix<value_type> A(2, 3);
  A = {1., 2., 3., 4., 5., 6.};
  matrix<value_type> B(2, 3);
  B = {1., 1., 1., 1., 1., 1.};

  print(A - B);
}
