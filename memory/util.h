// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#define CUDA_CALL(x)                                                           \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
      throw std::runtime_error("CUDA GAME OVER!");                             \
    }                                                                          \
  } while (0)

#define CURAND_CALL(x)                                                         \
  do {                                                                         \
    if ((x) != CURAND_STATUS_SUCCESS) {                                        \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
      throw std::runtime_error("CURAND GAME OVER!");                           \
    }                                                                          \
  } while (0)

#define CUBLAS_STATUS_CALL(x)                                                  \
  do {                                                                         \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                                        \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
      throw std::runtime_error("CUBLAS GAME OVER!");                           \
    }                                                                          \
  } while (0)

template <class T> void print(const T &t) {
  cudaDeviceSynchronize();
  std::cout << t << std::endl;
}
