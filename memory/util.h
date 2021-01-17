// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <Eigen/Dense>
#include <atomic>

#ifdef CUDA_DEVICE

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include <atomic>

inline const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

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
      std::string s = _cudaGetErrorEnum(x);                                    \
      std::cout << s << " file " << __FILE__ << " line " << __LINE__           \
                << std::endl;                                                  \
      throw std::runtime_error("CUBLAS GAME OVER!");                           \
    }                                                                          \
  } while (0)

#endif
template <class T> void print(const T &t) { std::cout << t << std::endl; }

// army of boolean for the static and the scheduling
static std::atomic<int> gpu_ready_(0);
static std::atomic<int> nmul_gpu(0);
static std::atomic<int> nadd_gpu(0);
static std::atomic<int> nmul_cpu(0);
static std::atomic<int> nadd_cpu(0);
static bool init_add_(false);
static bool init_mul_(false);
