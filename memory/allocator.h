// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "memory/util.h"

class cstandard {
public:
  typedef uint32_t size_type;

  void *allocate_policy(size_type size) {
    void *ptr = std::malloc(size);
    return ptr;
  }

  void deallocate_policy(void *ptr) { std::free(ptr); }
};

#ifdef CUDA_STRASSEN

class cuda_unify {
public:
  typedef uint32_t size_type;

  void *allocate_policy(size_type size) {
    void *ptr = nullptr;
    CUDA_CALL(cudaMallocManaged(&ptr, size));
    std::memset(ptr, 0, size);
    cudaDeviceSynchronize(); // specific jetson
    return ptr;
  }

  void deallocate_policy(void *ptr) { cudaFree(ptr); }
};

#endif

#ifdef CUDA_STRASSEN
typedef cuda_unify allocator_policy;
#else
typedef cstandard allocator_policy;
#endif
