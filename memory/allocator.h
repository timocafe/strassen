// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "memory/util.h"

template <class T> class cstandard : public std::allocator<T> {
public:
  typedef typename std::allocator<T>::size_type size_type;

  void *allocate_policy(size_type size) {
    void *ptr = std::malloc(size);
    return ptr;
  }

  void deallocate_policy(void *ptr) { std::free(ptr); }
};

#ifdef CUDA_STRASSEN
                                                                                          
template <class T> class cuda_unify : public std::allocator<T> {
public:
  typedef typename std::allocator<T>::size_type size_type;

  void *allocate_policy(size_type size) {
    void *ptr = nullptr;
    CUDA_CALL(cudaMallocManaged(&ptr, size));
    std::memset(ptr, 0, size);
    cudaDeviceSynchronize(); // specific jetson
    return ptr;
  }

  void deallocate_policy(void *ptr) { std::free(ptr); }
};

#endif

#ifdef CUDA_STRASSEN
template <class T> using allocator_policy = cuda_unify<T>;
#else
template <class T> using allocator_policy = cstandard<T>;
#endif