// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "memory/util.h"

template <typename T> struct cstandard {
  using value_type = T;

  cstandard() = default;
  template <class U> cstandard(const cstandard<U> &) {}

  T *allocate(std::size_t size) {
    return static_cast<T *>(std::malloc(size * sizeof(value_type)));
  }
  void deallocate(T *ptr, std::size_t n) { std::free(ptr); }
};

template <typename T, typename U>
inline bool operator==(const cstandard<T> &, const cstandard<U> &) {
  return true;
}

template <typename T, typename U>
inline bool operator!=(const cstandard<T> &a, const cstandard<U> &b) {
  return !(a == b);
}

#ifdef CUDA_STRASSEN

template <typename T> struct cuda_unify {
  using value_type = T;

  cuda_unify() = default;
  template <class U> cuda_unify(const cuda_unify<U> &) {}

  T *allocate(std::size_t size) {
    void *ptr = nullptr;
    CUDA_CALL(cudaMallocManaged(&ptr, size * sizeof(value_type)));
    cudaDeviceSynchronize(); // specific jetson
    return static_cast<T *>(ptr);
  }
  void deallocate(T *ptr, std::size_t n) { cudaFree(ptr); }
};

template <typename T, typename U>
inline bool operator==(const cuda_unify<T> &, const cuda_unify<U> &) {
  return true;
}

template <typename T, typename U>
inline bool operator!=(const cuda_unify<T> &a, const cuda_unify<U> &b) {
  return !(a == b);
}
#endif

#ifdef CUDA_STRASSEN
template <class T> using allocator_policy = cuda_unify<T>;
#else
template <class T> using allocator_policy = cstandard<T>;
#endif
