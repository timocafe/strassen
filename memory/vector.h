// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cuda.h>

#include <cassert>
#include <iostream>

#include "memory/util.h"

// policy for the specific copy constructor
struct policy_shared {};

#define DEVICE_CALLABLE __host__ __device__

///
/// \brief vector can be used on CPU and GPU.
/// and stl.
///        For GPU the vector must be first allocated on CPU using the good
///        memory policy. The Cuda kernel should be called using the following
///        syntax
///        my_kernel<<<blocks,threads>>>({policy_shared(),my_vector});
/// \note  DEVICE_CALLABLE indicate the functions is usuable on the GPU and CPU
///
template <class T> class vector {

public:
  typedef uint32_t size_type;
  typedef T value_type;
  typedef value_type *pointer;
  typedef pointer iterator;
  typedef const pointer const_iterator;
  typedef value_type &reference;
  typedef const value_type &const_reference;

  ///
  /// \brief usual constructor
  /// The device first divice is selected only one the memory is initialized
  ///
  explicit vector(size_type n = 0, const value_type &v = value_type())
      : data_(nullptr), size_(n), shared_(false) {
    if (n > 0) {
      CUDA_CALL(cudaMallocManaged(&data_, n * sizeof(value_type)));
      cudaDeviceSynchronize(); // specific jetson
      std::fill(begin(), end(), v);
    }
  }

  ///
  /// \brief initializer constructor {1.,2. ...} col order !
  ///
  vector(std::initializer_list<value_type> l) {
    *this = std::move(vector(l.size(), 0));
    std::copy(l.begin(), l.end(), data_);
  }

  ///
  /// \brief Specific copy constructor for the GPU. Provide the value to the
  /// GPU, using unify memory the GPU will be able to reach the memory
  ///
  vector(const policy_shared &p, const vector &other) {
    data_ = other.data_;
    size_ = other.size_;
    shared_ = true;
  }

  ///
  /// \brief copy constructor
  ///
  vector(const vector &other) : size_(other.size_), shared_(false) {
    CUDA_CALL(cudaMallocManaged(&data_, size_ * sizeof(value_type)));
    cudaDeviceSynchronize(); // specific jetson
    std::copy(other.begin(), other.end(), begin());
  }

  ///
  /// \brief move constructor
  ///
  vector(vector &&other)
      : data_(std::move(other.data_)), size_(std::move(other.size_)),
        shared_(std::move(other.shared_)) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.shared_ = true;
  }

  ///
  /// \brief assigm move operator
  /// GPU,
  ///        using unify memory the GPU will be able to reach the memory
  ///
  vector &operator=(vector &&other) {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(shared_, other.shared_);
    return *this;
  }

  ///
  /// \brief Destructor, the destruction only append if the value of the pointer
  /// is not null
  ///        (due to the move constructor) and not shared (because the
  ///        destruction on the GPU is impossible (shared = false))
  ///
  ~vector() {
    if (!shared_) {
      if (data_ != nullptr) {
        cudaFree(data_);
        size_ = 0;
        data_ = nullptr;
      }
    }
  }

  ///
  /// \brief Addition between two vectors
  ///
  vector &operator+=(const vector &v) {
    assert(size() == v.size() &&
           " can not make addition between vector of different size ");
    cublasHandle_t handle;
    CUBLAS_STATUS_CALL(cublasCreate(&handle));
    int n = size();
    const float alpha = 1.f;
    int incx(1);
    int incy(1);

    const float *x = v.data();
    float *y = data();
    // y = alpha * x + y
    cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
    CUBLAS_STATUS_CALL(cublasDestroy(handle));
    return *this;
  }

  ///
  /// \brief substraction between two vectors
  ///
  vector &operator-=(const vector &v) {
    assert(size() == v.size() &&
           " can not make substraction between vector of different size ");
    cublasHandle_t handle;
    CUBLAS_STATUS_CALL(cublasCreate(&handle));
    int n = size();
    const float alpha = -1.f;
    const float *x = v.data();
    float *y = data();
    int incx(1);
    int incy(1);
    // y = alpha * x + y
    cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
    CUBLAS_STATUS_CALL(cublasDestroy(handle));
    return *this;
  }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  DEVICE_CALLABLE
  iterator begin() { return data_; }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  DEVICE_CALLABLE
  const_iterator end() const { return data_ + size_; }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  DEVICE_CALLABLE
  const_iterator begin() const { return data_; }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  DEVICE_CALLABLE
  iterator end() { return data_ + size_; }

  ///
  /// \brief Return the size of the vector
  ///
  DEVICE_CALLABLE
  inline size_type size() const { return size_; }

  ///
  /// \brief Return a reference of the data using usual bracket operator syntax
  ///
  DEVICE_CALLABLE
  inline reference operator[](size_type i) {
    assert(i < size_ && " Too ambitious! \n");
    return data_[i];
  }

  ///
  /// \brief Return a cosnt reference of the data using usual bracket operator
  /// syntax
  ///
  DEVICE_CALLABLE
  inline const_reference operator[](size_type i) const {
    assert(i < size_ && " Too ambitious! \n");
    return data_[i];
  }

  ///
  /// \brief Print the vector
  ///
  void print(std::ostream &out) const {
    for (int i = 0; i < size(); ++i)
      out << (*this)[i] << " ";
    out << "\n";
  }

  ///
  /// \brief Return the data pointer
  ///
  pointer data() const { return data_; }

  ///
  /// \brief Return the memory allocated
  ///
  size_type memory_allocated() const { return sizeof(T) * size_; }

  ///
  //
  /// \brief Prefetch data on the gpu
  ///
  void prefetch_gpu(cudaStream_t s = 0) const {
    cudaMemPrefetchAsync(data(), memory_allocated(), 0,
                         s); // 0 is the default device
  }

  ///
  /// \brief Prefetch data on the cpu
  ///
  void prefetch_cpu(cudaStream_t s = 0) const {
    cudaMemPrefetchAsync(data(), memory_allocated(), cudaCpuDeviceId, s);
  }

private:
  size_type size_; // size of the vector
  pointer data_;   // pointer of the data
  bool shared_;    // only use for GPU to provide arument to cuda kernel
};

///
/// \brief Overload << stream operator
///
template <class T>
std::ostream &operator<<(std::ostream &out, const vector<T> &b) {
  b.print(out);
  return out;
}
