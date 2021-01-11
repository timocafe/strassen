// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cassert>
#include <iostream>

#include "memory/util.h"

// policy for the specific copy constructor
struct policy_shared {};

// using eigen_matrix_type =
//    Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic,
//    Eigen::ColMajor>;

///
/// \brief vector can be used on CPU and GPU.
/// and stl.
///        For GPU the vector must be first allocated on CPU using the good
///        memory policy. The Cuda kernel should be called using the following
///        syntax
///        my_kernel<<<blocks,threads>>>({policy_shared(),my_vector});
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
#ifdef CUDA_STRASSEN
      CUDA_CALL(cudaMallocManaged(&data_, n * sizeof(value_type)));
      cudaDeviceSynchronize(); // specific jetson
#else
      data_ = new value_type[n];
#endif
      std::fill(begin(), end(), v);
    }
  }

  ///
  /// \brief initializer constructor {1.,2. ...} col order !
  ///
  vector(std::initializer_list<value_type> l) {
    int n = l.size();
    size_ = n;
    shared_ = false;
    if (n > 0) {
#ifdef CUDA_STRASSEN
      CUDA_CALL(cudaMallocManaged(&data_, n * sizeof(value_type)));
      cudaDeviceSynchronize(); // specific jetson
#else
      data_ = new value_type[n];
#endif
    }
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
  vector(const vector &other)
      : data_(nullptr), size_(other.size_), shared_(false) {
    if (other.data_ != nullptr) {
#ifdef CUDA_STRASSEN
      CUDA_CALL(cudaMallocManaged(&data_, size_ * sizeof(value_type)));
      cudaDeviceSynchronize(); // specific jetson
#else
      data_ = new value_type[size_];
#endif
      std::copy(other.begin(), other.end(), begin());
    }
  }

  ///
  /// \brief move constructor
  ///
  vector(vector &&other)
      : data_(std::move(other.data_)), size_(std::move(other.size_)),
        shared_(std::move(other.shared_)) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.shared_ = false;
  }

  ///
  /// \brief assign move operator
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
  /// \brief assigm move operator
  /// GPU,
  ///        using unify memory the GPU will be able to reach the memory
  ///
  vector &operator=(const vector &other) {
    vector v(other);
    *this = std::move(v);
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

#ifdef CUDA_STRASSEN
        cudaFree(data_);
#else
        delete[] data_;
#endif
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
#ifdef CUDA_STRASSEN
    if (gpu_ready_.fetch_sub(1))
      add_vector_gpu(*this, v);
    else
#endif
      add_vector_cpu(*this, v);
    return *this;
  }

  ///
  /// \brief substraction between two vectors
  ///
  vector &operator-=(const vector &v) {
    assert(size() == v.size() &&
           " can not make substraction between vector of different size ");
#ifdef CUDA_STRASSEN
    if (gpu_ready_.fetch_sub(1))
      sub_vector_gpu(*this, v);
    else
#endif
      sub_vector_cpu(*this, v);
    return *this;
  }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  iterator begin() { return data_; }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  const_iterator end() const { return data_ + size_; }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  const_iterator begin() const { return data_; }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  iterator end() { return data_ + size_; }

  ///
  /// \brief Return the size of the vector
  ///
  inline size_type size() const { return size_; }

  ///
  /// \brief Return a reference of the data using usual bracket operator syntax
  ///
  inline reference operator[](size_type i) {
    assert(i < size_ && " Too ambitious! \n");
    return data_[i];
  }

  ///
  /// \brief Return a cosnt reference of the data using usual bracket operator
  /// syntax
  ///
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
  /// \brief Return the data pointer need for copy block/matrix
  ///
  pointer data() { return data_; }

  ///
  /// \brief set up the pointer directly need for copy block/matrix
  ///
  void data(pointer p) { data_ = p; }

  ///
  /// \brief Return the memory allocated
  ///
  size_type memory_allocated() const { return sizeof(T) * size_; }
#ifdef CUDA_STRASSEN
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
#endif
private:
  size_type size_; // size of the vector
  pointer data_;   // pointer of the data
  bool shared_;    // only use for GPU to provide arument to cuda kernel
};
#ifdef CUDA_STRASSEN
///
/// \brief Addition Substraction between two vectors on GPU
///
template <class T>
void helper_add(vector<T> &v_y, const vector<T> &v_x, const float a) {
  cublasHandle_t handle;
  CUBLAS_STATUS_CALL(cublasCreate(&handle));
  int n = v_y.size();
  const float alpha = a;
  int incx(1);
  int incy(1);
  const float *x = v_x.data();
  float *y = v_y.data();
  // y = alpha * x + y
  cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
  CUBLAS_STATUS_CALL(cublasDestroy(handle));
  gpu_ready_.fetch_add(1);
}

///
/// \brief Addition between two vectors on GPU
///
template <class T>
inline void add_vector_gpu(vector<T> &v_y, const vector<T> &v_x) {
  helper_add(v_y, v_x, 1.f);
}

///
/// \brief Substraction between two vectors on GPU
///
template <class T>
inline void sub_vector_gpu(vector<T> &v_y, const vector<T> &v_x) {
  helper_add(v_y, v_x, -1.f);
}
#endif
///
/// \brief Addition between two vectors on CPU
///
template <class T>
inline void add_vector_cpu(vector<T> &v_y, const vector<T> &v_x) {
  using eigen_vector_type =
      Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  Eigen::Map<eigen_vector_type>(v_y.data(), v_y.size()) +=
      Eigen::Map<eigen_vector_type>(v_x.data(), v_x.size());
}

///
/// \brief Substraction between two vectors on CPU
///
template <class T>
inline void sub_vector_cpu(vector<T> &v_y, const vector<T> &v_x) {
  using eigen_vector_type =
      Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  Eigen::Map<eigen_vector_type>(v_y.data(), v_y.size()) -=
      Eigen::Map<eigen_vector_type>(v_x.data(), v_x.size());
}

///
/// \brief Overload << stream operator
///
template <class T>
std::ostream &operator<<(std::ostream &out, const vector<T> &b) {
  b.print(out);
  return out;
}
