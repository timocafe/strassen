// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cassert>
#include <iostream>

#include "memory/util.h"

#include "memory/allocator.h"

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
template <class T, class Allocator = allocator_policy<T>> class vector {
  using size_type = typename std::vector<T, Allocator>::size_type;
  using reference = typename std::vector<T, Allocator>::reference;
  using const_reference = typename std::vector<T, Allocator>::const_reference;
  using const_pointer = typename std::vector<T, Allocator>::const_pointer;
  using pointer = typename std::vector<T, Allocator>::pointer;
  using value_type = T;

public:
  ///
  /// \brief usual constructor
  /// The device first divice is selected only one the memory is initialized
  ///
  explicit vector(const size_type n = 0, const value_type &v = value_type())
      : data_(n, v) {}

  ///
  /// \brief initializer constructor {1.,2. ...} col order !
  ///
  vector(std::initializer_list<value_type> l) { data_ = l; }

  ///
  /// \brief Addition between two vectors
  ///
  vector &operator+=(const vector &v) {
    assert(size() == v.size() &&
           " can not make addition between vector of different size ");
    int b(0);
    /*
    #ifdef CUDA_DEVICE
        if (gpu_ready_.compare_exchange_strong(b, 1)) {
          add_vector_gpu(*this, v);
          gpu_ready_ = 0;
        } else {
          add_vector_cpu(*this, v);
        }
    #else
        add_vector_cpu(*this, v);
    #endif
    */
    add_vector_cpu(*this, v);
    return *this;
  }

  ///
  /// \brief substraction between two vectors
  ///
  vector &operator-=(const vector &v) {
    assert(size() == v.size() &&
           " can not make substraction between vector of different size ");
    int b(0);
    /*
#ifdef CUDA_DEVICE
    if (gpu_ready_.compare_exchange_strong(b, 1)) {
      sub_vector_gpu(*this, v);
      gpu_ready_ = 0;
    } else {
      sub_vector_cpu(*this, v);
    }
#else
    sub_vector_cpu(*this, v);
#endif
*/
    sub_vector_cpu(*this, v);
    return *this;
  }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  inline auto begin() { return data_.begin(); }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  inline auto end() const { return data_.end(); }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  inline auto begin() const { return data_.begin(); }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  inline auto end() { return data_.end(); }

  ///
  /// \brief Return the size of the vector
  ///
  inline auto size() const { return data_.size(); }

  ///
  /// \brief Return a reference of the data using usual bracket operator syntax
  ///
  inline reference operator[](size_type i) {
    assert(i < data_.size() && " Too ambitious! \n");
    return data_[i];
  }

  ///
  /// \brief Return a cosnt reference of the data using usual bracket operator
  /// syntax
  ///
  inline const_reference operator[](size_type i) const {
    assert(i < data_.size() && " Too ambitious! \n");
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
  const_pointer data() const { return data_.data(); }

  ///
  /// \brief Return the data pointer need for copy block/matrix
  ///
  pointer data() { return data_.data(); }

  ///
  /// \brief Return the memory allocated
  ///
  size_type memory_allocated() const { return sizeof(T) * data_.size(); }
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
  std::vector<T, Allocator> data_; // pointer of the data
};

///
/// \brief Addition Substraction between two vectors on GPU
///
template <class T, class A>
void helper_add(vector<T, A> &v_y, const vector<T, A> &v_x, const float a) {
  cublasHandle_t handle;
  CUBLAS_STATUS_CALL(cublasCreate(&handle));
  int n = v_y.size();
  const float alpha = a;
  int incx(1);
  int incy(1);

  static float *x = nullptr;
  static float *y = nullptr;

  if (init_add_ == false) {
    CUDA_CALL(cudaMalloc((void **)&x, v_x.memory_allocated()));
    CUDA_CALL(cudaMalloc((void **)&y, v_y.memory_allocated()));
    init_add_ = true;
  }

  CUDA_CALL(cudaMemcpy(x, v_x.data(), v_x.memory_allocated(),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(y, v_y.data(), v_y.memory_allocated(),
                       cudaMemcpyHostToDevice));

  // const float *x = v_x.data();
  // float *y = v_y.data();
  // y = alpha * x + y
  CUBLAS_STATUS_CALL(cublasSaxpy(handle, n, &alpha, x, incx, y, incy));

  CUDA_CALL(cudaMemcpy(v_y.data(), y, v_y.memory_allocated(),
                       cudaMemcpyDeviceToHost));

  CUBLAS_STATUS_CALL(cublasDestroy(handle));
  nadd_gpu += 1;
}

///
/// \brief Addition between two vectors on GPU
///
template <class T, class A>
inline void add_vector_gpu(vector<T, A> &v_y, const vector<T, A> &v_x) {
  helper_add(v_y, v_x, 1.f);
}

///
/// \brief Substraction between two vectors on GPU
///
template <class T, class A>
inline void sub_vector_gpu(vector<T, A> &v_y, const vector<T, A> &v_x) {
  helper_add(v_y, v_x, -1.f);
}

///
/// \brief Addition between two vectors on CPU
///
template <class T, class A>
inline void add_vector_cpu(vector<T, A> &v_y, const vector<T, A> &v_x) {
  nadd_cpu += 1;
  using eigen_vector_type =
      Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using const_eigen_vector_type =
      const Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  Eigen::Map<eigen_vector_type>(v_y.data(), v_y.size()) +=
      Eigen::Map<const_eigen_vector_type>(v_x.data(), v_x.size());
}

///
/// \brief Substraction between two vectors on CPU
///
template <class T, class A>
inline void sub_vector_cpu(vector<T, A> &v_y, const vector<T, A> &v_x) {
  nadd_cpu += 1;
  using eigen_vector_type =
      Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using const_eigen_vector_type =
      const Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  Eigen::Map<eigen_vector_type>(v_y.data(), v_y.size()) -=
      Eigen::Map<const_eigen_vector_type>(v_x.data(), v_x.size());
}

///
/// \brief Overload << stream operator
///
template <class T, class A>
std::ostream &operator<<(std::ostream &out, const vector<T, A> &b) {
  b.print(out);
  return out;
}
