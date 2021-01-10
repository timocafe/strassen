// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cublas_v2.h>
#include <curand.h>

#include "memory/util.h"
#include "memory/vector.h"

// matrix col order to be compliant with BLAS else ...
template <class T> class matrix {
public:
  typedef uint32_t size_type;
  typedef T value_type;
  typedef value_type *pointer;
  typedef pointer iterator;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  ///
  /// \brief usual constructor
  /// The device first divice is selected only once the memory is initialized
  ///
  explicit matrix(const size_type rows = 0, const size_type cols = 0)
      : rows_(rows), cols_(cols), data_(rows * cols) {}

  ///
  /// \brief copy constructor, needed because following standard as soons as
  /// move assign or move copy is coded the copy destructor is marked deleted
  ///
  matrix(const matrix &other)
      : data_(other.data_), rows_(other.rows()), cols_(other.cols()) {}

  ///
  /// \brief move constructor, the automatic does not reset rows_ and cols_ to 0
  /// (no swap)
  ///
  matrix(matrix &&other)
      : data_(std::move(other.data_)), rows_(std::move(other.rows_)),
        cols_(std::move(other.cols_)) {
    other.rows_ = 0;
    other.cols_ = 0;
  }

  ///
  /// \brief assign move operator, needed because move constructor
  ///
  matrix &operator=(matrix &&other) {
    std::swap(data_, other.data_);
    std::swap(cols_, other.cols_);
    std::swap(rows_, other.rows_);
    return *this;
  }

  ///
  /// \brief initializer constructor {1.,2. ...} col order !
  ///
  matrix &operator=(std::initializer_list<value_type> l) {
    assert(l.size() <= rows() * cols() &&
           " more element than the dimenstion of the matrix ");
    data_ = std::move(vector<value_type>(l));
    return *this;
  }

  ///
  /// \brief Return the number of cols
  ///
  inline size_type cols() const { return cols_; }

  ///
  /// \brief Return the number of cols
  ///
  inline size_type rows() const { return rows_; }

  ///
  /// \brief Return the const data pointer
  ///
  pointer data() const { return data_.data(); }

  ///
  /// \brief Return the data pointer
  ///
  pointer data() { return data_.data(); }

  ///
  /// \brief Return the total number of element
  ///
  size_type size() const { return data_.size(); }

  ///
  /// \brief Return a reference of the data using usual bracket operator syntax,
  /// cols order
  ///
  DEVICE_CALLABLE
  inline reference operator()(size_type i, size_type j) {
    return data_[i + j * rows_];
  }

  ///
  /// \brief Return a const reference of the data using usual bracket operator,
  /// cols order syntax
  ///
  DEVICE_CALLABLE
  inline const_reference operator()(size_type i, size_type j) const {
    return data_[i + j * rows_];
  }

  ///
  /// \brief Return the memory allocated
  ///
  size_type memory_allocated() const { return data_.memory_allocated(); }

  ///
  //
  /// \brief Prefetch data on the gpu
  ///
  void prefetch_gpu(cudaStream_t s = 0) const { data_.prefetch_gpu(s); }

  ///
  /// \brief Prefetch data on the cpu
  ///
  void prefetch_cpu(cudaStream_t s = 0) const { data_.prefetch_cpu(s); }

  ///
  /// \brief Addition between two matrix
  ///
  matrix &operator+=(const matrix &m) {
    data_ += m.data_;
    return *this;
  }

  ///
  /// \brief Substraction between two matrix
  ///
  matrix &operator-=(const matrix &m) {
    data_ -= m.data_;
    return *this;
  }

  ///
  /// \brief Print the matrix
  ///
  void print(std::ostream &out) const {
    for (int i = 0; i < rows(); ++i) {
      for (int j = 0; j < cols(); ++j)
        out << (*this)(i, j) << " ";
      out << "\n";
    }
    out << "\n";
  }

private:
  size_type rows_;
  size_type cols_;
  vector<value_type> data_;
};

///
/// \brief Overload << stream operator
///
template <class T>
std::ostream &operator<<(std::ostream &out, const matrix<T> &b) {
  b.print(out);
  return out;
}

template <class T>
inline void copy_block(matrix<T> &m1, const matrix<T> &m2, uint32_t x,
                       uint32_t y) {
  m1.prefetch_cpu();
  m2.prefetch_cpu();
  using eigen_vector_type =
      Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using const_eigen_vector_type =
      const Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  const uint32_t size = m1.rows();
  const uint32_t size_m2 = 2 * m1.rows();
  for (int i = 0; i < m1.cols(); ++i) {
    Eigen::Map<eigen_vector_type>(m1.data() + i * size, size) =
        Eigen::Map<const_eigen_vector_type>((&m2(x, y) + i * size_m2), size);
  }
}

template <class T>
inline void copy_matrix(matrix<T> &m1, const matrix<T> &m2, uint32_t x,
                        uint32_t y) {
  m1.prefetch_cpu();
  m2.prefetch_cpu();
  using eigen_vector_type =
      Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using const_eigen_vector_type =
      const Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  const uint32_t size = m2.rows();
  const uint32_t size_m2 = 2 * m2.rows();
  for (int i = 0; i < m2.cols(); ++i) {
    Eigen::Map<eigen_vector_type>((&m1(x, y) + i * size_m2), size) =
        Eigen::Map<const_eigen_vector_type>(m2.data() + i * size, size);
  }
}

template <class T> void random(matrix<T> &m) {
  m.prefetch_gpu();
  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  /* Generate n floats on device */
  CURAND_CALL(curandGenerateUniform(gen, m.data(), m.size()));
  /* Cleanup */
  CURAND_CALL(curandDestroyGenerator(gen));
  m.prefetch_cpu();
}

template <class T>
inline void mul_matrix_gpu(matrix<T> &mC, const matrix<T> &mA,
                           const matrix<T> &mB) {
  using size_type = typename matrix<T>::size_type;
  cublasHandle_t handle;
  CUBLAS_STATUS_CALL(cublasCreate(&handle));

  float *A = mA.data();
  float *B = mB.data();
  float *C = mC.data();

  float alpha = 1.f;
  float beta = 0.f;

  size_type lda = mA.rows();
  size_type ldb = mB.rows();
  size_type ldc = mC.rows();

  assert(mC.rows() == mA.rows());
  size_type m = mC.rows();
  assert(mC.cols() == mB.cols());
  size_type n = mC.cols();
  assert(mA.cols() == mB.rows());
  size_type k = mB.rows();

  CUBLAS_STATUS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                 &alpha, A, lda, B, ldb, &beta, C, ldc));
  CUBLAS_STATUS_CALL(cublasDestroy(handle));
  gpu_ready_.fetch_add(1);
}

template <class T>
inline void mul_matrix_cpu(matrix<T> &mC, const matrix<T> &mA,
                           const matrix<T> &mB) {
  using value_type = T;
  using eigen_matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::ColMajor>;
  Eigen::Map<eigen_matrix_type>(mC.data(), mC.rows(), mC.cols()) =
      Eigen::Map<eigen_matrix_type>(mA.data(), mB.rows(), mC.cols()) *
      Eigen::Map<eigen_matrix_type>(mB.data(), mB.rows(), mC.cols());
}

template <class T>
inline auto operator*(const matrix<T> &mA, const matrix<T> &mB) {
  using size_type = typename matrix<float>::size_type;
  size_type rows = mA.rows();
  size_type cols = mB.cols();
  matrix<float> mC(rows, cols);
  if (gpu_ready_.fetch_sub(1))
    mul_matrix_gpu(mC, mA, mB);
  else
    mul_matrix_cpu(mC, mA, mB);

  return std::move(mC);
}

template <class T>
inline auto operator+(const matrix<T> &mA, const matrix<T> &mB) {
  matrix<float> m(mA); // copy constructor
  m += mB;
  return std::move(m);
}

template <class T>
inline auto operator-(const matrix<T> &mA, const matrix<T> &mB) {
  matrix<float> m(mA); // copy constructor
  m -= mB;
  return std::move(m);
}
