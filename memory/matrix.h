// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifdef CUDA_STRASSEN
#include <cublas_v2.h>
#include <curand.h>
#endif

#include <random>

#include "memory/util.h"
#include "memory/vector.h"

// matrix col order to be compliant with BLAS else ...
template <class T, class A = cstandard> class matrix {
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
  /// The device first divice is selected only once the memory is initialized
  ///
  explicit matrix(const size_type rows = 0, const size_type cols = 0,
                  const size_type init = 1)
      : rows_(rows), cols_(cols), data_(rows * cols * init) {}

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
  /// \brief assign operator
  ///
  matrix &operator=(const matrix &other) {
    cols_ = other.cols_;
    rows_ = other.rows_;
    data_ = other.data_;
    return *this;
  }

  ///
  /// \brief initializer constructor {1.,2. ...} col order !
  ///
  matrix &operator=(std::initializer_list<value_type> l) {
    assert(l.size() <= rows() * cols() &&
           " more element than the dimenstion of the matrix ");
    data_ = std::move(vector<value_type, A>(l));
    return *this;
  }
  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  iterator begin() { return data_.begin(); }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  const_iterator begin() const { return data_.begin(); }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  const_iterator end() const { return data_.end(); }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  iterator end() { return data_.end(); }

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
  inline reference operator()(size_type i, size_type j) {
    return data_[i + j * rows_];
  }

  ///
  /// \brief Return a const reference of the data using usual bracket operator,
  /// cols order syntax
  ///
  inline const_reference operator()(size_type i, size_type j) const {
    return data_[i + j * rows_];
  }

  ///
  /// \brief Return the memory allocated
  ///
  size_type memory_allocated() const { return data_.memory_allocated(); }

#ifdef CUDA_STRASSEN
  ///
  /// \brief Prefetch data on the gpu
  ///
  void prefetch_gpu(cudaStream_t s = 0) const { data_.prefetch_gpu(s); }

  ///
  /// \brief Prefetch data on the cpu
  ///
  void prefetch_cpu(cudaStream_t s = 0) const { data_.prefetch_cpu(s); }
#endif
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
  /// \brief Substraction between two matrix
  ///
  bool operator==(const matrix &m) {
    auto epsilon = [](value_type a, value_type b) -> value_type {
      return std::max(std::abs(a), std::abs(b)) * 1; // tolerence ...
    };
    auto g = [&](value_type a, value_type b) {
      return ((a - b) < epsilon(a, b)) && ((b - a) < epsilon(a, b));
    };
    bool b = true;
    for (int i = 0; i < rows(); ++i)
      for (int j = 0; j < cols(); ++j)
        b &= g((*this)(i, j), m(i, j)); // as soon as false game over
    return b;
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
  vector<value_type, A> data_;
};

///
/// \brief Overload << stream operator
///
template <class T, class A>
std::ostream &operator<<(std::ostream &out, const matrix<T, A> &b) {
  b.print(out);
  return out;
}

template <class T, class A>
inline void copy_block(matrix<T, A> &m1, const matrix<T, A> &m2, uint32_t x,
                       uint32_t y) {

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

template <class T, class A>
inline void copy_matrix(matrix<T, A> &m1, const matrix<T, A> &m2, uint32_t x,
                        uint32_t y) {

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

template <class T, class A> void random(matrix<T, A> &m) { random_cpu(m); }
#ifdef CUDA_STRASSEN
template <class T, class A> void random_gpu(matrix<T, A> &m) {
  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  /* Generate n floats on device */
  CURAND_CALL(curandGenerateUniform(gen, m.data(), m.size()));
  /* Cleanup */
  CURAND_CALL(curandDestroyGenerator(gen));
}
#endif

template <class T, class A> void random_cpu(matrix<T, A> &m) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<> dist{-1., 1.};
  std::generate(m.begin(), m.end(), [&]() { return dist(mersenne_engine); });
}

#ifdef CUDA_STRASSEN

template <class T, class A>
inline void mul_matrix_gpu(matrix<T, A> &mC, const matrix<T, A> &mA,
                           const matrix<T, A> &mB) {
  using size_type = typename matrix<T, A>::size_type;
  cublasHandle_t handle;
  CUBLAS_STATUS_CALL(cublasCreate(&handle));

  matrix<T, cuda_unify> uA(mA.rows(), mA.cols());
  matrix<T, cuda_unify> uB(mB.rows(), mB.cols());
  matrix<T, cuda_unify> uC(mC.rows(), mC.cols());

  uA.prefetch_cpu();
  uB.prefetch_cpu();
  uC.prefetch_cpu();

  std::copy(mA.begin(), mA.end(), uA.begin());
  std::copy(mB.begin(), mB.end(), uB.begin());
  std::copy(mC.begin(), mC.end(), uC.begin());

  uA.prefetch_gpu();
  uB.prefetch_gpu();
  uC.prefetch_gpu();

  float *pA = uA.data();
  float *pB = uB.data();
  float *pC = uC.data();

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
                                 &alpha, pA, lda, pB, ldb, &beta, pC, ldc));
  CUBLAS_STATUS_CALL(cublasDestroy(handle));

  uC.prefetch_cpu();
  std::copy(uC.begin(), uC.end(), mC.begin());
}
#endif

template <class T, class A>
inline void mul_matrix_cpu(matrix<T, A> &mC, const matrix<T, A> &mA,
                           const matrix<T, A> &mB) {
  using value_type = T;
  using eigen_matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::ColMajor>;
  Eigen::Map<eigen_matrix_type>(mC.data(), mC.rows(), mC.cols()) =
      Eigen::Map<eigen_matrix_type>(mA.data(), mB.rows(), mC.cols()) *
      Eigen::Map<eigen_matrix_type>(mB.data(), mB.rows(), mC.cols());
}

template <class T, class A>
inline auto operator*(const matrix<T, A> &mA, const matrix<T, A> &mB) {
  using size_type = typename matrix<float>::size_type;
  size_type rows = mA.rows();
  size_type cols = mB.cols();
  matrix<float> mC(rows, cols);

  int b(0);
  if (!gpu_ready_.compare_exchange_strong(b, 1)) {
    mul_matrix_gpu(mC, mA, mB);
    gpu_ready_ = 0;
  } else {
    mul_matrix_cpu(mC, mA, mB);
  }

  return std::move(mC);
}

template <class T, class A>
inline auto operator+(const matrix<T, A> &mA, const matrix<T, A> &mB) {
  matrix<float> m(mA); // copy constructor
  m += mB;
  return std::move(m);
}

template <class T, class A>
inline auto operator-(const matrix<T, A> &mA, const matrix<T, A> &mB) {
  matrix<float> m(mA); // copy constructor
  m -= mB;
  return std::move(m);
}
