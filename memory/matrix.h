// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifdef CUDA_DEVICE

#include <cublas_v2.h>
#include <curand.h>

#endif

#include <random>

#include "memory/util.h"
#include "memory/vector.h"

// matrix col order to be compliant with BLAS else ...
template <class T> class matrix {
public:
  typedef uint32_t size_type;
  typedef T value_type;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;
  typedef pointer iterator;
  typedef const pointer const_iterator;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  ///
  /// \brief usual constructor
  /// The device first divice is selected only once the memory is initialized
  ///
  explicit matrix(const size_type rows = 0, const size_type cols = 0,
                  const T &t = T())
      : rows_(rows), cols_(cols), data_(rows * cols, t) {}

  ///
  /// \brief initializer constructor {1.,2. ...} col order !
  ///
  matrix &operator=(std::initializer_list<value_type> l) {
    assert(l.size() <= rows() * cols() &&
           " more element than the dimenstion of the matrix ");
    data_ = l;
    return *this;
  }
  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  auto begin() { return data_.begin(); }

  ///
  /// \brief Return an iterator at the beginning of the vector
  ///
  auto begin() const { return data_.begin(); }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  auto end() const { return data_.end(); }

  ///
  /// \brief Return an iterator at the end of the vector
  ///
  auto end() { return data_.end(); }

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
  const_pointer data() const { return data_.data(); }

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

template <class T> void random(matrix<T> &m) { random_cpu(m); }
#ifdef CUDA_STRASSEN
template <class T> void random_gpu(matrix<T> &m) {
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

template <class T> void random_cpu(matrix<T> &m) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<> dist{1., 2.};
  std::generate(m.begin(), m.end(), [&]() { return dist(mersenne_engine); });
}
#ifdef CUDA_DEVICE
template <class T>
inline void mul_matrix_gpu(matrix<T> &mC, const matrix<T> &mA,
                           const matrix<T> &mB) {
  using size_type = typename matrix<T>::size_type;
  auto handle = singleton::get().cublas_handle();

  static float *pA = nullptr;
  static float *pB = nullptr;
  static float *pC = nullptr;

  if (init_mul_ == false) {
    CUDA_CALL(cudaMalloc((void **)&pA, mA.memory_allocated()));
    CUDA_CALL(cudaMalloc((void **)&pB, mB.memory_allocated()));
    CUDA_CALL(cudaMalloc((void **)&pC, mC.memory_allocated()));
    init_mul_ = true;
  }

  CUDA_CALL(
      cudaMemcpy(pA, mA.data(), mA.memory_allocated(), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(pB, mB.data(), mB.memory_allocated(), cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(pC, mC.data(), mC.memory_allocated(), cudaMemcpyHostToDevice));

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
  CUDA_CALL(
      cudaMemcpy(mC.data(), pC, mC.memory_allocated(), cudaMemcpyDeviceToHost));

  nmul_gpu += 1;
}
#endif
template <class T>
inline void mul_matrix_cpu(matrix<T> &mC, const matrix<T> &mA,
                           const matrix<T> &mB) {
  nmul_cpu += 1;
  using value_type = T;
  using eigen_matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::ColMajor>;
  using const_eigen_matrix_type =
      const Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor>;
  Eigen::Map<eigen_matrix_type>(mC.data(), mC.rows(), mC.cols()) =
      Eigen::Map<const_eigen_matrix_type>(mA.data(), mA.rows(), mA.cols()) *
      Eigen::Map<const_eigen_matrix_type>(mB.data(), mB.rows(), mB.cols());
}

template <class T>
inline auto operator*(const matrix<T> &mA, const matrix<T> &mB) {
  using size_type = typename matrix<float>::size_type;
  size_type rows = mA.rows();
  size_type cols = mB.cols();
  matrix<T> mC(rows, cols);

  int b(0);

#ifdef CUDA_DEVICE
  if (gpu_ready_.compare_exchange_strong(b, 1)) {
    mul_matrix_gpu(mC, mA, mB);
    gpu_ready_ = 0;
  } else
    mul_matrix_cpu(mC, mA, mB);
#else
  mul_matrix_cpu(mC, mA, mB);
#endif
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
