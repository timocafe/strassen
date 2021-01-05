// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <curand.h>

#include "memory/util.h"
#include "memory/vector.h"

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
  explicit matrix(const size_type rows = 1, const size_type cols = 1)
      : rows_(rows), cols_(cols), data_(rows * cols) {}

  ///
  /// \brief Return the number of cols
  ///
  inline size_type cols() const { return cols_; }

  ///
  /// \brief Return the number of cols
  ///
  inline size_type rows() const { return rows_; }

  ///
  /// \brief Return the data pointer
  ///
  pointer data() const { return data_.data(); }

  ///
  /// \brief Return the total number of element
  ///
  size_type size() const { return data_.size(); }

  ///
  /// \brief Return a reference of the data using usual bracket operator syntax
  ///
  DEVICE_CALLABLE
  inline reference operator()(size_type i, size_type j) {
    return data_[i * cols_ + j];
  }

  ///
  /// \brief Return a const reference of the data using usual bracket operator
  /// syntax
  ///
  DEVICE_CALLABLE
  inline const_reference operator()(size_type i, size_type j) const {
    return data_[i * cols_ + j];
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
  /// \brief Print the vector
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

///
/// \brief Overload << stream operator
///
template <class T>
std::ostream &operator<<(std::ostream &out, const matrix<T> &b) {
  b.print(out);
  return out;
}
