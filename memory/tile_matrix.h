// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <random>
#include <vector>

#include <tbb/tbb.h>

#include "memory/matrix.h"
#include "memory/util.h"
#include "memory/vector.h"

// matrix col order to be compliant with BLAS else ...
template <class T> class tile_matrix {
public:
  typedef uint32_t size_type;
  typedef T value_type;
  typedef value_type *pointer;
  typedef pointer iterator;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  ///
  /// \brief usual constructor
  /// compurte the number of tile needed
  ///
  explicit tile_matrix(const size_type rows = 0, const size_type cols = 0,
                       const size_type tile = 64)
      : rows_(rows), cols_(cols), tile_rows_((rows + tile - 1) / tile),
        tile_cols_((rows + tile - 1) / tile), tile_(tile),
        data_(tile_rows_ * tile_cols_, matrix<value_type>(tile, tile)){};

  ///
  /// \brief usual constructor
  /// compurte the number of tile needed
  ///
  explicit tile_matrix(const size_type rows, const size_type cols,
                       const size_type tile, bool b)
      : rows_(rows), cols_(cols), tile_rows_((rows + tile - 1) / tile),
        tile_cols_((rows + tile - 1) / tile), tile_(tile),
        data_(tile_rows_ * tile_cols_, matrix<value_type>(0, 0)){};

  ///
  /// \brief Return the number of cols
  ///
  inline size_type cols() const { return cols_; }

  ///
  /// \brief Return the number of rows
  ///
  inline size_type rows() const { return rows_; }

  ///
  /// \brief Return the number of tile cols
  ///
  inline size_type tile_cols() const { return tile_cols_; }

  ///
  /// \brief Return the number of rows
  ///
  inline size_type tile_rows() const { return tile_rows_; }

  ///
  /// \brief Return  a tile from the vector API
  ///
  inline matrix<T> &operator[](size_type i) { return data_[i]; }

  ///
  /// \brief Return  a tile from the vector API
  ///
  const inline matrix<T> &operator[](size_type i) const { return data_[i]; }

  ///
  /// \brief Return a reference of the data using usual bracket operator syntax,
  /// cols order, first get the tile and then the data (col order) for the
  /// corresponding tile
  ///
  inline reference operator()(size_type i, size_type j) {
    // get the tile
    size_type ti = i / tile_;
    size_type tj = j / tile_;
    // get the corresponding value in tile
    size_type iti = i % tile_;
    size_type itj = j % tile_;
    return data_[ti + tj * tile_rows_](iti, itj);
  }

  ///
  /// \brief Return a const reference of the data using usual bracket operator,
  /// cols order syntax
  ///
  inline const_reference operator()(size_type i, size_type j) const {
    // get the tile
    size_type ti = i / tile_;
    size_type tj = j / tile_;
    // get the corresponding value in tile
    size_type iti = i % tile_;
    size_type itj = j % tile_;
    return data_[ti + tj * tile_rows_](iti, itj);
  }

  ///
  /// \brief return the needed tile
  ///
  inline matrix<value_type> &tile(size_type i, size_type j) {
    // get the tile
    assert(i < tile_rows_ && " i < tile_rows ");
    assert(j < tile_cols_ && " i < tile_cols ");
    return data_[i + j * tile_rows_];
  }

  ///
  /// \brief return the needed tile
  ///
  inline const matrix<value_type> &tile(size_type i, size_type j) const {
    // get the tile
    assert(i < tile_rows_ && " i < tile_rows ");
    assert(j < tile_cols_ && " i < tile_cols ");
    return data_[i + j * tile_rows_];
  }

  ///
  /// \brief Return the total number of element
  ///
  size_type size() const { return data_.size(); }

  ///
  /// \brief Return the total number of element
  ///
  size_type tile() const { return tile_; }

  ///
  /// \brief Return the vector of tile
  ///
  const std::vector<matrix<value_type>> &data() const { return data_; }

  ///
  /// \brief Return the vector of tile
  ///
  std::vector<matrix<value_type>> &data() { return data_; }

  ///
  /// \brief Addition between two tile_matrix
  ///
  tile_matrix &operator+=(const tile_matrix &m) {
    // much simpler than std::transform
    for (int i = 0; i < data_.size(); ++i)
      data_[i] += m.data()[i];
    return *this;
  }

  ///
  /// \brief Substraction between two vectors
  ///
  tile_matrix &operator-=(const tile_matrix &m) {
    // much simpler than std::transform
    for (int i = 0; i < data_.size(); ++i)
      data_[i] -= m.data()[i];
    return *this;
  }

  ///
  /// \brief Print the tile matrix
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
  size_type tile_rows_;
  size_type tile_cols_;
  size_type tile_;
  std::vector<matrix<value_type>> data_;
};

///
/// \brief Overload << stream operator
///
template <class T>
std::ostream &operator<<(std::ostream &out, const tile_matrix<T> &b) {
  b.print(out);
  return out;
}

///
/// \brief fill up the matrix with random number
///
template <class T> void random(tile_matrix<T> &m) {
  parallel_for(tbb::blocked_range<size_t>(0, m.size()),
               [&](const tbb::blocked_range<size_t> &r) {
                 for (size_t i = r.begin(); i != r.end(); ++i)
                   random(m[i]);
               });
}

template <class T>
inline auto operator+(tile_matrix<T> mA, const tile_matrix<T> &mB) {
  mA += mB;
  return std::move(mA);
}

template <class T>
inline auto operator-(tile_matrix<T> mA, const tile_matrix<T> &mB) {
  mA -= mB;
  return std::move(mA);
}

template <class T>
inline auto mul(const tile_matrix<T> &mA, const tile_matrix<T> &mB) {
  tile_matrix<float> m(mA.rows(), mA.cols(), mA.tile());
  m.tile(0, 0) = std::move(mA.tile(0, 0) * mB.tile(0, 0));
  return std::move(m);
}

///
/// \brief copy block from a matrix to an other one
///
template <class T>
inline void copy_block(tile_matrix<T> &m1, const tile_matrix<T> &m2, uint32_t x,
                       uint32_t y) {
  uint32_t nrows = (m2.tile_rows() == 1) ? 1 : m2.tile_rows() / 2;
  uint32_t ncols = (m2.tile_cols() == 1) ? 1 : m2.tile_cols() / 2;
  for (int i = 0; i < ncols; ++i)
    for (int j = 0; j < nrows; ++j)
      m1.tile(i, j) = std::move(m2.tile(i + x, j + y));
}

///
/// \brief copyasuefull matrix to an other one subblock
/// It just copy adress and block size
///
template <class T>
inline void copy_matrix(tile_matrix<T> &m1, const tile_matrix<T> &m2,
                        uint32_t x, uint32_t y) {
  uint32_t nrows = m2.tile_rows();
  uint32_t ncols = m2.tile_cols();
  for (int i = 0; i < ncols; ++i)
    for (int j = 0; j < nrows; ++j)
      m1.tile(i + x, j + y) = std::move(m2.tile(i, j));
}

///
/// \brief add matrix to an other one subblock
/// It just copy adress and block size
///
template <class T>
inline void tile_add_matrix(tile_matrix<T> &m1, const tile_matrix<T> &m2,
                            uint32_t x, uint32_t y) {
  uint32_t nrows = m2.tile_rows();
  uint32_t ncols = m2.tile_cols();
  for (int i = 0; i < ncols; ++i)
    for (int j = 0; j < nrows; ++j)
      m1.tile(i + x, j + y) += m2.tile(i, j);
}

/// \brief sub matrix to an other one subblock
/// It just copy adress and block size
///
template <class T>
inline void tile_sub_matrix(tile_matrix<T> &m1, const tile_matrix<T> &m2,
                            uint32_t x, uint32_t y) {
  uint32_t nrows = m2.tile_rows();
  uint32_t ncols = m2.tile_cols();
  for (int i = 0; i < ncols; ++i)
    for (int j = 0; j < nrows; ++j)
      m1.tile(i + x, j + y) -= m2.tile(i, j);
}

///
/// \brief affregate tile matix to a single one for validation purpose
///
template <class T> inline matrix<T> aggregate(const tile_matrix<T> &A) {
  matrix<T> B(A.rows(), A.cols());
  uint32_t nrows = A.tile_rows();
  uint32_t ncols = A.tile_cols();

  for (int i = 0; i < nrows; ++i)
    for (int j = 0; j < ncols; ++j) {
      const auto &m = A.tile(i, j);
      for (int k = 0; k < m.cols(); ++k) {
        auto begin = &m(0, k);
        auto end = begin + m.rows();
        std::copy(begin, end, &B(i * m.rows(), j * m.cols() + k));
      }
    }
  return B;
}
