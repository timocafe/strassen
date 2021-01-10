// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

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
  /// \brief Return a reference of the data using usual bracket operator syntax,
  /// cols order, first get the tile and then the data (col order) for the
  /// corresponding tile
  ///
  DEVICE_CALLABLE
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
  DEVICE_CALLABLE
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
  /// \brief Return the total number of element
  ///
  size_type size() const { return data_.size(); }

private:
  size_type rows_;
  size_type cols_;
  size_type tile_rows_;
  size_type tile_cols_;
  size_type tile_;
  std::vector<matrix<value_type>> data_;
};
