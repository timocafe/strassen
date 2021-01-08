// Strassen matrix multiplication implementation.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "memory/matrix.h"

//
// \brief strassen algo with notation wikipedia, lbs indicate the limit block size to stop the reccursive algo
//

template<class T>
auto strassen(const matrix<T>& A, const matrix<T>& B, const uint32_t lbs = 64){
     const uint32_t n = A.rows();
     const uint32_t k = A.rows()/2; // limit blocksize
     if(lbs == n)
         return std::move(A*B);

     matrix<T> A11(k,k); 
     matrix<T> A12(k,k); 
     matrix<T> A21(k,k); 
     matrix<T> A22(k,k); 

     matrix<T> B11(k,k); 
     matrix<T> B12(k,k); 
     matrix<T> B21(k,k); 
     matrix<T> B22(k,k); 

     copy_block(A11, A, 0, 0);
     copy_block(A12, A, 0, k);
     copy_block(A21, A, k, 0);
     copy_block(A22, A, k, k);

     copy_block(B11, B, 0, 0);
     copy_block(B12, B, 0, k);
     copy_block(B21, B, k, 0);
     copy_block(B22, B, k, k);

     const auto& M1 = strassen(A11 +  A22, B11 + B22);
     const auto& M2 = strassen(A21 +  A22, B11);
     const auto& M3 = strassen(A11, B12 - B22);
     const auto& M4 = strassen(A22, B21 - B11);
     const auto& M5 = strassen(A11 +  A12, B22);
     const auto& M6 = strassen(A21 +  A11, B11 +  B12);
     const auto& M7 = strassen(A12 -  A22, B21 + B22);

     matrix<T> C11(k,k); 
     matrix<T> C12(k,k); 
     matrix<T> C21(k,k); 
     matrix<T> C22(k,k); 

     C11 = M1 + M4 - M5 + M7;
     C12 = M3 + M5;
     C21 = M2 + M4;
     C22 = M1 - M2 + M3 + M5;

     matrix<T> C(n,n);

     copy_matrix(C, C11, 0,0);
     copy_matrix(C, C12, 0,k);
     copy_matrix(C, C21, k,0);
     copy_matrix(C, C22, k,k);

     return std::move(C);
}
