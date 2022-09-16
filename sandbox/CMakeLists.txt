
ADD_EXECUTABLE(strassen main.cpp)
ADD_EXECUTABLE(sgemm sgemm.cpp)
target_link_libraries (strassen Eigen3::Eigen TBB::tbb TBB::tbbmalloc_proxy ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
target_link_libraries (sgemm Eigen3::Eigen -fopenmp TBB::tbb TBB::tbbmalloc_proxy ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
target_compile_options(sgemm PRIVATE -fopenmp)
