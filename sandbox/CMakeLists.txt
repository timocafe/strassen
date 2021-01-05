CUDA_ADD_EXECUTABLE(strassen main.cu)
target_link_libraries (strassen Eigen3::Eigen ${CUDA_curand_LIBRARY})
