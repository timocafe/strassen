
if(CUDA_STRASSEN)
CUDA_ADD_EXECUTABLE(strassen main.cpp)
else()
ADD_EXECUTABLE(strassen main.cpp)
target_link_libraries (strassen Eigen3::Eigen ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
Endif()
