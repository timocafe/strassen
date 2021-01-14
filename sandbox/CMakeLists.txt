
if(CUDA_STRASSEN)
CUDA_ADD_EXECUTABLE(strassen main.cpp)
else()
ADD_EXECUTABLE(strassen main.cpp)
Endif()
target_link_libraries (strassen Eigen3::Eigen TBB::tbb TBB::tbbmalloc_proxy TBB::tbbmalloc ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
