
# Link runTests with what we want to test and the GTest and pthread library
include(GoogleTest)
if(CUDA_STRASSEN)
CUDA_ADD_EXECUTABLE(test_memory test_memory_vector.cpp test_memory_matrix.cpp test_memory_tile_matrix.cpp)
else()
ADD_EXECUTABLE(test_memory test_memory_vector.cpp test_memory_matrix.cpp test_memory_tile_matrix.cpp)
endif()

target_link_libraries(test_memory GTest::GTest GTest::Main Eigen3::Eigen ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
gtest_discover_tests(test_memory)
#add_test(NAME test_memory COMMAND test_memory})
