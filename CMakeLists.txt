cmake_minimum_required (VERSION 3.8)

project(strassen CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR})

option (CUDA_STRASSEN " using unify memory CUDA" OFF)
option (CUDA_DEVICE " using accelerator" OFF)

if(CUDA_DEVICE)
add_definitions(-DCUDA_DEVICE)
find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

if(CUDA_STRASSEN)
add_definitions(-DCUDA_STRASSEN)
find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

find_package(TBB )
find_package(Eigen3)

#include_directories(${TBB_INCLUDE_DIRS})

#include(ClangFormat)
#clangformat_setup(
#  sandbox/main.cpp
#  sandbox/sgemm.cpp
#  memory/vector.h
#  memory/matrix.h
#  memory/tile_matrix.h
#  memory/util.h
#  memory/allocator.h
#  algo/strassen.h
#  algo/classic.h
#  test/test_memory_vector.cpp
#  test/test_memory_matrix.cpp
#  test/test_memory_tile_matrix.cpp
#)

add_subdirectory(sandbox)

# Locate GTest
enable_testing()
find_package(GTest REQUIRED)
add_subdirectory(test)


