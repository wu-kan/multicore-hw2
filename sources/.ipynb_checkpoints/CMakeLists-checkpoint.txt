cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -use_fast_math -Xcompiler -Ofast -Xcompiler -fopenmp")

file(GLOB SOURCES
    "src/*.h"
    "src/*.cpp"
    "src/*.cu"
)

cuda_add_executable(
    main
    ${SOURCES}
)