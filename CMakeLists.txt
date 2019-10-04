cmake_minimum_required(VERSION 3.0)
project(benchmark)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA.
find_package(CUDA REQUIRED)

cuda_add_executable(device_device_cpy device_device_cpy.cpp)
cuda_add_executable(host_device_cpy host_device_cpy.cpp)
cuda_add_executable(host_device_pinned_cpy host_device_pinned_cpy.cpp)
