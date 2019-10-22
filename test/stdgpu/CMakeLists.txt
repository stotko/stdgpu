
add_executable(teststdgpu main.cpp)

target_sources(teststdgpu PRIVATE cuda/atomic.cu
                                  algorithm.cpp
                                  atomic.cu
                                  bit.cu
                                  bitset.cu
                                  cmath.cpp
                                  cstdlib.cpp
                                  deque.cu
                                  functional.cpp
                                  iterator.cu
                                  limits.cpp
                                  memory.cpp
                                  memory.cu
                                  mutex.cu
                                  unordered_map.cu
                                  unordered_set.cu
                                  vector.cu)

target_include_directories(teststdgpu PRIVATE
                                      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                                      "${CMAKE_CURRENT_SOURCE_DIR}/..") # test_utils

target_link_libraries(teststdgpu PRIVATE
                                 stdgpu::stdgpu
                                 gtest)


add_test(NAME teststdgpu
         COMMAND teststdgpu)
