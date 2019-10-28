
add_executable(teststdgpu main.cpp)

add_subdirectory(cuda)

target_sources(teststdgpu PRIVATE algorithm.cpp
                                  atomic.cu
                                  bit.cpp
                                  bitset.cu
                                  cmath.cpp
                                  cstdlib.cpp
                                  deque.cu
                                  functional.cpp
                                  iterator.cpp
                                  limits.cpp
                                  memory.cpp
                                  memory.cu
                                  mutex.cu
                                  unordered_map.cu
                                  unordered_set.cu
                                  vector.cu)

target_include_directories(teststdgpu PRIVATE
                                      "${CMAKE_CURRENT_SOURCE_DIR}/..") # test_utils

target_link_libraries(teststdgpu PRIVATE
                                 stdgpu::stdgpu
                                 gtest)


add_test(NAME teststdgpu
         COMMAND teststdgpu)
