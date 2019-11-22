
add_executable(teststdgpu main.cpp)

target_sources(teststdgpu PRIVATE algorithm.cpp
                                  bit.cpp
                                  cmath.cpp
                                  cstdlib.cpp
                                  functional.cpp
                                  iterator.cpp
                                  limits.cpp
                                  memory.cpp)

add_subdirectory(${STDGPU_BACKEND_DIRECTORY})

target_include_directories(teststdgpu PRIVATE
                                      "${CMAKE_CURRENT_SOURCE_DIR}/..") # test_utils

target_link_libraries(teststdgpu PRIVATE
                                 stdgpu::stdgpu
                                 gtest)


add_test(NAME teststdgpu
         COMMAND teststdgpu)
