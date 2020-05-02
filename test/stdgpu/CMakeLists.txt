
add_executable(teststdgpu main.cpp)

target_sources(teststdgpu PRIVATE algorithm.cpp
                                  bit.cpp
                                  cmath.cpp
                                  cstdlib.cpp
                                  functional.cpp
                                  iterator.cpp
                                  limits.cpp
                                  memory.cpp
                                  ranges.cpp)

add_subdirectory(${STDGPU_BACKEND_DIRECTORY})

target_include_directories(teststdgpu PRIVATE
                                      "${CMAKE_CURRENT_SOURCE_DIR}/..") # test_utils

target_compile_options(teststdgpu PRIVATE ${STDGPU_DEVICE_FLAGS}
                                          ${STDGPU_HOST_FLAGS}
                                          ${STDGPU_TEST_DEVICE_FLAGS}
                                          ${STDGPU_TEST_HOST_FLAGS})

target_link_libraries(teststdgpu PRIVATE
                                 stdgpu::stdgpu
                                 gtest)

set_target_properties(teststdgpu PROPERTIES CXX_CLANG_TIDY "${STDGPU_PROPERTY_CLANG_TIDY}")
set_target_properties(teststdgpu PROPERTIES CXX_CPPCHECK "${STDGPU_PROPERTY_CPPCHECK}")


add_test(NAME teststdgpu
         COMMAND teststdgpu)
