
add_executable(benchmarkstdgpu main.cpp)

add_subdirectory(${STDGPU_BACKEND_DIRECTORY})

target_include_directories(benchmarkstdgpu PRIVATE
                                          "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_compile_options(benchmarkstdgpu PRIVATE ${STDGPU_DEVICE_FLAGS}
                                               ${STDGPU_HOST_FLAGS}
                                               ${STDGPU_TEST_DEVICE_FLAGS}
                                               ${STDGPU_TEST_HOST_FLAGS})

target_link_libraries(benchmarkstdgpu PRIVATE
                                      stdgpu::stdgpu
                                      benchmark::benchmark)

set_target_properties(benchmarkstdgpu PROPERTIES CXX_CLANG_TIDY "${STDGPU_PROPERTY_CLANG_TIDY}")
set_target_properties(benchmarkstdgpu PROPERTIES CXX_CPPCHECK "${STDGPU_PROPERTY_CPPCHECK}")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_target_properties(benchmarkstdgpu PROPERTIES COMPILE_WARNING_AS_ERROR "${STDGPU_COMPILE_WARNING_AS_ERROR}")
endif()
