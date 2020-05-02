
# Input parameters:
# - File name of the example without file extension
# - File extension of the example
macro(stdgpu_detail_add_example)
    set(STDGPU_EXAMPLES_NAME "${ARGV0}")
    add_executable(${STDGPU_EXAMPLES_NAME} "${STDGPU_EXAMPLES_NAME}.${ARGV1}")
    target_compile_options(${STDGPU_EXAMPLES_NAME} PRIVATE ${STDGPU_DEVICE_FLAGS}
                                                           ${STDGPU_HOST_FLAGS})
    target_link_libraries(${STDGPU_EXAMPLES_NAME} PRIVATE stdgpu::stdgpu)
    set_target_properties(${STDGPU_EXAMPLES_NAME} PROPERTIES CXX_CLANG_TIDY "${STDGPU_PROPERTY_CLANG_TIDY}")
    set_target_properties(${STDGPU_EXAMPLES_NAME} PROPERTIES CXX_CPPCHECK "${STDGPU_PROPERTY_CPPCHECK}")
endmacro()

macro(stdgpu_add_example_cpp)
    stdgpu_detail_add_example(${ARGV0} "cpp")
endmacro()


stdgpu_add_example_cpp(contract)
stdgpu_add_example_cpp(createAndDestroyDeviceArray)
stdgpu_add_example_cpp(createAndDestroyDeviceObject)

add_subdirectory(${STDGPU_BACKEND_DIRECTORY})
