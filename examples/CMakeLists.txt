macro(stdgpu_detail_add_example)
    set(STDGPU_EXAMPLES_NAME "${ARGV0}")
    add_executable(${STDGPU_EXAMPLES_NAME} "${STDGPU_EXAMPLES_NAME}.${ARGV1}")
    target_link_libraries(${STDGPU_EXAMPLES_NAME} PRIVATE stdgpu::stdgpu)
endmacro()

macro(stdgpu_add_example_cu)
    stdgpu_detail_add_example(${ARGV0} "cu")
endmacro()

macro(stdgpu_add_example_cpp)
    stdgpu_detail_add_example(${ARGV0} "cpp")
endmacro()


stdgpu_add_example_cu(bitset)
stdgpu_add_example_cu(container_iterator)
stdgpu_add_example_cu(container_kernel)
stdgpu_add_example_cpp(contract)
stdgpu_add_example_cu(createAndDestroyDeviceArray)
stdgpu_add_example_cu(createAndDestroyDeviceObject)
stdgpu_add_example_cu(mutex_array)
stdgpu_add_example_cu(thrust_interoperability)
stdgpu_add_example_cu(thrust_towards_ranges)
