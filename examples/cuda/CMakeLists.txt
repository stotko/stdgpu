
macro(stdgpu_add_example_cu)
    stdgpu_detail_add_example(${ARGV0} "cu")
endmacro()

stdgpu_add_example_cu(bitset)
stdgpu_add_example_cu(container_iterator)
stdgpu_add_example_cu(container_kernel)
stdgpu_add_example_cu(mutex_array)
stdgpu_add_example_cu(thrust_interoperability)
stdgpu_add_example_cu(thrust_towards_ranges)
