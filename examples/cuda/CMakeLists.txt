
macro(stdgpu_add_example_cu)
    stdgpu_detail_add_example(${ARGV0} "cu")
endmacro()

stdgpu_add_example_cu(atomic)
stdgpu_add_example_cu(bitset)
stdgpu_add_example_cu(deque)
stdgpu_add_example_cu(iterator)
stdgpu_add_example_cu(mutex_array)
stdgpu_add_example_cu(ranges)
stdgpu_add_example_cu(unordered_map)
stdgpu_add_example_cu(unordered_set)
stdgpu_add_example_cu(vector)
