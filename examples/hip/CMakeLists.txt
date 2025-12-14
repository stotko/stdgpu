
macro(stdgpu_add_example_hip)
    stdgpu_detail_add_example(${ARGV0} "hip")
endmacro()

stdgpu_add_example_hip(atomic)
stdgpu_add_example_hip(bitset)
stdgpu_add_example_hip(deque)
stdgpu_add_example_hip(iterator)
stdgpu_add_example_hip(mutex_array)
stdgpu_add_example_hip(ranges)
stdgpu_add_example_hip(unordered_map)
stdgpu_add_example_hip(unordered_set)
stdgpu_add_example_hip(vector)
