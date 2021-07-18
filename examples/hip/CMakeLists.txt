
macro(stdgpu_add_example_hip)
    stdgpu_detail_add_example(${ARGV0} "hip")
endmacro()

stdgpu_add_example_hip(atomic)
stdgpu_add_example_hip(iterator)
stdgpu_add_example_hip(ranges)
