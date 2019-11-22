
find_package(thrust 1.9.3 REQUIRED)
find_package(OpenMP 2.0 REQUIRED)

target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cpp
                              impl/mutex.cpp)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

target_link_libraries(stdgpu PUBLIC OpenMP::OpenMP_CXX
                                    thrust::thrust)
