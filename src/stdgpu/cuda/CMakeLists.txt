
find_package(CUDAToolkit 11.0 REQUIRED MODULE)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(CUDAToolkit 11.0 REQUIRED MODULE)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/device.cpp
                              impl/memory.cpp)

target_compile_features(stdgpu PUBLIC cuda_std_17)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_link_libraries(stdgpu PUBLIC CUDA::cudart)
