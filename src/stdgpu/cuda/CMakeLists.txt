
find_package(CUDAToolkit 11.0 REQUIRED MODULE)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(CUDAToolkit 11.0 REQUIRED MODULE)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/device.cpp
                              impl/memory.cpp)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(stdgpu PUBLIC FILE_SET stdgpu_backend_headers
                                 TYPE HEADERS
                                 BASE_DIRS ${STDGPU_INCLUDE_LOCAL_DIR}
                                 FILES atomic.cuh
                                       device.h
                                       memory.h
                                       platform.h
                                       platform_check.h)

    target_sources(stdgpu PUBLIC FILE_SET stdgpu_backend_header_implementations
                                 TYPE HEADERS
                                 BASE_DIRS ${STDGPU_INCLUDE_LOCAL_DIR}
                                 FILES impl/atomic_detail.cuh)
endif()

target_compile_features(stdgpu PUBLIC cuda_std_17)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_link_libraries(stdgpu PUBLIC CUDA::cudart)
