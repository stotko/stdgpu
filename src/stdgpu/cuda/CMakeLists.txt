
find_package(CUDAToolkit 10.0 REQUIRED MODULE)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(CUDAToolkit 10.0 REQUIRED MODULE)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/device.cpp
                              impl/memory.cpp)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_compile_features(stdgpu PUBLIC cuda_std_14)
endif()

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_link_libraries(stdgpu PUBLIC CUDA::cudart)


# Install custom CUDA module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/FindCUDAToolkit.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
