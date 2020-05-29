
find_package(CUDAToolkit 10.0 REQUIRED MODULE)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(CUDAToolkit 10.0 REQUIRED MODULE)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cu
                              impl/mutex.cu)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_link_libraries(stdgpu PUBLIC CUDA::cudart)


# Install custom CUDA module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/FindCUDAToolkit.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
