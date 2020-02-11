
find_package(CUDAToolkit 10.0 REQUIRED MODULE)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
list(PREPEND CMAKE_MODULE_PATH \${CMAKE_CURRENT_LIST_DIR})
find_dependency(CUDAToolkit 10.0 REQUIRED MODULE)
list(POP_FRONT CMAKE_MODULE_PATH)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cu
                              impl/mutex.cu)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_link_libraries(stdgpu PUBLIC CUDA::cudart)


# Install custom CUDA module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/FindCUDAToolkit.cmake"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_CMAKE_INSTALL_DIR}"
        COMPONENT stdgpu)
