
set(STDGPU_DEPENDENCIES_BACKEND_INIT "
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cu
                              impl/mutex.cu)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_include_directories(stdgpu PUBLIC
                                  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
