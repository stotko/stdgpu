
target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cu
                              impl/mutex.cu)

target_include_directories(stdgpu PUBLIC
                                  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
