
target_sources(stdgpu PRIVATE impl/memory.cpp)

target_include_directories(stdgpu PUBLIC
                                  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
