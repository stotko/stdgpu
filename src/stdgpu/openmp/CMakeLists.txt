
find_package(OpenMP 2.0 REQUIRED)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(OpenMP 2.0 REQUIRED)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cpp
                              impl/mutex.cpp)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

target_link_libraries(stdgpu PUBLIC OpenMP::OpenMP_CXX)
