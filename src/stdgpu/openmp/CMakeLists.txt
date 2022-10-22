
find_package(OpenMP 2.0 REQUIRED)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(OpenMP 2.0 REQUIRED)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/device.cpp
                              impl/memory.cpp)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(stdgpu PUBLIC FILE_SET stdgpu_backend_headers
                                 TYPE HEADERS
                                 BASE_DIRS ${STDGPU_INCLUDE_LOCAL_DIR}
                                 FILES atomic.h
                                       device.h
                                       memory.h
                                       platform.h
                                       platform_check.h)

    target_sources(stdgpu PUBLIC FILE_SET stdgpu_backend_header_implementations
                                 TYPE HEADERS
                                 BASE_DIRS ${STDGPU_INCLUDE_LOCAL_DIR}
                                 FILES impl/atomic_detail.h)
endif()

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)

target_link_libraries(stdgpu PUBLIC OpenMP::OpenMP_CXX)
