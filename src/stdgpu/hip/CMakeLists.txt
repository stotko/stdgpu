find_package(hip 5.1 REQUIRED)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(hip 5.1 REQUIRED)
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
                                 FILES impl/atomic_detail.h
                                       impl/error.h)
endif()

target_compile_features(stdgpu PUBLIC hip_std_17)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)

# CMake always links libraries and executables involving HIP with the device linker, even if only host code is used.
# Suppress the linker warning about unused HIP architectures *globally* via a public linker flag.
set(STDGPU_HIP_DEVICE_LINKER_FLAGS "$<$<LINK_LANGUAGE:HIP>:-Wno-unused-command-line-argument>")
message(STATUS "Created HIP device linker flags : ${STDGPU_HIP_DEVICE_LINKER_FLAGS}")
target_link_options(stdgpu PUBLIC "${STDGPU_HIP_DEVICE_LINKER_FLAGS}")

target_link_libraries(stdgpu PUBLIC hip::host)
