
if(DEFINED ROCM_PATH)
    set(STDGPU_ROCM_PATH "${ROCM_PATH}")
elseif(DEFINED ENV{ROCM_PATH})
    set(STDGPU_ROCM_PATH "$ENV{ROCM_PATH}")
else()
    set(STDGPU_ROCM_PATH "/opt/rocm")
endif()

find_package(hip 5.1 REQUIRED
             PATHS
             "${STDGPU_ROCM_PATH}/hip")

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
                                 FILES impl/atomic_detail.h)
endif()

target_compile_features(stdgpu PUBLIC hip_std_17)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)

target_link_libraries(stdgpu PUBLIC hip::host)


# Install custom thrust module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/Findthrust.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
