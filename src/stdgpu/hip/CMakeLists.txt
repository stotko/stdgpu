#find_package(hip 7.1 REQUIRED)

#set(STDGPU_DEPENDENCIES_BACKEND_INIT "
#find_dependency(hip 7.1 REQUIRED)
#" PARENT_SCOPE)

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
                                       impl/error.h
                                       impl/memory_detail.h)
endif()

target_compile_features(stdgpu PUBLIC hip_std_17)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)

# thrust (rocthrust) already links against hip::device, which in turn links against hip::host
#target_link_libraries(stdgpu PUBLIC hip::host)


# Install custom thrust module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/Findthrust.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
