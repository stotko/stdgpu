
# NOTE For version checking only
find_package(hip 3.1 REQUIRED)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(hip 3.1 REQUIRED)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cpp
                              impl/mutex.cpp)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)

target_link_options(stdgpu PUBLIC ${STDGPU_DEVICE_LINK_FLAGS})


# Install custom thrust module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/Findthrust.cmake"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
