
# NOTE For version checking only
find_package(hip 3.5 REQUIRED)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(hip 3.5 REQUIRED)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)


# Install custom thrust module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/Findthrust.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
