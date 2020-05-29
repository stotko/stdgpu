
# NOTE For version checking only
find_package(hip 3.1 REQUIRED)

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(hip 3.1 REQUIRED)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp
                              impl/bitset.cpp
                              impl/mutex.cpp)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)

include("${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/set_device_flags.cmake")
# NOTE HIP architecture flags must be passed as device link flags
stdgpu_hip_set_architecture_flags(STDGPU_DEVICE_LINK_FLAGS)
message(STATUS "Created HIP device link flags : ${STDGPU_DEVICE_LINK_FLAGS}")

target_link_options(stdgpu PUBLIC ${STDGPU_DEVICE_LINK_FLAGS})


# Install custom thrust module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/Findthrust.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
