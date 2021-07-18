
if(DEFINED ROCM_PATH)
    set(STDGPU_ROCM_PATH "${ROCM_PATH}")
elseif(DEFINED ENV{ROCM_PATH})
    set(STDGPU_ROCM_PATH "$ENV{ROCM_PATH}")
else()
    set(STDGPU_ROCM_PATH "/opt/rocm")
endif()

find_package(hip 4.2 REQUIRED
             PATHS
             "${STDGPU_ROCM_PATH}/hip")

set(STDGPU_DEPENDENCIES_BACKEND_INIT "
find_dependency(hip 4.2 REQUIRED)
" PARENT_SCOPE)

target_sources(stdgpu PRIVATE impl/memory.cpp)

target_compile_features(stdgpu PUBLIC hip_std_14)

target_compile_definitions(stdgpu PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)

target_link_libraries(stdgpu PUBLIC hip::host)


# Install custom thrust module
install(FILES "${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/Findthrust.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}/${STDGPU_BACKEND_DIRECTORY}"
        COMPONENT stdgpu)
