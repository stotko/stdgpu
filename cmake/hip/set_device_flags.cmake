function(stdgpu_set_device_flags STDGPU_OUTPUT_DEVICE_FLAGS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_DEVICE_FLAGS})

    set(STDGPU_HIP_HCC_ID "Clang")
    if(CMAKE_CXX_COMPILER_ID STREQUAL STDGPU_HIP_HCC_ID)
        # FIXME These are needed to suppress some weird warnings/errors
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wno-invalid-noreturn")
    endif()

    set(${STDGPU_OUTPUT_DEVICE_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:${${STDGPU_OUTPUT_DEVICE_FLAGS}}>")

    # Make output variable visible
    set(${STDGPU_OUTPUT_DEVICE_FLAGS} ${${STDGPU_OUTPUT_DEVICE_FLAGS}} PARENT_SCOPE)
endfunction()


# Auxiliary compiler flags for tests to be used with target_compile_options
function(stdgpu_set_test_device_flags STDGPU_OUTPUT_DEVICE_TEST_FLAGS)
    # No flags required
endfunction()


function(stdgpu_hip_set_architecture_flags STDGPU_OUTPUT_DEVICE_LINK_FLAGS)
    # NOTE ROCm can auto-detect the GPU, so the list below is overly pessimistic.
    # NOTE However, this list of hard-coded targets allows cross-compilation and compilation on systems without AMD GPUs.
    list(APPEND STDGPU_HIP_AMDGPU_TARGETS "gfx803" "gfx900" "gfx906" "gfx908")

    set(STDGPU_HIP_HAVE_SUITABLE_GPU FALSE)

    foreach(STDGPU_HIP_TARGET IN LISTS STDGPU_HIP_AMDGPU_TARGETS)
        set(STDGPU_HIP_HCC_ID "Clang")
        if(CMAKE_CXX_COMPILER_ID STREQUAL STDGPU_HIP_HCC_ID)
            list(APPEND ${STDGPU_OUTPUT_DEVICE_LINK_FLAGS} "--amdgpu-target=${STDGPU_HIP_TARGET}")
            message(STATUS "  Enabled compilation for AMDGPU target ${STDGPU_HIP_TARGET}")
            set(STDGPU_HIP_HAVE_SUITABLE_GPU TRUE)
        endif()
    endforeach()

    if(NOT STDGPU_HIP_HAVE_SUITABLE_GPU)
        message(FATAL_ERROR "  No HIP-capable GPU detected")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_DEVICE_LINK_FLAGS} ${${STDGPU_OUTPUT_DEVICE_LINK_FLAGS}} PARENT_SCOPE)
endfunction()
