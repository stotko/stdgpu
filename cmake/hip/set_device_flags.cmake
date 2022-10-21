function(stdgpu_set_device_flags STDGPU_OUTPUT_DEVICE_FLAGS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_DEVICE_FLAGS})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wall")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wextra")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wshadow")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wsign-compare")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wconversion")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wfloat-equal")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wundef")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wdouble-promotion")

        if(STDGPU_COMPILE_WARNING_AS_ERROR AND CMAKE_VERSION VERSION_LESS 3.24)
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Werror")
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-O3")
        endif()
    endif()

    if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wno-pass-failed")
    endif()

    set(${STDGPU_OUTPUT_DEVICE_FLAGS} "$<$<COMPILE_LANGUAGE:HIP>:${${STDGPU_OUTPUT_DEVICE_FLAGS}}>")

    # Make output variable visible
    set(${STDGPU_OUTPUT_DEVICE_FLAGS} ${${STDGPU_OUTPUT_DEVICE_FLAGS}} PARENT_SCOPE)
endfunction()


# Auxiliary compiler flags for tests to be used with target_compile_options
function(stdgpu_set_test_device_flags STDGPU_OUTPUT_DEVICE_TEST_FLAGS)
    # No flags required
endfunction()
