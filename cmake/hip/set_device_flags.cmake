function(stdgpu_set_device_flags STDGPU_OUTPUT_DEVICE_FLAGS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_DEVICE_FLAGS})

    list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wno-unused-command-line-argument")
    list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wno-pass-failed")

    set(${STDGPU_OUTPUT_DEVICE_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:${${STDGPU_OUTPUT_DEVICE_FLAGS}}>")

    # Make output variable visible
    set(${STDGPU_OUTPUT_DEVICE_FLAGS} ${${STDGPU_OUTPUT_DEVICE_FLAGS}} PARENT_SCOPE)
endfunction()


# Auxiliary compiler flags for tests to be used with target_compile_options
function(stdgpu_set_test_device_flags STDGPU_OUTPUT_DEVICE_TEST_FLAGS)
    # No flags required
endfunction()
