function(stdgpu_determine_thrust_paths STDGPU_OUTPUT_THRUST_PATHS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_THRUST_PATHS})

    if(DEFINED ROCM_PATH)
        set(STDGPU_ROCM_PATH "${ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(STDGPU_ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(STDGPU_ROCM_PATH "/opt/rocm")
    endif()

    set(${STDGPU_OUTPUT_THRUST_PATHS} "${STDGPU_ROCM_PATH}/include")

    # Make output variable visible
    set(${STDGPU_OUTPUT_THRUST_PATHS} ${${STDGPU_OUTPUT_THRUST_PATHS}} PARENT_SCOPE)
endfunction()
