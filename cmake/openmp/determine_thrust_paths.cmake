function(stdgpu_determine_thrust_paths STDGPU_OUTPUT_THRUST_PATHS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_THRUST_PATHS})

    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        list(APPEND ${STDGPU_OUTPUT_THRUST_PATHS} "${CUDAToolkit_INCLUDE_DIRS}")
    endif()
    list(APPEND "/usr/include")
    list(APPEND "/usr/local/include")

    # Make output variable visible
    set(${STDGPU_OUTPUT_THRUST_PATHS} ${${STDGPU_OUTPUT_THRUST_PATHS}} PARENT_SCOPE)
endfunction()
