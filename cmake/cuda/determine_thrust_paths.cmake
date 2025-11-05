function(stdgpu_determine_thrust_paths STDGPU_OUTPUT_THRUST_PATHS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_THRUST_PATHS})

    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13 AND CMAKE_VERSION VERSION_LESS 3.31.9)
        set(STDGPU_CUDATOOLKIT_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

        foreach(dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
            list(APPEND STDGPU_CUDATOOLKIT_INCLUDE_DIRS "${dir}/cccl")
        endforeach()

        set(${STDGPU_OUTPUT_THRUST_PATHS} "${STDGPU_CUDATOOLKIT_INCLUDE_DIRS}")
    else()
        set(${STDGPU_OUTPUT_THRUST_PATHS} "${CUDAToolkit_INCLUDE_DIRS}")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_THRUST_PATHS} ${${STDGPU_OUTPUT_THRUST_PATHS}} PARENT_SCOPE)
endfunction()
