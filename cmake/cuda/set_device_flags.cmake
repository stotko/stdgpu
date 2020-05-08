function(stdgpu_set_device_flags STDGPU_OUTPUT_DEVICE_FLAGS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_DEVICE_FLAGS})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wall")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wextra")
        # Enabled only for CMake 3.17+ due to thrust and a bug in CMake (fixed in 3.17+)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wshadow")
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wsign-compare")
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wconversion")
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Wfloat-equal")
        endif()

        if(STDGPU_TREAT_WARNINGS_AS_ERRORS)
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "-Werror")
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
            message(STATUS "Appended optimization flag (-O3,/O2) implicitly")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "/W2") # or /W3 or /W4 depending on how useful this is

        if(STDGPU_TREAT_WARNINGS_AS_ERRORS)
            list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "/WX")
        endif()

        #list(APPEND ${STDGPU_OUTPUT_DEVICE_FLAGS} "/O2")
    endif()

    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        string(REPLACE ";" "," ${STDGPU_OUTPUT_DEVICE_FLAGS} "${${STDGPU_OUTPUT_DEVICE_FLAGS}}")
        set(${STDGPU_OUTPUT_DEVICE_FLAGS} "-Xcompiler=${${STDGPU_OUTPUT_DEVICE_FLAGS}}")
    endif()

    set(${STDGPU_OUTPUT_DEVICE_FLAGS} "$<$<COMPILE_LANGUAGE:CUDA>:${${STDGPU_OUTPUT_DEVICE_FLAGS}}>")

    # Make output variable visible
    set(${STDGPU_OUTPUT_DEVICE_FLAGS} ${${STDGPU_OUTPUT_DEVICE_FLAGS}} PARENT_SCOPE)
endfunction()


# Auxiliary compiler flags for tests to be used with target_compile_options
function(stdgpu_set_test_device_flags STDGPU_OUTPUT_DEVICE_TEST_FLAGS)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(${STDGPU_OUTPUT_DEVICE_TEST_FLAGS} "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_DEVICE_TEST_FLAGS} ${${STDGPU_OUTPUT_DEVICE_TEST_FLAGS}} PARENT_SCOPE)
endfunction()


function(stdgpu_cuda_set_architecture_flags STDGPU_OUTPUT_DEVICE_COMPILE_AND_LINK_FLAGS)
    # NOTE Available in CMake 3.17+
    # include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/check_compute_capability.cmake")
    include("${stdgpu_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/check_compute_capability.cmake")

    set(STDGPU_CUDA_HAVE_SUITABLE_GPU FALSE)

    foreach(STDGPU_CUDA_CC IN LISTS STDGPU_CUDA_COMPUTE_CAPABILITIES)
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            string(APPEND ${STDGPU_OUTPUT_DEVICE_COMPILE_AND_LINK_FLAGS} " --generate-code arch=compute_${STDGPU_CUDA_CC},code=sm_${STDGPU_CUDA_CC}")
            message(STATUS "Enabled compilation for CC ${STDGPU_CUDA_CC}")
            set(STDGPU_CUDA_HAVE_SUITABLE_GPU TRUE)
        endif()
    endforeach()

    if(NOT STDGPU_CUDA_HAVE_SUITABLE_GPU)
        message(FATAL_ERROR "No CUDA-capable GPU detected")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_DEVICE_COMPILE_AND_LINK_FLAGS} ${${STDGPU_OUTPUT_DEVICE_COMPILE_AND_LINK_FLAGS}} PARENT_SCOPE)
endfunction()
