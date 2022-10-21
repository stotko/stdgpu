function(stdgpu_set_host_flags STDGPU_OUTPUT_HOST_FLAGS)
    # Clear list before appending flags
    unset(${STDGPU_OUTPUT_HOST_FLAGS})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wall")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-pedantic")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wextra")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wshadow")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wsign-compare")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wconversion")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wfloat-equal")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wundef")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Wdouble-promotion")

        if(STDGPU_COMPILE_WARNING_AS_ERROR AND CMAKE_VERSION VERSION_LESS 3.24)
            list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-Werror")
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
            list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "-O3")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "/W2") # or /W3 or /W4 depending on how useful this is

        if(STDGPU_COMPILE_WARNING_AS_ERROR AND CMAKE_VERSION VERSION_LESS 3.24)
            list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "/WX")
        endif()

        #list(APPEND ${STDGPU_OUTPUT_HOST_FLAGS} "/O2")
    endif()

    set(${STDGPU_OUTPUT_HOST_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:${${STDGPU_OUTPUT_HOST_FLAGS}}>")

    # Make output variable visible
    set(${STDGPU_OUTPUT_HOST_FLAGS} ${${STDGPU_OUTPUT_HOST_FLAGS}} PARENT_SCOPE)
endfunction()


# Auxiliary compiler flags for tests to be used with target_compile_options
function(stdgpu_set_test_host_flags STDGPU_OUTPUT_HOST_TEST_FLAGS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(${STDGPU_OUTPUT_HOST_TEST_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(${STDGPU_OUTPUT_HOST_TEST_FLAGS} "$<$<COMPILE_LANGUAGE:CXX>:/wd4996>")
    endif()

    # Make output variable visible
    set(${STDGPU_OUTPUT_HOST_TEST_FLAGS} ${${STDGPU_OUTPUT_HOST_TEST_FLAGS}} PARENT_SCOPE)
endfunction()
