
enable_language(CUDA)

include("${CMAKE_CURRENT_LIST_DIR}/check_compute_capability.cmake")

# Minimum CC : Determined by used features, limits CUDA version at EOL
set(STDGPU_CUDA_MIN_CC_MAJOR 3)
set(STDGPU_CUDA_MIN_CC_MINOR 5)
set(STDGPU_CUDA_MIN_CC ${STDGPU_CUDA_MIN_CC_MAJOR}${STDGPU_CUDA_MIN_CC_MINOR})

# Maximum CC : Determined by minimum CUDA version
set(STDGPU_CUDA_MAX_CC_MAJOR 7)
set(STDGPU_CUDA_MAX_CC_MINOR 5)
set(STDGPU_CUDA_MAX_CC ${STDGPU_CUDA_MAX_CC_MAJOR}${STDGPU_CUDA_MAX_CC_MINOR})

message(STATUS "CUDA Compute Capability (CC) Configuration")
message(STATUS "  Minimum required CC  : ${STDGPU_CUDA_MIN_CC}")
message(STATUS "  Maximum supported CC : ${STDGPU_CUDA_MAX_CC} (newer supported via JIT compilation)")
set(STDGPU_CUDA_HAVE_SUITABLE_GPU FALSE)

foreach(STDGPU_CUDA_CC IN LISTS STDGPU_CUDA_COMPUTE_CAPABILITIES)
    if(${STDGPU_CUDA_CC} LESS ${STDGPU_CUDA_MIN_CC})
        # STDGPU_CUDA_CC < STDGPU_CUDA_MIN_CC
        message(STATUS "  Skip compilation for CC ${STDGPU_CUDA_CC} which is too old")
    elseif(NOT ${STDGPU_CUDA_CC} GREATER ${STDGPU_CUDA_MAX_CC})
        # STDGPU_CUDA_MIN_CC <= STDGPU_CUDA_CC <= STDGPU_CUDA_MAX_CC
        string(APPEND STDGPU_DEVICE_FLAGS " --generate-code arch=compute_${STDGPU_CUDA_CC},code=sm_${STDGPU_CUDA_CC}")
        message(STATUS "  Enabled compilation for CC ${STDGPU_CUDA_CC}")
        set(STDGPU_CUDA_HAVE_SUITABLE_GPU TRUE)
    else()
        # STDGPU_CUDA_MAX_CC < STDGPU_CUDA_CC
        string(APPEND STDGPU_DEVICE_FLAGS " --generate-code arch=compute_${STDGPU_CUDA_MAX_CC},code=compute_${STDGPU_CUDA_MAX_CC}")
        message(STATUS "  Enabled compilation for CC ${STDGPU_CUDA_CC} via JIT compilation of ${STDGPU_CUDA_MAX_CC}")
        set(STDGPU_CUDA_HAVE_SUITABLE_GPU TRUE)
    endif()
endforeach()

if(NOT STDGPU_CUDA_HAVE_SUITABLE_GPU)
    message(FATAL_ERROR "  No CUDA-capable GPU with at least CC ${STDGPU_CUDA_MIN_CC} detected")
endif()

if(NOT MSVC)
    string(APPEND STDGPU_DEVICE_FLAGS " -Xcompiler -Wall")
    string(APPEND STDGPU_DEVICE_FLAGS " -Xcompiler -Wextra")
    #string(APPEND STDGPU_DEVICE_FLAGS " -Xcompiler -Wshadow") # Currently disabled due to thrust

    if(${CMAKE_BUILD_TYPE} MATCHES "Release" OR ${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
        message(STATUS "Appended optimization flag (-O3,/O2) implicitly")
    endif()
else()
    #string(APPEND STDGPU_DEVICE_FLAGS " -Xcompiler /W3") # or /W4 depending on how useful this is
    #string(APPEND STDGPU_DEVICE_FLAGS " /O2")
endif()

# Apply compiler flags
string(APPEND CMAKE_CUDA_FLAGS ${STDGPU_DEVICE_FLAGS})

message(STATUS "Created device flags : ${STDGPU_DEVICE_FLAGS}")
message(STATUS "Building with CUDA flags : ${CMAKE_CUDA_FLAGS}")

# Auxiliary compiler flags for tests to be used with target_compile_options
if(NOT MSVC)
    set(STDGPU_TEST_DEVICE_FLAGS "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>")
endif()
