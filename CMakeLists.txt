cmake_minimum_required(VERSION 3.13...3.15)


project(stdgpu VERSION 1.0.0
               DESCRIPTION "Efficient STL-like Data Structures on the GPU"
               LANGUAGES CXX CUDA)


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT ON)
else()
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT OFF)
endif()
option(STDGPU_SETUP_COMPILER_FLAGS "Constructs the compiler flags, default: ON if standalone, OFF if included via add_subdirectory" ${STDGPU_SETUP_COMPILER_FLAGS_DEFAULT})

if(STDGPU_SETUP_COMPILER_FLAGS)
    # Device flags (.cu files)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_compute_capability.cmake")

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
        if(${CMAKE_BUILD_TYPE} MATCHES "Release" OR ${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
            message(STATUS "Appended optimization flag (-O3,/O2) implicitly")
        else()
            string(APPEND STDGPU_DEVICE_FLAGS " -O3")
        endif()
    else()
        #string(APPEND STDGPU_DEVICE_FLAGS " -Xcompiler /W3") # or /W4 depending on how useful this is
        #string(APPEND STDGPU_DEVICE_FLAGS " /O2")
    endif()
    message(STATUS "Created device flags : ${STDGPU_DEVICE_FLAGS}")


    # Host flags (.cpp files)
    if(NOT MSVC)
        string(APPEND STDGPU_HOST_FLAGS " -Wall")
        string(APPEND STDGPU_HOST_FLAGS " -pedantic")
        string(APPEND STDGPU_HOST_FLAGS " -Wextra")
        string(APPEND STDGPU_HOST_FLAGS " -O3")
    else()
        #string(APPEND STDGPU_HOST_FLAGS " /W3") # or /W4 depending on how useful this is
        #string(APPEND STDGPU_HOST_FLAGS " /O2")
    endif()
    message(STATUS "Created  host  flags : ${STDGPU_HOST_FLAGS}")


    # Apply compiler flags
    string(APPEND CMAKE_CUDA_FLAGS ${STDGPU_DEVICE_FLAGS})
    string(APPEND CMAKE_CXX_FLAGS ${STDGPU_HOST_FLAGS})

    message(STATUS "Building with CUDA flags : ${CMAKE_CUDA_FLAGS}")
    message(STATUS "Building with CXX  flags : ${CMAKE_CXX_FLAGS}")
endif()


# Setup install paths
set(STDGPU_LIB_INSTALL_DIR "lib")
set(STDGPU_BIN_INSTALL_DIR "bin")
set(STDGPU_INCLUDE_INSTALL_DIR "include")
set(STDGPU_CMAKE_INSTALL_DIR "lib/cmake/stdgpu")
set(STDGPU_DOC_INSTALL_DIR "doc/stdgpu")

set(STDGPU_INSTALL_DESTINATIONS
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_BIN_INSTALL_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_LIB_INSTALL_DIR}" COMPONENT shlib
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_LIB_INSTALL_DIR}" COMPONENT lib)


add_subdirectory(src)

add_subdirectory(doc)


# Install exported targets and cmake files
install(EXPORT stdgpu-targets
        NAMESPACE stdgpu::
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_CMAKE_INSTALL_DIR}"
        COMPONENT stdgpu)

include(CMakePackageConfigHelpers)
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/stdgpu-config.cmake.in"
                              "${CMAKE_CURRENT_BINARY_DIR}/stdgpu-config.cmake"
                              INSTALL_DESTINATION ${STDGPU_CMAKE_INSTALL_DIR}
                              PATH_VARS STDGPU_INCLUDE_INSTALL_DIR)

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/stdgpu-config-version.cmake"
                                 VERSION ${stdgpu_VERSION}
                                 COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stdgpu-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/stdgpu-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_CMAKE_INSTALL_DIR}"
        COMPONENT stdgpu)


option(STDGPU_BUILD_EXAMPLES "Build the examples, default: ON" ON)
if(STDGPU_BUILD_EXAMPLES)
    enable_testing()
    add_subdirectory(examples)
endif()

option(STDGPU_BUILD_TESTS "Build the unit tests, default: ON" ON)
if(STDGPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/config_summary.cmake")
stdgpu_print_configuration_summary()

