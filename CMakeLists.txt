cmake_minimum_required(VERSION 3.13...3.15)


project(stdgpu VERSION 1.1.0
               DESCRIPTION "Efficient STL-like Data Structures on the GPU"
               LANGUAGES CXX CUDA)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT ON)
else()
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT OFF)
endif()
option(STDGPU_SETUP_COMPILER_FLAGS "Constructs the compiler flags, default: ON if standalone, OFF if included via add_subdirectory" ${STDGPU_SETUP_COMPILER_FLAGS_DEFAULT})

if(STDGPU_SETUP_COMPILER_FLAGS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda/set_device_flags.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/set_host_flags.cmake")
endif()


# Setup install paths
set(STDGPU_LIB_INSTALL_DIR "lib")
set(STDGPU_BIN_INSTALL_DIR "bin")
set(STDGPU_INCLUDE_INSTALL_DIR "include")
set(STDGPU_CMAKE_INSTALL_DIR "lib/cmake/stdgpu")
set(STDGPU_DOC_INSTALL_DIR "doc/stdgpu")

option(STDGPU_BUILD_SHARED_LIBS "Builds the project as a shared library, if set to ON, or as a static library, if set to OFF, default: BUILD_SHARED_LIBS" ${BUILD_SHARED_LIBS})

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

