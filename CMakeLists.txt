cmake_minimum_required(VERSION 3.15...3.16)


project(stdgpu VERSION 1.2.0
               DESCRIPTION "Efficient STL-like Data Structures on the GPU"
               LANGUAGES CXX)


set(STDGPU_BACKEND_CUDA "STDGPU_BACKEND_CUDA")
set(STDGPU_BACKEND_OPENMP "STDGPU_BACKEND_OPENMP")
set(STDGPU_BACKEND_OPTIONS ${STDGPU_BACKEND_CUDA} ${STDGPU_BACKEND_OPENMP})

set(STDGPU_BACKEND ${STDGPU_BACKEND_CUDA} CACHE STRING "Device system backend, default: STDGPU_BACKEND_CUDA")
if (NOT STDGPU_BACKEND IN_LIST STDGPU_BACKEND_OPTIONS)
    message(FATAL_ERROR "STDGPU_BACKEND is set to \"${STDGPU_BACKEND}\", but must be one of \"${STDGPU_BACKEND_OPTIONS}\"")
endif()

set(STDGPU_BACKEND_DIRECTORY ${STDGPU_BACKEND})
string(REGEX REPLACE "^STDGPU_BACKEND_" "" STDGPU_BACKEND_DIRECTORY ${STDGPU_BACKEND_DIRECTORY})
string(TOLOWER ${STDGPU_BACKEND_DIRECTORY} STDGPU_BACKEND_DIRECTORY)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}")


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT ON)
else()
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT OFF)
endif()
option(STDGPU_SETUP_COMPILER_FLAGS "Constructs the compiler flags, default: ON if standalone, OFF if included via add_subdirectory" ${STDGPU_SETUP_COMPILER_FLAGS_DEFAULT})

if(STDGPU_SETUP_COMPILER_FLAGS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/set_device_flags.cmake")
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
option(STDGPU_BUILD_TEST_COVERAGE "Build a test coverage report, default: OFF" OFF)
if(STDGPU_BUILD_TESTS)
    enable_testing()

    if(STDGPU_BUILD_TEST_COVERAGE)
        include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/code_coverage.cmake")
        append_coverage_compiler_flags()
        set(COVERAGE_EXCLUDES '*CMake*' '*external/*' '*test/*' '/usr/*')
    endif()

    add_subdirectory(test)

    if(STDGPU_BUILD_TEST_COVERAGE)
        string(TOLOWER ${CMAKE_BUILD_TYPE} STDGPU_SCRIPT_BUILD_TYPE)
        setup_target_for_coverage(NAME stdgpu_coverage
                                  EXECUTABLE ${CMAKE_COMMAND} -E chdir .. sh scripts/run_tests_${STDGPU_SCRIPT_BUILD_TYPE}.sh
                                  DEPENDENCIES teststdgpu)
    endif()

endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/config_summary.cmake")
stdgpu_print_configuration_summary()

