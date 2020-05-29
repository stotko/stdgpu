cmake_minimum_required(VERSION 3.15...3.17)


project(stdgpu VERSION 1.2.0
               DESCRIPTION "Efficient STL-like Data Structures on the GPU"
               LANGUAGES CXX)


set(STDGPU_BACKEND_CUDA "STDGPU_BACKEND_CUDA")
set(STDGPU_BACKEND_OPENMP "STDGPU_BACKEND_OPENMP")
set(STDGPU_BACKEND_HIP "STDGPU_BACKEND_HIP")
list(APPEND STDGPU_BACKEND_OPTIONS ${STDGPU_BACKEND_CUDA} ${STDGPU_BACKEND_OPENMP} ${STDGPU_BACKEND_HIP})

# STDGPU_BACKEND
set(STDGPU_BACKEND ${STDGPU_BACKEND_CUDA} CACHE STRING "Device system backend, default: STDGPU_BACKEND_CUDA")
if(NOT STDGPU_BACKEND IN_LIST STDGPU_BACKEND_OPTIONS)
    message(FATAL_ERROR "STDGPU_BACKEND is set to \"${STDGPU_BACKEND}\", but must be one of \"${STDGPU_BACKEND_OPTIONS}\"")
endif()

# STDGPU_BACKEND_DIRECTORY
set(STDGPU_BACKEND_DIRECTORY ${STDGPU_BACKEND})
string(REGEX REPLACE "^STDGPU_BACKEND_" "" STDGPU_BACKEND_DIRECTORY ${STDGPU_BACKEND_DIRECTORY})
string(TOLOWER ${STDGPU_BACKEND_DIRECTORY} STDGPU_BACKEND_DIRECTORY)

# STDGPU_BACKEND_NAMESPACE
set(STDGPU_BACKEND_NAMESPACE ${STDGPU_BACKEND_DIRECTORY})

# STDGPU_BACKEND_MACRO_NAMESPACE
set(STDGPU_BACKEND_MACRO_NAMESPACE ${STDGPU_BACKEND_NAMESPACE})
string(TOUPPER ${STDGPU_BACKEND_MACRO_NAMESPACE} STDGPU_BACKEND_MACRO_NAMESPACE)


# Enable backend-specific languages
if(STDGPU_BACKEND STREQUAL STDGPU_BACKEND_CUDA)
    enable_language(CUDA)
endif()


# Backend-specific modules have higher priority than generic modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT ON)
else()
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT OFF)
endif()
option(STDGPU_SETUP_COMPILER_FLAGS "Constructs the compiler flags, default: ON if standalone, OFF if included via add_subdirectory" ${STDGPU_SETUP_COMPILER_FLAGS_DEFAULT})
option(STDGPU_TREAT_WARNINGS_AS_ERRORS "Treats compiler warnings as errors, default: OFF" OFF)

if(STDGPU_SETUP_COMPILER_FLAGS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/set_device_flags.cmake")
    stdgpu_set_device_flags(STDGPU_DEVICE_FLAGS)
    stdgpu_set_test_device_flags(STDGPU_TEST_DEVICE_FLAGS)
    message(STATUS "Created device flags : ${STDGPU_DEVICE_FLAGS}")
    message(STATUS "Created test device flags : ${STDGPU_TEST_DEVICE_FLAGS}")

    if(STDGPU_BACKEND STREQUAL STDGPU_BACKEND_CUDA)
        # FIXME Use the architecture flags for both device compiling and device linking until there is a proper abstraction
        stdgpu_cuda_set_architecture_flags(STDGPU_DEVICE_COMPILE_AND_LINK_FLAGS)
        string(APPEND CMAKE_CUDA_FLAGS "${STDGPU_DEVICE_COMPILE_AND_LINK_FLAGS}")
        message(STATUS "Building with modified CMAKE_CUDA_FLAGS : ${CMAKE_CUDA_FLAGS}")
    endif()

    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/set_host_flags.cmake")
    stdgpu_set_host_flags(STDGPU_HOST_FLAGS)
    stdgpu_set_test_host_flags(STDGPU_TEST_HOST_FLAGS)
    message(STATUS "Created host flags : ${STDGPU_HOST_FLAGS}")
    message(STATUS "Created test host flags : ${STDGPU_TEST_HOST_FLAGS}")
endif()


option(STDGPU_ANALYZE_WITH_CLANG_TIDY "Analyzes the code with clang-tidy, default: OFF" OFF)
if(STDGPU_ANALYZE_WITH_CLANG_TIDY)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_clang_tidy.cmake")
    stdgpu_setup_clang_tidy(STDGPU_PROPERTY_CLANG_TIDY)
endif()

option(STDGPU_ANALYZE_WITH_CPPCHECK "Analyzes the code with cppcheck, default: OFF" OFF)
if(STDGPU_ANALYZE_WITH_CPPCHECK)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_cppcheck.cmake")
    stdgpu_setup_cppcheck(STDGPU_PROPERTY_CPPCHECK)
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
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}"
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
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}"
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

