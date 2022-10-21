cmake_minimum_required(VERSION 3.18...3.24)
message(STATUS "Configuring with CMake ${CMAKE_VERSION}")


project(stdgpu VERSION 1.3.0
               DESCRIPTION "Efficient STL-like Data Structures on the GPU"
               LANGUAGES CXX)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT ON)
else()
    set(STDGPU_SETUP_COMPILER_FLAGS_DEFAULT OFF)
endif()

option(STDGPU_BUILD_SHARED_LIBS "Builds the project as a shared library, if set to ON, or as a static library, if set to OFF, default: BUILD_SHARED_LIBS" ${BUILD_SHARED_LIBS})
option(STDGPU_SETUP_COMPILER_FLAGS "Constructs the compiler flags, default: ON if standalone, OFF if included via add_subdirectory" ${STDGPU_SETUP_COMPILER_FLAGS_DEFAULT})
option(STDGPU_COMPILE_WARNING_AS_ERROR "Treats compiler warnings as errors, default: OFF" OFF)
option(STDGPU_BUILD_EXAMPLES "Build the examples, default: ON" ON)
option(STDGPU_BUILD_BENCHMARKS "Build the benchmarks, default: ON" ON)
option(STDGPU_BUILD_TESTS "Build the unit tests, default: ON" ON)
option(STDGPU_BUILD_TEST_COVERAGE "Build a test coverage report, default: OFF" OFF)
option(STDGPU_ANALYZE_WITH_CLANG_TIDY "Analyzes the code with clang-tidy, default: OFF" OFF)
option(STDGPU_ANALYZE_WITH_CPPCHECK "Analyzes the code with cppcheck, default: OFF" OFF)


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
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(STDGPU_CUDA_ARCHITECTURE_FLAGS_USER ${CMAKE_CUDA_ARCHITECTURES})
    endif()

    enable_language(CUDA)
elseif(STDGPU_BACKEND STREQUAL STDGPU_BACKEND_HIP)
    cmake_minimum_required(VERSION 3.21.3...3.24)

    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set(STDGPU_HIP_ARCHITECTURE_FLAGS_USER ${CMAKE_HIP_ARCHITECTURES})
    endif()

    enable_language(HIP)
endif()


# Backend-specific modules have higher priority than generic modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")


if(STDGPU_SETUP_COMPILER_FLAGS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/${STDGPU_BACKEND_DIRECTORY}/set_device_flags.cmake")
    stdgpu_set_device_flags(STDGPU_DEVICE_FLAGS)
    stdgpu_set_test_device_flags(STDGPU_TEST_DEVICE_FLAGS)
    message(STATUS "Created device flags : ${STDGPU_DEVICE_FLAGS}")
    message(STATUS "Created test device flags : ${STDGPU_TEST_DEVICE_FLAGS}")

    if(STDGPU_BACKEND STREQUAL STDGPU_BACKEND_CUDA)
        if(STDGPU_CUDA_ARCHITECTURE_FLAGS_USER)
            # CMAKE_CUDA_ARCHITECTURES already set by the user
            message(STATUS "Detected user-provided CCs : ${STDGPU_CUDA_ARCHITECTURE_FLAGS_USER}")
        else()
            stdgpu_cuda_set_architecture_flags(STDGPU_CUDA_ARCHITECTURE_FLAGS)
            if(STDGPU_CUDA_ARCHITECTURE_FLAGS)
                set(CMAKE_CUDA_ARCHITECTURES ${STDGPU_CUDA_ARCHITECTURE_FLAGS})
            else()
                message(WARNING "Falling back to default CCs : ${CMAKE_CUDA_ARCHITECTURES}")
            endif()
        endif()

        # Workaround for bug in libstdc++ (see https://gitlab.kitware.com/cmake/cmake/-/merge_requests/4442#note_737136)
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            message(STATUS "Building with disabled CXX extensions")
            set(CMAKE_CXX_EXTENSIONS OFF)
        endif()
    elseif(STDGPU_BACKEND STREQUAL STDGPU_BACKEND_HIP)
        if(STDGPU_HIP_ARCHITECTURE_FLAGS_USER)
            # CMAKE_HIP_ARCHITECTURES already set by the user
            message(STATUS "Detected user-provided CCs : ${STDGPU_HIP_ARCHITECTURE_FLAGS_USER}")
        else()
            # NOTE : Architectures could be detected via the "gcnArchName" property
            message(WARNING "Falling back to default CCs : ${CMAKE_HIP_ARCHITECTURES}")
        endif()
    endif()

    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/set_host_flags.cmake")
    stdgpu_set_host_flags(STDGPU_HOST_FLAGS)
    stdgpu_set_test_host_flags(STDGPU_TEST_HOST_FLAGS)
    message(STATUS "Created host flags : ${STDGPU_HOST_FLAGS}")
    message(STATUS "Created test host flags : ${STDGPU_TEST_HOST_FLAGS}")
endif()


if(STDGPU_BUILD_TESTS AND STDGPU_BUILD_TEST_COVERAGE)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/code_coverage.cmake")
    append_coverage_compiler_flags()
    set(COVERAGE_EXCLUDES '*CMake*' '*benchmark/*' '*examples/*' '*external/*' '*test/*' '/usr/*')
endif()


if(STDGPU_ANALYZE_WITH_CLANG_TIDY)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_clang_tidy.cmake")
    stdgpu_setup_clang_tidy(STDGPU_PROPERTY_CLANG_TIDY)
endif()

if(STDGPU_ANALYZE_WITH_CPPCHECK)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_cppcheck.cmake")
    stdgpu_setup_cppcheck(STDGPU_PROPERTY_CPPCHECK)
endif()


option(STDGPU_ALLOW_NEWER_CLANG_FORMAT_VERSIONS "CAUTION: Allows finding newer versions of clang-format which may produce different results (WILL BE REJECTED), default: OFF" OFF)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_clang_format.cmake")
stdgpu_setup_clang_format()


# Setup output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")


# Setup install paths
include(GNUInstallDirs)
set(STDGPU_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(STDGPU_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(STDGPU_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(STDGPU_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/stdgpu")
set(STDGPU_DOC_INSTALL_DIR "${CMAKE_INSTALL_DOCDIR}/stdgpu")


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


include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_uninstall_target.cmake")


if(STDGPU_BUILD_EXAMPLES)
    enable_testing()
    add_subdirectory(examples)
endif()

if(STDGPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(STDGPU_BUILD_TESTS)
    enable_testing()

    add_subdirectory(test)

    if(STDGPU_BUILD_TEST_COVERAGE)
        setup_target_for_coverage(NAME stdgpu_coverage
                                  EXECUTABLE ${CMAKE_COMMAND} -E chdir .. sh scripts/run_tests.sh ${CMAKE_BUILD_TYPE}
                                  DEPENDENCIES teststdgpu)
    endif()

endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/config_summary.cmake")
stdgpu_print_configuration_summary()
