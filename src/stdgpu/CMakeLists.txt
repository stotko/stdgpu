
set(STDGPU_INCLUDE_LOCAL_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(STDGPU_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
set(STDGPU_BUILD_CMAKE_DIR "${CMAKE_CURRENT_BINARY_DIR}/cmake")

# Setup dependencies
find_package(thrust 1.9.9 REQUIRED MODULE)

set(STDGPU_DEPENDENCIES_INIT "
find_dependency(thrust 1.9.9 REQUIRED MODULE)
")


# Setup library options and config file
if(CMAKE_BUILD_TYPE MATCHES "Release" OR CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
    set(STDGPU_ENABLE_CONTRACT_CHECKS_DEFAULT OFF)
else()
    set(STDGPU_ENABLE_CONTRACT_CHECKS_DEFAULT ON)
endif()

option(STDGPU_ENABLE_CONTRACT_CHECKS "Enable contract checks, default: OFF if CMAKE_BUILD_TYPE is Release or MinSizeRel, ON otherwise" ${STDGPU_ENABLE_CONTRACT_CHECKS_DEFAULT})
option(STDGPU_USE_32_BIT_INDEX "Use 32-bit instead of 64-bit signed integer for index_t, default: ON" ON)

configure_file("${STDGPU_INCLUDE_LOCAL_DIR}/stdgpu/config.h.in"
               "${STDGPU_BUILD_INCLUDE_DIR}/stdgpu/config.h"
               @ONLY)


if(STDGPU_BUILD_SHARED_LIBS)
    add_library(stdgpu SHARED)
else()
    add_library(stdgpu STATIC)
endif()

target_sources(stdgpu PRIVATE impl/device.cpp
                              impl/iterator.cpp
                              impl/memory.cpp)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(stdgpu PUBLIC FILE_SET stdgpu_headers
                                 TYPE HEADERS
                                 BASE_DIRS ${STDGPU_INCLUDE_LOCAL_DIR}
                                 FILES algorithm.h
                                       atomic.cuh
                                       atomic_fwd
                                       bit.h
                                       bitset.cuh
                                       bitset_fwd
                                       cmath.h
                                       compiler.h
                                       contract.h
                                       cstddef.h
                                       deque.cuh
                                       deque_fwd
                                       device.h
                                       execution.h
                                       functional.h
                                       iterator.h
                                       limits.h
                                       memory.h
                                       mutex.cuh
                                       mutex_fwd
                                       numeric.h
                                       platform.h
                                       queue.cuh
                                       queue_fwd
                                       ranges.h
                                       stack.cuh
                                       stack_fwd
                                       unordered_map.cuh
                                       unordered_map_fwd
                                       unordered_set.cuh
                                       unordered_set_fwd
                                       utility.h
                                       vector.cuh
                                       vector_fwd)

    target_sources(stdgpu PUBLIC FILE_SET stdgpu_header_implementations
                                 TYPE HEADERS
                                 BASE_DIRS ${STDGPU_INCLUDE_LOCAL_DIR}
                                 FILES impl/algorithm_detail.h
                                       impl/atomic_detail.cuh
                                       impl/bit_detail.h
                                       impl/bitset_detail.cuh
                                       impl/cmath_detail.h
                                       impl/deque_detail.cuh
                                       impl/functional_detail.h
                                       impl/iterator_detail.h
                                       impl/limits_detail.h
                                       impl/memory_detail.h
                                       impl/mutex_detail.cuh
                                       impl/numeric_detail.h
                                       impl/platform_check.h
                                       impl/preprocessor.h
                                       impl/queue_detail.cuh
                                       impl/ranges_detail.h
                                       impl/stack_detail.cuh
                                       impl/type_traits.h
                                       impl/unordered_base.cuh
                                       impl/unordered_base_detail.cuh
                                       impl/unordered_map_detail.cuh
                                       impl/unordered_set_detail.cuh
                                       impl/utility_detail.h
                                       impl/vector_detail.cuh)

    target_sources(stdgpu PUBLIC FILE_SET stdgpu_config_header
                                 TYPE HEADERS
                                 BASE_DIRS ${STDGPU_BUILD_INCLUDE_DIR}
                                 FILES "${STDGPU_BUILD_INCLUDE_DIR}/stdgpu/config.h")
else()
    target_include_directories(stdgpu PUBLIC
                                      $<BUILD_INTERFACE:${STDGPU_INCLUDE_LOCAL_DIR}>
                                      $<BUILD_INTERFACE:${STDGPU_BUILD_INCLUDE_DIR}>)
endif()

# Still required as the interface header file sets are *conditionally* enabled for clients based on *their* CMake version
target_include_directories(stdgpu PUBLIC $<INSTALL_INTERFACE:${STDGPU_INCLUDE_INSTALL_DIR}>)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_target_properties(stdgpu PROPERTIES INTERFACE_HEADER_SETS_TO_VERIFY stdgpu_headers)

    # Manually specify the correct compile language for all headers that must be compiled by the device compiler
    set(STDGPU_DEVICE_HEADERS atomic.cuh
                              bitset.cuh
                              deque.cuh
                              mutex.cuh
                              queue.cuh
                              stack.cuh
                              unordered_map.cuh
                              unordered_set.cuh
                              vector.cuh)

    if(STDGPU_BACKEND STREQUAL STDGPU_BACKEND_CUDA)
        set_source_files_properties(${STDGPU_DEVICE_HEADERS} PROPERTIES LANGUAGE CUDA)
    elseif(STDGPU_BACKEND STREQUAL STDGPU_BACKEND_HIP)
        set_source_files_properties(${STDGPU_DEVICE_HEADERS} PROPERTIES LANGUAGE HIP)
    endif()
endif()

target_compile_features(stdgpu PUBLIC cxx_std_17)

target_compile_options(stdgpu PRIVATE ${STDGPU_DEVICE_FLAGS}
                                      ${STDGPU_HOST_FLAGS})

target_link_libraries(stdgpu PUBLIC thrust::thrust)

set_target_properties(stdgpu PROPERTIES CXX_CLANG_TIDY "${STDGPU_PROPERTY_CLANG_TIDY}")
set_target_properties(stdgpu PROPERTIES CXX_CPPCHECK "${STDGPU_PROPERTY_CPPCHECK}")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_target_properties(stdgpu PROPERTIES COMPILE_WARNING_AS_ERROR "${STDGPU_COMPILE_WARNING_AS_ERROR}")
endif()

add_library(stdgpu::stdgpu ALIAS stdgpu)


add_subdirectory(${STDGPU_BACKEND_DIRECTORY})


# Export targets and install header files
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    install(TARGETS stdgpu
            EXPORT stdgpu-targets
            ARCHIVE DESTINATION "${STDGPU_LIB_INSTALL_DIR}"
            LIBRARY DESTINATION "${STDGPU_LIB_INSTALL_DIR}"
            FILE_SET stdgpu_headers DESTINATION "${STDGPU_INCLUDE_INSTALL_DIR}"
            FILE_SET stdgpu_header_implementations DESTINATION "${STDGPU_INCLUDE_INSTALL_DIR}"
            FILE_SET stdgpu_config_header DESTINATION "${STDGPU_INCLUDE_INSTALL_DIR}"
            FILE_SET stdgpu_backend_headers DESTINATION "${STDGPU_INCLUDE_INSTALL_DIR}"
            FILE_SET stdgpu_backend_header_implementations DESTINATION "${STDGPU_INCLUDE_INSTALL_DIR}"
            COMPONENT stdgpu)
else()
    install(TARGETS stdgpu
            EXPORT stdgpu-targets
            DESTINATION "${STDGPU_LIB_INSTALL_DIR}"
            COMPONENT stdgpu)

    install(DIRECTORY "${STDGPU_INCLUDE_LOCAL_DIR}/" "${STDGPU_BUILD_INCLUDE_DIR}/"
            DESTINATION "${STDGPU_INCLUDE_INSTALL_DIR}"
            COMPONENT stdgpu
            FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.cuh"
            PATTERN "*_fwd")
endif()

# Install dependencies file and custom thrust module
configure_file("${stdgpu_SOURCE_DIR}/cmake/stdgpu-dependencies.cmake.in"
               "${STDGPU_BUILD_CMAKE_DIR}/stdgpu-dependencies.cmake"
               @ONLY)

install(FILES "${STDGPU_BUILD_CMAKE_DIR}/stdgpu-dependencies.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}"
        COMPONENT stdgpu)

install(FILES "${stdgpu_SOURCE_DIR}/cmake/Findthrust.cmake"
        DESTINATION "${STDGPU_CMAKE_INSTALL_DIR}"
        COMPONENT stdgpu)
