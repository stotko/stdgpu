
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
                              impl/memory.cpp
                              impl/limits.cpp)

target_include_directories(stdgpu PUBLIC
                                  $<BUILD_INTERFACE:${STDGPU_INCLUDE_LOCAL_DIR}>
                                  $<BUILD_INTERFACE:${STDGPU_BUILD_INCLUDE_DIR}>
                                  $<INSTALL_INTERFACE:${STDGPU_INCLUDE_INSTALL_DIR}>)

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
