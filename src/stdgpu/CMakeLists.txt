
if(${CMAKE_BUILD_TYPE} MATCHES "Release" OR ${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
    set(STDGPU_ENABLE_CONTRACT_CHECKS_DEFAULT OFF)
else()
    set(STDGPU_ENABLE_CONTRACT_CHECKS_DEFAULT ON)
endif()

option(STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING "Enable warnings when auxiliary arrays are allocated in memory API, default: OFF" OFF)
option(STDGPU_ENABLE_CONTRACT_CHECKS "Enable contract checks, default: OFF if CMAKE_BUILD_TYPE is Release or MinSizeRel, ON otherwise" ${STDGPU_ENABLE_CONTRACT_CHECKS_DEFAULT})
option(STDGPU_ENABLE_MANAGED_ARRAY_WARNING "Enable warnings when managed memory is initialized on the host side but should be on device in memory API, default: OFF" OFF)
option(STDGPU_USE_32_BIT_INDEX "Use 32-bit instead of 64-bit signed integer for index_t, default: ON" ON)
option(STDGPU_USE_FAST_DESTROY "Use fast destruction of allocated arrays (filled with a default value) by omitting destructor calls in memory API, default: OFF" OFF)
option(STDGPU_USE_FIBONACCI_HASHING "Use Fibonacci Hashing instead of Modulo to compute hash bucket indices, default: ON" ON)


set(STDGPU_INCLUDE_LOCAL_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(STDGPU_CONFIG_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

configure_file("${STDGPU_INCLUDE_LOCAL_DIR}/stdgpu/config.h.in"
               "${STDGPU_CONFIG_BUILD_DIR}/stdgpu/config.h")

if(STDGPU_BUILD_SHARED_LIBS)
    add_library(stdgpu SHARED)
else()
    add_library(stdgpu STATIC)
endif()

target_sources(stdgpu PRIVATE impl/bitset.cu
                              impl/iterator.cpp
                              impl/memory.cpp
                              impl/mutex.cu
                              impl/limits.cpp)

add_subdirectory(${STDGPU_BACKEND_DIRECTORY})

target_include_directories(stdgpu PUBLIC
                                  $<BUILD_INTERFACE:${STDGPU_INCLUDE_LOCAL_DIR}>
                                  $<BUILD_INTERFACE:${STDGPU_CONFIG_BUILD_DIR}>
                                  $<INSTALL_INTERFACE:${STDGPU_INCLUDE_INSTALL_DIR}>)

target_compile_features(stdgpu PUBLIC cxx_std_14)

add_library(stdgpu::stdgpu ALIAS stdgpu)

# Export targets and install header files
install(TARGETS stdgpu
        EXPORT stdgpu-targets
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${STDGPU_LIB_INSTALL_DIR}"
        COMPONENT stdgpu)

install(DIRECTORY "${STDGPU_CONFIG_BUILD_DIR}/"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${STDGPU_INCLUDE_INSTALL_DIR}
        COMPONENT stdgpu)

install(DIRECTORY "${STDGPU_INCLUDE_LOCAL_DIR}/"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${STDGPU_INCLUDE_INSTALL_DIR}
        COMPONENT stdgpu
        PATTERN "*.cpp" EXCLUDE
        PATTERN "*.cu" EXCLUDE
        PATTERN "*.txt" EXCLUDE)
