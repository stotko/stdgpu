# Check for GPUs present and their compute capability
# based on http://stackoverflow.com/questions/2285185/easiest-way-to-test-for-existence-of-cuda-capable-gpu-from-cmake/2297877#2297877 (Christopher Bruns)

set(STDGPU_CUDA_COMPUTE_CAPABILITIES_SOURCE "${CMAKE_CURRENT_LIST_DIR}/compute_capability.cpp")
message(STATUS "Detecting CCs of GPUs : ${STDGPU_CUDA_COMPUTE_CAPABILITIES_SOURCE}")

# Workaround for unset variables when compiling with Visual Studio (Windows)
if(MSVC)
    get_filename_component(STDGPU_CUDA_COMPILER_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${STDGPU_CUDA_COMPILER_DIR}/../include")
    set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${STDGPU_CUDA_COMPILER_DIR}/../lib/x64")
endif()

# Detect CUDA runtime library to build the .cpp file (implicitly used if .cu file was used)
find_library(STDGPU_CUDART_LIBRARY cudart
             HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

try_run(STDGPU_RUN_RESULT_VAR STDGPU_COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}
        ${STDGPU_CUDA_COMPUTE_CAPABILITIES_SOURCE}
        CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            "-DLINK_LIBRARIES=${STDGPU_CUDART_LIBRARY}"
        COMPILE_OUTPUT_VARIABLE STDGPU_COMPILE_OUTPUT_VAR
        RUN_OUTPUT_VARIABLE STDGPU_RUN_OUTPUT_VAR)

# COMPILE_RESULT_VAR is TRUE when compile succeeds
# RUN_RESULT_VAR is zero when a GPU is found
if(STDGPU_COMPILE_RESULT_VAR AND NOT STDGPU_RUN_RESULT_VAR)
    message(STATUS "Detecting CCs of GPUs : ${STDGPU_CUDA_COMPUTE_CAPABILITIES_SOURCE} - Success (found CCs : ${STDGPU_RUN_OUTPUT_VAR})")
    set(STDGPU_CUDA_HAVE_GPUS TRUE CACHE BOOL "Whether CUDA-capable GPUs are present")
    set(STDGPU_CUDA_COMPUTE_CAPABILITIES ${STDGPU_RUN_OUTPUT_VAR} CACHE STRING "Compute capabilities of CUDA-capable GPUs")
    mark_as_advanced(STDGPU_CUDA_COMPUTE_CAPABILITIES)
elseif(NOT STDGPU_COMPILE_RESULT_VAR)
    message(STATUS "Detecting CCs of GPUs : ${STDGPU_CUDA_COMPUTE_CAPABILITIES_SOURCE} - Failed to compile")
    set(STDGPU_CUDA_HAVE_GPUS FALSE CACHE BOOL "Whether CUDA-capable GPUs are present")
else()
    message(STATUS "Detecting CCs of GPUs : ${STDGPU_CUDA_COMPUTE_CAPABILITIES_SOURCE} - No CUDA-capable GPU found")
    set(STDGPU_CUDA_HAVE_GPUS FALSE CACHE BOOL "Whether CUDA-capable GPUs are present")
endif()
