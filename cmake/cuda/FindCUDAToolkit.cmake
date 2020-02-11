
# NOTE: This module is a self-written, oversimplified subset of CMake's FindCUDAToolkit.cmake module (introduced in CMake 3.17).
#       Therefore, it only provides the most commonly used target CUDA::cudart which is required for the project.
#       This module becomes obsolete once CMake 3.17 will be the minimum required version.

# Assume the CUDA language has been enabled
if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "FindCUDAToolkit.cmake: CUDA language not enabled or CUDA compiler not found. This oversimplified module requires CUDA to be successfully found")
endif()


# Workaround for unset variables when compiling with Visual Studio (Windows)
if(MSVC)
    get_filename_component(STDGPU_CUDA_COMPILER_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    set(STDGPU_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${STDGPU_CUDA_COMPILER_DIR}/../include")
    set(STDGPU_CUDA_IMPLICIT_LINK_DIRECTORIES "${STDGPU_CUDA_COMPILER_DIR}/../lib/x64")
else()
    set(STDGPU_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(STDGPU_CUDA_IMPLICIT_LINK_DIRECTORIES ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


find_path(STDGPU_CUDART_INCLUDE_DIR
          HINTS
          ${STDGPU_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
          NAMES
          "cuda_runtime_api.h")

find_library(STDGPU_CUDART_LIBRARY cudart
             HINTS
             ${STDGPU_CUDA_IMPLICIT_LINK_DIRECTORIES})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDAToolkit
                                  REQUIRED_VARS STDGPU_CUDART_INCLUDE_DIR STDGPU_CUDART_LIBRARY
                                  VERSION_VAR CMAKE_CUDA_COMPILER_VERSION)


if(CUDAToolkit_FOUND)
    if(NOT TARGET CUDA::cudart)
        add_library(CUDA::cudart INTERFACE IMPORTED)
        set_target_properties(CUDA::cudart PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${STDGPU_CUDART_INCLUDE_DIR}")
        set_target_properties(CUDA::cudart PROPERTIES INTERFACE_LINK_LIBRARIES "${STDGPU_CUDART_LIBRARY}")
    endif()

    mark_as_advanced(STDGPU_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    mark_as_advanced(STDGPU_CUDA_IMPLICIT_LINK_DIRECTORIES)
    mark_as_advanced(STDGPU_CUDART_INCLUDE_DIR)
    mark_as_advanced(STDGPU_CUDART_LIBRARY)
endif()
