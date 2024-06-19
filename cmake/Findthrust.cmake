include("${CMAKE_CURRENT_LIST_DIR}/${STDGPU_BACKEND_DIRECTORY}/determine_thrust_paths.cmake")
stdgpu_determine_thrust_paths(STDGPU_THRUST_PATHS)

find_path(THRUST_INCLUDE_DIR
          HINTS
          ${STDGPU_THRUST_PATHS}
          NAMES
          "thrust/version.h")

list(APPEND THRUST_INCLUDE_DIR_VARS THRUST_INCLUDE_DIR)

if(THRUST_INCLUDE_DIR)
    file(STRINGS "${THRUST_INCLUDE_DIR}/thrust/version.h"
         THRUST_VERSION_STRING
         REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)")

    string(REGEX REPLACE "#define THRUST_VERSION[ \t]+([0-9]+).*" "\\1" THRUST_VERSION_STRING ${THRUST_VERSION_STRING})

    math(EXPR THRUST_VERSION_MAJOR "${THRUST_VERSION_STRING} / 100000")
    math(EXPR THRUST_VERSION_MINOR "(${THRUST_VERSION_STRING} / 100) % 1000")
    math(EXPR THRUST_VERSION_PATCH "${THRUST_VERSION_STRING} % 100")
    unset(THRUST_VERSION_STRING)

    set(THRUST_VERSION "${THRUST_VERSION_MAJOR}.${THRUST_VERSION_MINOR}.${THRUST_VERSION_PATCH}")


    if(THRUST_VERSION VERSION_GREATER_EQUAL "2.0.0")
        find_path(LIBCUDACXX_INCLUDE_DIR
                  HINTS
                  "${THRUST_INCLUDE_DIR}/../libcudacxx/include"
                  ${STDGPU_THRUST_PATHS}
                  NAMES
                  "cuda/std/version")

        find_path(CUB_INCLUDE_DIR
                  HINTS
                  "${THRUST_INCLUDE_DIR}/../cub"
                  ${STDGPU_THRUST_PATHS}
                  NAMES
                  "cub/version.cuh")

        list(APPEND THRUST_INCLUDE_DIR_VARS LIBCUDACXX_INCLUDE_DIR CUB_INCLUDE_DIR)

        mark_as_advanced(LIBCUDACXX_INCLUDE_DIR
                         CUB_INCLUDE_DIR)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(thrust
                                  REQUIRED_VARS ${THRUST_INCLUDE_DIR_VARS}
                                  VERSION_VAR THRUST_VERSION)


if(thrust_FOUND)
    foreach(inc IN LISTS THRUST_INCLUDE_DIR_VARS)
        list(APPEND THRUST_INCLUDE_DIRS "${${inc}}")
    endforeach()
    list(REMOVE_DUPLICATES THRUST_INCLUDE_DIRS)

    add_library(thrust::thrust INTERFACE IMPORTED)
    set_target_properties(thrust::thrust PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${THRUST_INCLUDE_DIRS}")

    mark_as_advanced(THRUST_INCLUDE_DIR
                     THRUST_INCLUDE_DIR_VARS
                     THRUST_INCLUDE_DIRS
                     THRUST_VERSION
                     THRUST_VERSION_MAJOR
                     THRUST_VERSION_MINOR
                     THRUST_VERSION_PATCH)
endif()
