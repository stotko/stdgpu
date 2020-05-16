
# Required for rocprim
find_package(hip QUIET CONFIG
             PATHS "/opt/rocm/hip")

# Required for rocthrust
find_package(rocprim QUIET CONFIG
             PATHS "/opt/rocm/rocprim")

find_package(rocthrust QUIET CONFIG
             PATHS "/opt/rocm/rocthrust")

if(hip_FOUND AND rocprim_FOUND AND rocthrust_FOUND)
    find_path(THRUST_INCLUDE_DIR
              HINTS
              "/opt/rocm/rocthrust/include"
              NAMES
              "thrust/version.h")
endif()

if(THRUST_INCLUDE_DIR)
    file(STRINGS "${THRUST_INCLUDE_DIR}/thrust/version.h"
         THRUST_VERSION_STRING
         REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)")

    string(REGEX REPLACE "#define THRUST_VERSION[ \t]+" "" THRUST_VERSION_STRING ${THRUST_VERSION_STRING})

    math(EXPR THRUST_VERSION_MAJOR "${THRUST_VERSION_STRING} / 100000")
    math(EXPR THRUST_VERSION_MINOR "(${THRUST_VERSION_STRING} / 100) % 1000")
    math(EXPR THRUST_VERSION_PATCH "${THRUST_VERSION_STRING} % 100")
    unset(THRUST_VERSION_STRING)

    set(THRUST_VERSION "${THRUST_VERSION_MAJOR}.${THRUST_VERSION_MINOR}.${THRUST_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(thrust
                                  REQUIRED_VARS THRUST_INCLUDE_DIR
                                  VERSION_VAR THRUST_VERSION)


if(thrust_FOUND)
    add_library(thrust::thrust INTERFACE IMPORTED)
    # WORKAROUND: ROCm 3.1 forgot to export the namespace, so workaround this bug
    if(TARGET roc::rocthrust)
        set_target_properties(thrust::thrust PROPERTIES INTERFACE_LINK_LIBRARIES roc::rocthrust)
    elseif(TARGET rocthrust)
        set_target_properties(thrust::thrust PROPERTIES INTERFACE_LINK_LIBRARIES rocthrust)
    endif()

    mark_as_advanced(THRUST_INCLUDE_DIR
                     THRUST_VERSION
                     THRUST_VERSION_MAJOR
                     THRUST_VERSION_MINOR
                     THRUST_VERSION_PATCH)
endif()
