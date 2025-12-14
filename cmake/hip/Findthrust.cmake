find_package(rocthrust)

if(rocthrust_INCLUDE_DIR)
    file(STRINGS "${rocthrust_INCLUDE_DIR}/thrust/version.h"
         THRUST_VERSION_STRING
         REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)")

    string(REGEX REPLACE "#define THRUST_VERSION[ \t]+([0-9]+).*" "\\1" THRUST_VERSION_STRING ${THRUST_VERSION_STRING})

    math(EXPR THRUST_VERSION_MAJOR "${THRUST_VERSION_STRING} / 100000")
    math(EXPR THRUST_VERSION_MINOR "(${THRUST_VERSION_STRING} / 100) % 1000")
    math(EXPR THRUST_VERSION_PATCH "${THRUST_VERSION_STRING} % 100")
    unset(THRUST_VERSION_STRING)

    set(THRUST_VERSION "${THRUST_VERSION_MAJOR}.${THRUST_VERSION_MINOR}.${THRUST_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(thrust
                                REQUIRED_VARS rocthrust_INCLUDE_DIR
                                VERSION_VAR THRUST_VERSION)

if(thrust_FOUND)
    add_library(thrust::thrust INTERFACE IMPORTED)
    target_link_libraries(thrust::thrust INTERFACE roc::rocthrust)

    mark_as_advanced(THRUST_INCLUDE_DIR
                     THRUST_VERSION
                     THRUST_VERSION_MAJOR
                     THRUST_VERSION_MINOR
                     THRUST_VERSION_PATCH)
endif()
