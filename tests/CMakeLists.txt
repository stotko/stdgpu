include(FetchContent)

FetchContent_Declare(
    googletest
    PREFIX googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
    URL_HASH SHA256=a9607c9215866bd425a725610c5e0f739eeb50887a57903df48891446ce6fb3c
    DOWNLOAD_DIR "${STDGPU_EXTERNAL_DIR}/googletest"
)

set(BUILD_GMOCK OFF CACHE INTERNAL "")
set(INSTALL_GTEST OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(googletest)

# Suppress clang-tidy errors on googletest by treating it as a system library
# Use SYSTEM in FetchContent_Declare for CMake 3.25+ instead when it becomes available
get_target_property(gtest_INCLUDE_DIRS gtest INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_INCLUDE_DIRS}")


add_subdirectory(stdgpu)
