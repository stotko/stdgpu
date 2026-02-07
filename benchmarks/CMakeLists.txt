include(FetchContent)

FetchContent_Declare(
    benchmark
    PREFIX benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.zip
    URL_HASH SHA256=68c9c65cee4864db42c3af9ff5b5cfa32ce1b01d9653136c5f4ff96e18a9b8f2
    DOWNLOAD_DIR "${STDGPU_EXTERNAL_DIR}/benchmark"
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_WERROR OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_DOXYGEN OFF CACHE INTERNAL "")
set(BENCHMARK_INSTALL_DOCS OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(benchmark)

# Suppress clang-tidy errors on benchmark by treating it as a system library
# Use SYSTEM in FetchContent_Declare for CMake 3.25+ instead when it becomes available
get_target_property(benchmark_INCLUDE_DIRS benchmark INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(benchmark PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${benchmark_INCLUDE_DIRS}")


add_subdirectory(stdgpu)
