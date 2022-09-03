set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_WERROR OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_DOXYGEN OFF CACHE INTERNAL "")
set(BENCHMARK_INSTALL_DOCS OFF CACHE INTERNAL "")
add_subdirectory(benchmark-1.7.0)

# Linking google benchmark statically on Windows results in linker errors for 1.7.0.
# This is fixed upstream (see https://github.com/google/benchmark/pull/1470).
#
# TODO: Remove this backported fix when upgrading google benchmark again!
if (NOT BUILD_SHARED_LIBS)
    target_compile_definitions(benchmark PUBLIC -DBENCHMARK_STATIC_DEFINE)
endif()

add_subdirectory(stdgpu)
