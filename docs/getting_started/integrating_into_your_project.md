# Integrating Into Your Project

This guide shows you how to integrate stdgpu into your project such that you can use its functionality. In particular, since CMake is used as the primary build system for stdgpu, the following instructions will focus on projects that use CMake as well for building.


In order to use stdgpu in your project, you need to either find a pre-installed version of it, or alternatively embed it as part of your project and build it alongside with your project from source.


:::::{tab-set}

::::{tab-item} Pre-Installed

```cmake
find_package(stdgpu REQUIRED)

add_library(your_project ...)
# ...

# Link your project against stdgpu
target_link_libraries(your_project PUBLIC stdgpu::stdgpu)
```

::::

::::{tab-item} From Source (git submodule)

```cmake
# Exclude unneeded parts from the build
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

add_subdirectory(stdgpu)

add_library(your_project ...)
# ...

# Link your project against stdgpu
target_link_libraries(your_project PUBLIC stdgpu::stdgpu)
```

:::{note}
Make sure that all dependencies listed in the [](building_from_source.md#prerequisites) are installed to build stdgpu within your project.
:::

:::{seealso}
A complete list of options to control the build of stdgpu within your project can be found in [](building_from_source.md#configuration-options).
:::

::::

::::{tab-item} From Source (Modern CMake)

```cmake
include(FetchContent)

FetchContent_Declare(
    stdgpu
    GIT_REPOSITORY https://github.com/stotko/stdgpu.git
    GIT_TAG        master
)

# Exclude unneeded parts from the build
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(stdgpu)

add_library(your_project ...)
# ...

# Link your project against stdgpu
target_link_libraries(your_project PUBLIC stdgpu::stdgpu)
```

:::{note}
Make sure that all dependencies listed in the [](building_from_source.md#prerequisites) are installed to build stdgpu within your project.
:::

:::{seealso}
A complete list of options to control the build of stdgpu within your project can be found in [](building_from_source.md#configuration-options).
:::

::::

:::::


In all of the above cases, the `stdgpu::stdgpu` target will become available in your project. The only further step required is to link your project target(s) against this target which will then propagate all necessary properties from stdgpu such as compile flags, include directories, library files, etc. to them.
