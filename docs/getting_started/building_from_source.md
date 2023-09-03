# Building From Source

This guide shows you how to build stdgpu from source. Since we use CMake for cross-platform building, the building instructions should work across operating systems and distributions. Furthermore, the prerequisites cover specific instructions to install the build dependencies on Linux (Ubuntu) and Windows.


## Prerequisites

Before building the library, please make sure that all required development tools and dependencies for the respective backend are installed on your system. The following table shows the **minimum required versions** of each depenency. *Newer versions* of these tools are supported as well.


::::::{tab-set}

:::::{tab-item} Linux (Ubuntu)
:sync: linux

::::{tab-set}

:::{tab-item} CUDA
:sync: cuda

- C++17 compiler
    - GCC 9: `sudo apt install g++`
    - Clang 10: `sudo apt install clang`
- CUDA compiler
    - NVCC (Already included in CUDA Toolkit)
    - Clang 10: `sudo apt install clang`
- CUDA Toolkit 11.0: <https://developer.nvidia.com/cuda-downloads>
- CMake 3.18: `sudo apt install cmake`

:::

:::{tab-item} OpenMP
:sync: openmp

- C++17 compiler with OpenMP 2.0
    - GCC 9: `sudo apt install g++`
    - Clang 10: `sudo apt install clang libomp-dev`
- CMake 3.18: `sudo apt install cmake`
- thrust 1.9.9: <https://github.com/NVIDIA/thrust>

:::

:::{tab-item} HIP (experimental)
:sync: hip

- C++17 compiler
    - GCC 9: `sudo apt install g++`
    - Clang 10: `sudo apt install clang`
- HIP compiler
    - Clang (Already included in ROCm)
- ROCm 5.1 <https://github.com/RadeonOpenCompute/ROCm>
- CMake 3.21.3: `sudo apt install cmake`

:::

::::

:::::

:::::{tab-item} Windows
:sync: windows

::::{tab-set}

:::{tab-item} CUDA
:sync: cuda

- C++17 compiler
    - MSVC 19.20 (Visual Studio 2019) <https://visualstudio.microsoft.com/downloads/>
- CUDA compiler
    - NVCC (Already included in CUDA Toolkit)
- CUDA Toolkit 11.0: <https://developer.nvidia.com/cuda-downloads>
- CMake 3.18: <https://cmake.org/download>

:::

:::{tab-item} OpenMP
:sync: openmp

- C++17 compiler including OpenMP 2.0
    - MSVC 19.20 (Visual Studio 2019) <https://visualstudio.microsoft.com/downloads/>
- CMake 3.18: <https://cmake.org/download>
- thrust 1.9.9: <https://github.com/NVIDIA/thrust>

:::

:::{tab-item} HIP (experimental)
:sync: hip

- C++17 compiler
    - MSVC 19.20 (Visual Studio 2019) <https://visualstudio.microsoft.com/downloads/>
- HIP compiler
    - Clang (Already included in ROCm)
- ROCm 5.1 <https://github.com/RadeonOpenCompute/ROCm>
- CMake 3.21.3: <https://cmake.org/download>

:::

::::

:::::

::::::

While the instructions will likely also work for instance for Debian, other Linux distributions (e.g. Arch Linux and derivatives) may use a different naming scheme for the required packages.


## Downloading the Source Code

In order to get the source code needed for building, the most common approach is to clone the upstream GitHub repository.

```sh
git clone https://github.com/stotko/stdgpu
```


## Build Instructions

Since stdgpu is built with CMake, the usual steps for building CMake-based projects can be performed here as well. In the examplary instructions below, we built stdgpu in **Release** mode and installed it into a local **bin** directory. Different modes and installation directories works can be used in a similar way.


### Configuring

First, a build directory (usually `build/`) should be created and the build configuration should be evaluated by CMake and stored into this directory.

::::{tab-set}

:::{tab-item} Direct Command
:sync: direct

```sh
mkdir build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=bin
```

:::

:::{tab-item} Provided Script
:sync: script

```sh
bash tools/backend/configure_cuda.sh Release
```

:::

::::


If you leave out `CMAKE_INSTALL_PREFIX`, CMake will automatically select an appropriate (platform-dependent) system directory for installation instead.


:::{seealso}
A complete list of options used via `-D<option>=<value>` to control the build of stdgpu can be found in [](#configuration-options).
:::


### Compiling

Then, the library itself as well as further components (examples, benchmarks, or tests depending on the configuration) are compiled. To speed-up the compilation, the maximum number of parallel jobs used for building can be specified to override the default number.

::::{tab-set}

:::{tab-item} Direct Command
:sync: direct

```sh
cmake --build build --config Release --parallel 8
```

:::

:::{tab-item} Provided Script
:sync: script

```sh
bash tools/build.sh Release
```

:::

::::


### Testing

Running the unit tests is **optional** but *recommended* to verify that all features of stdgpu work correctly on your system. This requires the CMake option `STDGPU_BUILD_TESTS` to be set to `ON` during configuration (already the default if not altered), see [](#configuration-options) for a complete list of options.

::::{tab-set}

:::{tab-item} Direct Command
:sync: direct

```sh
cmake -E chdir build ctest -V -C Release
```

:::

:::{tab-item} Provided Script
:sync: script

```sh
bash tools/run_tests.sh Release
```

:::

::::


### Installing

As a final **optional** step, you can install the locally compiled version of stdgpu on your system.

:::::{tab-set}

::::{tab-item} Direct Command
:sync: direct

```sh
cmake --install build --config Release
```

:::{admonition} Uninstalling
:class: tip dropdown
If the `build/` directory from which stdgpu has been installed has not been deleted, you can also revert the installation by calling the `uninstall` target:

```sh
cmake --build build --config Release --target uninstall
```
:::

::::

::::{tab-item} Provided Script
:sync: script

```sh
bash tools/install.sh Release
```

:::{admonition} Uninstalling
:class: tip dropdown
If the `build/` directory from which stdgpu has been installed is still present, you can revert the installation by running the `uninstall` script:

```sh
bash tools/uninstall.sh Release
```
:::

::::

:::::



## Configuration Options

Building stdgpu from source can be customized in various ways. The following list of CMake options divided into *build-related* and *library-related* options can be used for this purpose.


### Build-related Options

The build process can be controlled by the following options and allow to enable/disable certains parts of the library sources:

`STDGPU_BACKEND`
: Selected device system backend. Must be either `STDGPU_BACKEND_CUDA`, `STDGPU_BACKEND_OPENMP`, or `STDGPU_BACKEND_HIP`. \
**default:** `STDGPU_BACKEND_CUDA`

`STDGPU_BUILD_SHARED_LIBS`
: Build the project as a shared library if set to `ON`, or as a static library if set to `OFF`. \
**default:** `BUILD_SHARED_LIBS`

`STDGPU_SETUP_COMPILER_FLAGS`
: Construct and use a set of common compiler flags for the C++ and further involved compilers (e.g. the CUDA compiler). This is usually only useful if stdgpu is built as a standalone project. \
**default:** `ON` if standalone, `OFF` if included via `add_subdirectory` or `FetchContent`

`STDGPU_COMPILE_WARNING_AS_ERROR`
: Treat compiler warnings as errors. Useful in CI environments. \
**default:** `OFF`

`STDGPU_BUILD_DOCUMENTATION`
: Enable building the documentation. Note that this will not actually generate the documentation during building, which instead must be separately built using a dedicated CMake target/script. \
**default:** `OFF`

`STDGPU_BUILD_EXAMPLES`
: Enable building the examples. \
**default:** `ON`

`STDGPU_BUILD_BENCHMARKS`
: Enable building the benchmarks. \
**default:** `ON`

`STDGPU_BUILD_TESTS`
: Enable building the tests. \
**default:** `ON`

`STDGPU_BUILD_TEST_COVERAGE`
: Enable building a test coverage report. Note that this will not actually generate the report during building, which instead must be separately built using a dedicated CMake target/script. \
**default:** `OFF`

`STDGPU_ANALYZE_WITH_CLANG_TIDY`
: Analyze the code with clang-tidy while building. \
**default:** `OFF`

`STDGPU_ANALYZE_WITH_CPPCHECK`
: Analyze the code with cppcheck while building. \
**default:** `OFF`


### Library-related Options

In addition, the behavior of the library implementation can also be customized to, e.g., provide maximum performance:

`STDGPU_ENABLE_CONTRACT_CHECKS`
: Enable contract checks. Useful for debugging purposes, but comes with a significant impact on the runtime performance. \
**default:** `OFF` if `CMAKE_BUILD_TYPE` equals `Release` or `MinSizeRel`, `ON` otherwise

`STDGPU_USE_32_BIT_INDEX`
: Use 32-bit instead of 64-bit signed integer for `index_t`. Useful for tuning the performance if processing of more than 2.1 billion elements is not required. \
**default:** `ON`
