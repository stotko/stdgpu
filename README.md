<p align="center">
<img src="./doc/stdgpu_logo.png" width="500" />
</p>

<h1 align="center">stdgpu: Efficient STL-like Data Structures on the GPU</h1>

<p align="center">
<a href="https://dev.azure.com/patrickstotko/stdgpu/_build/latest?definitionId=3&branchName=master" alt="Build Status">
    <img src="https://dev.azure.com/patrickstotko/stdgpu/_apis/build/status/stotko.stdgpu?branchName=master"/>
</a>
<a href="https://codecov.io/gh/stotko/stdgpu" alt="Code Coverage">
  <img src="https://codecov.io/gh/stotko/stdgpu/branch/master/graph/badge.svg" />
</a>
<a href="https://scan.coverity.com/projects/stotko-stdgpu" alt="Coverity Scan">
   <img src="https://scan.coverity.com/projects/20259/badge.svg"/>
</a>
<a href="https://bestpractices.coreinfrastructure.org/projects/3645" alt="Best Practices">
    <img src="https://bestpractices.coreinfrastructure.org/projects/3645/badge">
</a>
<a href="https://stotko.github.io/stdgpu" alt="Documentation">
    <img src="https://img.shields.io/badge/docs-doxygen-blue.svg"/>
</a>
<a href="https://github.com/stotko/stdgpu/issues" alt="Issues">
    <img src="https://img.shields.io/github/issues/stotko/stdgpu"/>
</a>
<a href="https://github.com/stotko/stdgpu/blob/master/CONTRIBUTING.md" alt="Contributing">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat"/>
</a>
<a href="https://github.com/stotko/stdgpu/blob/master/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/stotko/stdgpu"/>
</a>
<a href="https://github.com/stotko/stdgpu/releases" alt="Latest Release">
    <img src="https://img.shields.io/github/v/release/stotko/stdgpu"/>
</a>
<a href="https://github.com/stotko/stdgpu/compare/release...master" alt="Commits Since Latest Release">
    <img src="https://img.shields.io/github/commits-since/stotko/stdgpu/latest"/>
</a>
</p>

<p align="center">
<a href="#features">Features</a> |
<a href="#examples">Examples</a> |
<a href="#getting-started">Getting Started</a> |
<a href="#usage">Usage</a> |
<a href="#contributing">Contributing</a> |
<a href="#license">License</a> |
<a href="#contact">Contact</a>
</p>


## Features

stdgpu is an open-source library providing several generic GPU data structures for fast and reliable data management. Multiple platforms such as **CUDA**, **OpenMP**, and **HIP** are supported allowing you to rapidly write highly complex **agnostic** and **native** algorithms that look like sequential CPU code but are executed in parallel on the GPU.

- **Productivity**. Previous libraries such as thrust, VexCL, ArrayFire or Boost.Compute focus on the fast and efficient implementation of various algorithms for contiguously stored data to enhance productivity. stdgpu follows an *orthogonal approach* and focuses on *fast and reliable data management* to enable the rapid development of more general and flexible GPU algorithms just like their CPU counterparts.

- **Interoperability**. Instead of providing yet another ecosystem, stdgpu is designed to be a *lightweight container library*. Therefore, a core feature of stdgpu is its interoperability with previous established frameworks, i.e. the thrust library, to enable a *seamless integration* into new as well as existing projects.

- **Maintainability**. Following the trend in recent C++ standards of providing functionality for safer and more reliable programming, the philosophy of stdgpu is to provide *clean and familiar functions* with strong guarantees that encourage users to write *more robust code* while giving them full control to achieve a high performance.

At its heart, stdgpu offers the following GPU data structures and containers:

<table>
<tr align="center">
<td><code>atomic</code> &amp; <code>atomic_ref</code><br>Atomic primitive types and references</td>
<td><code>bitset</code><br>Space-efficient bit array</td>
<td><code>deque</code><br>Dynamically sized double-ended queue</td>
</tr>
<tr align="center">
<td><code>queue</code> &amp; <code>stack</code><br>Container adapters</td>
<td><code>unordered_map</code> &amp; <code>unordered_set</code><br>Hashed collection of unique keys and key-value pairs</td>
<td><code>vector</code><br>Dynamically sized contiguous array</td>
</tr>
</table>

In addition, stdgpu also provides commonly required functionality in `algorithm`, `bit`, `cmath`, `contract`, `cstddef`, `cstlib`, `functional`, `iterator`, `memory`, `mutex`, `ranges`, and `utility` to complement the GPU data structures and to increase their usability and interoperability.


## Examples

In order to reliably perform tasks on the GPU, stdgpu offers flexible interfaces that can be used in both **agnostic code**, e.g. via the algorithms provided by thrust, as well as in **native code**, e.g. custom CUDA kernels.

<b>Agnostic code</b>. In the context of the SLAMCast live telepresence system, a simple task is the integration of a range of updated blocks into the duplicate-free set of queued blocks for streaming which can be expressed very conveniently:

```cpp
class stream_set
{
public:
    void
    add_blocks(const short3* blocks,
               const stdgpu::index_t n)
    {
        set.insert(stdgpu::make_device(blocks),
                   stdgpu::make_device(blocks + n));
    }

    // Further functions

private:
    stdgpu::unordered_set<short3> set;
    // Further members
};
```

<b>Native code</b>. More complex operations such as the creation of the update set or other complex algorithms can be implemented natively, e.g. in custom CUDA kernels with stdgpu's CUDA backend enabled:

```cpp
__global__ void
compute_update_set(const short3* blocks,
                   const stdgpu::index_t n,
                   const stdgpu::unordered_map<short3, voxel*> tsdf_block_map,
                   stdgpu::unordered_set<short3> mc_update_set)
{
    // Global thread index
    stdgpu::index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    short3 b_i = blocks[i];

    // Neighboring candidate blocks for the update
    short3 mc_blocks[8]
    = {
        short3(b_i.x - 0, b_i.y - 0, b_i.z - 0),
        short3(b_i.x - 1, b_i.y - 0, b_i.z - 0),
        short3(b_i.x - 0, b_i.y - 1, b_i.z - 0),
        short3(b_i.x - 0, b_i.y - 0, b_i.z - 1),
        short3(b_i.x - 1, b_i.y - 1, b_i.z - 0),
        short3(b_i.x - 1, b_i.y - 0, b_i.z - 1),
        short3(b_i.x - 0, b_i.y - 1, b_i.z - 1),
        short3(b_i.x - 1, b_i.y - 1, b_i.z - 1),
    };

    for (stdgpu::index_t j = 0; j < 8; ++j)
    {
        // Only consider existing neighbors
        if (tsdf_block_map.contains(mc_blocks[j]))
        {
            mc_update_set.insert(mc_blocks[j]);
        }
    }
}
```

More examples can be found in the <a href="https://github.com/stotko/stdgpu/tree/master/examples">`examples`</a> directory.


## Getting Started

To compile the library, please make sure to fulfill the build requirements and execute the respective build scripts.


### Prerequisites

The following general and backend-specific dependencies (or newer versions) are required to compile the library.

<b>Required</b>

- C++14 compiler (one of the following)
    - GCC 7
        - (Ubuntu 18.04) `sudo apt install g++ make`
    - Clang 6
        - (Ubuntu 18.04) `sudo apt install clang make`
    - MSVC 19.20
        - (Windows) Visual Studio 2019 https://visualstudio.microsoft.com/downloads/
- CMake 3.15
    - (Ubuntu 18.04) https://apt.kitware.com
    - (Windows) https://cmake.org/download
- thrust 1.9.2
    - (Ubuntu/Windows) https://github.com/thrust/thrust
    - May already be installed by backend dependencies

<b>Required for CUDA backend</b>

- CUDA Toolkit 10.0
    - (Ubuntu/Windows) https://developer.nvidia.com/cuda-downloads
    - Includes thrust

<b>Required for OpenMP backend</b>

- OpenMP 2.0
    - GCC 7
        - (Ubuntu 18.04) Already installed
    - Clang 6
        - (Ubuntu 18.04) `sudo apt install libomp-dev`
    - MSVC 19.20
        - (Windows) Already installed

<b>Required for HIP backend (experimental)</b>

- ROCm 3.1
    - (Ubuntu) https://github.com/RadeonOpenCompute/ROCm
    - Includes thrust

<b>Optional</b>

- Doxygen 1.8.13
    - (Ubuntu 18.04) `sudo apt install doxygen`
    - (Windows) http://www.doxygen.nl/download.html
- lcov 1.13
    - (Ubuntu 18.04) `sudo apt install lcov`


### Building

For convenience, we provide several cross-platform scripts to build the project. Note that some scripts depend on the build type, so there are scripts for both `debug` and `release` builds.

Command | Effect
--- | ---
<code>sh&nbsp;scripts/setup_&lt;build_type&gt;.sh</code> | Performs a full clean build of the project. Removes old build, configures the project (build path: `./build`), builds the project, and runs the unit tests.
<code>sh&nbsp;scripts/build_&lt;build_type&gt;.sh</code> | (Re-)Builds the project. Requires that project is configured (or set up).
<code>sh&nbsp;scripts/run_tests_&lt;build_type&gt;.sh</code> | Runs the unit tests. Requires that project is built.
<code>sh&nbsp;scripts/create_documentation.sh</code> | Builds the documentation locally. Requires doxygen and that project is configured (or set up).
<code>sh&nbsp;scripts/run_coverage.sh</code> | Builds a test coverage report locally. Requires lcov and that project is configured (or set up).
<code>sh&nbsp;scripts/install_&lt;build_type&gt;.sh</code> | Installs the project at the configured install path (default: `./bin`).


## Usage

In the following, we show some examples on how the library can be integrated into and used in a project.


### CMake Integration

To use the library in your project, you can either install it externally first and then include it using `find_package`:

```cmake
find_package(stdgpu 1.0.0 REQUIRED)

add_library(foo ...)

target_link_libraries(foo PUBLIC stdgpu::stdgpu)
```

Or you can embed it into your project and build it from a subdirectory:

```cmake
# Exclude the examples from the build
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")

# Exclude the tests from the build
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

add_subdirectory(stdgpu)

add_library(foo ...)

target_link_libraries(foo PUBLIC stdgpu::stdgpu)
```


### CMake Options

To configure the library, two sets of options are provided. The following build options control the build process:

Build Option | Effect | Default
--- | --- | ---
`STDGPU_BACKEND` | Device system backend | `STDGPU_BACKEND_CUDA`
`STDGPU_SETUP_COMPILER_FLAGS` | Constructs the compiler flags | `ON` if standalone, `OFF` if included via `add_subdirectory`
`STDGPU_TREAT_WARNINGS_AS_ERRORS` | Treats compiler warnings as errors | `OFF`
`STDGPU_ANALYZE_WITH_CLANG_TIDY` | Analyzes the code with clang-tidy | `OFF`
`STDGPU_ANALYZE_WITH_CPPCHECK` | Analyzes the code with cppcheck | `OFF`
`STDGPU_BUILD_SHARED_LIBS` | Builds the project as a shared library, if set to `ON`, or as a static library, if set to `OFF` | `BUILD_SHARED_LIBS`
`STDGPU_BUILD_EXAMPLES` | Build the examples | `ON`
`STDGPU_BUILD_TESTS` | Build the unit tests | `ON`
`STDGPU_BUILD_TEST_COVERAGE` | Build a test coverage report | `OFF`

In addition, the implementation of some functionality can be controlled via configuration options:

Configuration Option | Effect | Default
--- | --- | ---
`STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING` | Enable warnings when auxiliary arrays are allocated in memory API (**deprecated**) | `OFF`
`STDGPU_ENABLE_CONTRACT_CHECKS` | Enable contract checks | `OFF` if `CMAKE_BUILD_TYPE` equals `Release` or `MinSizeRel`, `ON` otherwise
`STDGPU_ENABLE_MANAGED_ARRAY_WARNING` | Enable warnings when managed memory is initialized on the host side but should be on device in memory API (**deprecated**) | `OFF`
`STDGPU_USE_32_BIT_INDEX` | Use 32-bit instead of 64-bit signed integer for `index_t` | `ON`
`STDGPU_USE_FAST_DESTROY` | Use fast destruction of allocated arrays (filled with a default value) by omitting destructor calls in memory API (**deprecated**) | `OFF`
`STDGPU_USE_FIBONACCI_HASHING` | Use Fibonacci Hashing instead of Modulo to compute hash bucket indices (**deprecated**) | `ON`


## Contributing

For detailed information on how to contribute, see <a href="https://github.com/stotko/stdgpu/blob/master/CONTRIBUTING.md">`CONTRIBUTING`</a>.


## License

Distributed under the Apache 2.0 License. See <a href="https://github.com/stotko/stdgpu/blob/master/LICENSE">`LICENSE`</a> for more information.

stdgpu has been developed as part of the SLAMCast live telepresence system which performs real-time, large-scale 3D scene reconstruction from RGB-D camera images as well as real-time data streaming between a server and an arbitrary number of remote clients.

If you use stdgpu in one of your projects, please cite the following publications:

```
@UNPUBLISHED{stotko2019stdgpu,
    author = {Stotko, Patrick},
     title = {{stdgpu: Efficient STL-like Data Structures on the GPU}},
      year = {2019},
     month = aug,
      note = {arXiv:1908.05936},
       url = {https://arxiv.org/abs/1908.05936}
}
```

```
@article{stotko2019slamcast,
    author = {Stotko, P. and Krumpen, S. and Hullin, M. B. and Weinmann, M. and Klein, R.},
     title = {{SLAMCast: Large-Scale, Real-Time 3D Reconstruction and Streaming for Immersive Multi-Client Live Telepresence}},
   journal = {IEEE Transactions on Visualization and Computer Graphics},
    volume = {25},
    number = {5},
     pages = {2102--2112},
      year = {2019}
}
```


## Contact

Patrick Stotko - <a href="mailto:stotko@cs.uni-bonn.de">stotko@cs.uni-bonn.de</a>
