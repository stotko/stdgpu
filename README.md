<p align="center">
<img src="./doc/stdgpu_logo.png" width="500" />
</p>

<h1 align="center">stdgpu: Efficient STL-like Data Structures on the GPU</h1>

<p align="center">
<a href="https://dev.azure.com/patrickstotko/stdgpu/_build/latest?definitionId=3&branchName=master" alt="Build Status">
    <img src="https://dev.azure.com/patrickstotko/stdgpu/_apis/build/status/stotko.stdgpu?branchName=master"/>
</a>
<a href="https://stotko.github.io/stdgpu" alt="Documentation">
    <img src="https://img.shields.io/badge/docs-doxygen-blue.svg"/>
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
<a href="#about-the-project">About the Project</a> |
<a href="#getting-started">Getting Started</a> |
<a href="#usage">Usage</a> |
<a href="#contributing">Contributing</a> |
<a href="#license">License</a> |
<a href="#contact">Contact</a>
</p>


## About the Project

stdgpu is an open-source library which provides several generic GPU data structures for fast and reliable data management. Our library aims to extend previous established frameworks, therefore bridging the gap between CPU and GPU computing. This way, it provides clean and familiar interfaces and integrates seamlessly into new as well as existing projects. stdgpu has been developed as part of the SLAMCast live telepresence system which performs real-time, large-scale 3D scene reconstruction from RGB-D camera images as well as real-time data streaming between a server and an arbitrary number of remote clients. We hope to foster further developments towards unified CPU and GPU computing and welcome contributions from the community.


## Getting Started

To compile the library, please make sure to fulfill the build requirements and execute the respective build scripts.


### Prerequisites

The following components (or newer versions) are required to build the library:

- C++14 compiler (one of the following):
    - Ubuntu 18.04:
        - GCC 7: `sudo apt install build-essential`
        - Clang 6: `sudo apt install clang`
    - Windows:
        - MSVC 19.2x (Visual Studio 2019)
- CMake 3.13: `https://apt.kitware.com/`
- Doxygen 1.8.13 (optional): `sudo apt install doxygen`

Depending on the used backend, additional components (or newer versions) are required:

- CUDA Backend:
    - CUDA 10.0: `https://developer.nvidia.com/cuda-downloads`
- OpenMP Backend:
    - OpenMP 2.0
    - thrust 1.9.3: included in CUDA, but also available at `https://github.com/thrust/thrust`

Older compiler versions may also work but are currently considered experimental and untested.


### Building

For convenience, we provide several cross-platform scripts to build the project. Note that some scripts depend on the build type, so there are scripts for both `debug` and `release` builds.

Command | Effect
--- | ---
<code>sh&nbsp;scripts/setup_&lt;build_type&gt;.sh</code> | Performs a full clean build of the project. Removes old build, configures the project (build path: `./build`), builds the project, and runs the unit tests.
<code>sh&nbsp;scripts/build_&lt;build_type&gt;.sh</code> | (Re-)Builds the project. Requires that project is configured (or set up).
<code>sh&nbsp;scripts/run_tests_&lt;build_type&gt;.sh</code> | Runs the unit tests. Requires that project is built.
<code>sh&nbsp;scripts/create_documentation.sh</code> | Builds the documentation locally. Requires doxygen and that project is configured (or set up).
<code>sh&nbsp;scripts/install.sh</code> | Installs the project at the configured install path (default: `./bin`).


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
`STDGPU_BUILD_SHARED_LIBS` | Builds the project as a shared library, if set to `ON`, or as a static library, if set to `OFF` | `BUILD_SHARED_LIBS`
`STDGPU_BUILD_EXAMPLES` | Build the example | `ON`
`STDGPU_BUILD_TESTS` | Build the unit tests | `ON`

In addition, the implementation of some functionality can be controlled via configuration options:

Configuration Option | Effect | Default
--- | --- | ---
`STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING` | Enable warnings when auxiliary arrays are allocated in memory API | `OFF`
`STDGPU_ENABLE_CONTRACT_CHECKS` | Enable contract checks | `OFF` if `CMAKE_BUILD_TYPE` equals `Release` or `MinSizeRel`, `ON` otherwise
`STDGPU_ENABLE_MANAGED_ARRAY_WARNING` | Enable warnings when managed memory is initialized on the host side but should be on device in memory API | `OFF`
`STDGPU_USE_32_BIT_INDEX` | Use 32-bit instead of 64-bit signed integer for `index_t` | `ON`
`STDGPU_USE_FAST_DESTROY` | Use fast destruction of allocated arrays (filled with a default value) by omitting destructor calls in memory API | `OFF`
`STDGPU_USE_FIBONACCI_HASHING` | Use Fibonacci Hashing instead of Modulo to compute hash bucket indices | `ON`


### Examples

This library offers flexible interfaces to reliably perform complex tasks on the GPU such as the creation of the update set at the server component of the related SLAMCast system:

```cpp
namespace stdgpu
{
template <>
struct hash<short3>
{
    inline STDGPU_HOST_DEVICE std::size_t
    operator()(const short3& key) const
    {
        return key.x * 73856093
             ^ key.y * 19349669
             ^ key.z * 83492791;
    }
};
}

// Spatial hash map for voxel block management
using block_map = stdgpu::unordered_map<short3, voxel*>;


__global__ void
compute_update_set(const short3* blocks,
                   const stdgpu::index_t n,
                   const block_map tsdf_block_map,
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

On the other hand, simpler tasks such as the integration of a range of values can be expressed very conveniently:

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

More examples can be found in the <a href="https://github.com/stotko/stdgpu/tree/master/examples">`examples`</a> directory.


## Contributing

For detailed information on how to contribute, see <a href="https://github.com/stotko/stdgpu/blob/master/CONTRIBUTING.md">`CONTRIBUTING`</a>.


## License

Distributed under the Apache 2.0 License. See <a href="https://github.com/stotko/stdgpu/blob/master/LICENSE">`LICENSE`</a> for more information.

If you use this library in one of your projects, please also cite the following publications:

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
