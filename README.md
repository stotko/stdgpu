<p align="center">
<img src="./doc/stdgpu_logo.png" width="400" />
</p>

<p align="center">
<h1>stdgpu: Efficient STL-like Data Structures on the GPU</h1>
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

* C++14 compiler:
    * Ubuntu 18.04: GCC 7: `sudo apt install build-essential`
    * Windows: Visual Studio 2019
* CUDA 10.0: `https://developer.nvidia.com/cuda-downloads`
* CMake 3.13: `https://apt.kitware.com/`
* Doxygen 1.8.13 (optional): `sudo apt install doxygen`

Older compiler versions may also work but are currently considered experimental and untested.


### Building

Use the setup script to perform a clean build (release mode in this case, with default installation path to `./bin`) and run the tests:

```sh
sh scripts/setup_release.sh
```

Building the library - in the respective build mode - afterwards can be done using

```sh
sh scripts/build_release.sh
```

and the tests can be executed using

```sh
sh scripts/run_tests_release.sh
```

The documentation can be build (optional, requires Doxygen) using

```sh
sh scripts/create_documentation.sh
```

To install the library at the configured path, use

```sh
sh scripts/install.sh
```


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

To configure the library, you can set the following options:

Build:

* `STDGPU_SETUP_COMPILER_FLAGS`: Constructs the compiler flags, default: `ON` if standalone, `OFF` if included via `add_subdirectory`
* `STDGPU_BUILD_EXAMPLES`: Build the example, default: `ON`
* `STDGPU_BUILD_TESTS`: Build the unit tests, default: `ON`

Configuration:

* `STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING`: Enable warnings when auxiliary arrays are allocated in memory API, default: `OFF`
* `STDGPU_ENABLE_CONTRACT_CHECKS`: Enable contract checks, default: `OFF` if `CMAKE_BUILD_TYPE` equals `Release` or `MinSizeRel`, `ON` otherwise
* `STDGPU_ENABLE_MANAGED_ARRAY_WARNING`: Enable warnings when managed memory is initialized on the host side but should be on device in memory API, default: `OFF`
* `STDGPU_USE_32_BIT_INDEX`: Use 32-bit instead of 64-bit signed integer for `index_t`, default: `ON`
* `STDGPU_USE_FAST_DESTROY`: Use fast destruction of allocated arrays (filled with a default value) by omitting destructor calls in memory API, default: `OFF`
* `STDGPU_USE_FIBONACCI_HASHING`: Use Fibonacci Hashing instead of Modulo to compute hash bucket indices, default: `ON`


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

For more examples, please refer to the `examples` directory.


## Contributing

If you encounter a bug or want to propose a new feature, please open an issue or pull request.


## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

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

Patrick Stotko - stotko@cs.uni-bonn.de
