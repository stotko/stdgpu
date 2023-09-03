<p align="center">
<img src="./docs/_static/stdgpu_logo.png" width="500" />
</p>

<h1 align="center">stdgpu: Efficient STL-like Data Structures on the GPU</h1>

<!-- start badges -->

<p align="center">
<a href="https://github.com/stotko/stdgpu/actions?query=workflow%3A%22Ubuntu+OpenMP%22" alt="Ubuntu">
    <img src="https://github.com/stotko/stdgpu/workflows/Ubuntu%20OpenMP/badge.svg"/>
</a>
<a href="https://github.com/stotko/stdgpu/actions?query=workflow%3A%22Windows+OpenMP%22" alt="Windows">
    <img src="https://github.com/stotko/stdgpu/workflows/Windows%20OpenMP/badge.svg"/>
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
    <img src="https://img.shields.io/badge/docs-Latest-green.svg"/>
</a>
<a href="https://github.com/stotko/stdgpu/blob/master/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/stotko/stdgpu"/>
</a>
</p>

<!-- end badges -->

<b>
<p align="center">
<a style="font-weight:bold" href="#features">Features</a> |
<a style="font-weight:bold" href="#examples">Examples</a> |
<a style="font-weight:bold" href="#getting-started">Getting Started</a> |
<a style="font-weight:bold" href="#contributing">Contributing</a> |
<a style="font-weight:bold" href="#license">License</a> |
<a style="font-weight:bold" href="#contact">Contact</a>
</p>
</b>


<!-- start readme -->

## Features

stdgpu is an open-source library providing several generic GPU data structures for fast and reliable data management. Multiple platforms such as **CUDA**, **OpenMP**, and **HIP** are supported allowing you to rapidly write highly complex **agnostic** and **native** algorithms that look like sequential CPU code but are executed in parallel on the GPU.

- **Productivity**. Previous libraries such as thrust, VexCL, ArrayFire or Boost.Compute focus on the fast and efficient implementation of various algorithms for contiguously stored data to enhance productivity. stdgpu follows an *orthogonal approach* and focuses on *fast and reliable data management* to enable the rapid development of more general and flexible GPU algorithms just like their CPU counterparts.

- **Interoperability**. Instead of providing yet another ecosystem, stdgpu is designed to be a *lightweight container library*. Therefore, a core feature of stdgpu is its interoperability with previous established frameworks, i.e. the thrust library, to enable a *seamless integration* into new as well as existing projects.

- **Maintainability**. Following the trend in recent C++ standards of providing functionality for safer and more reliable programming, the philosophy of stdgpu is to provide *clean and familiar functions* with strong guarantees that encourage users to write *more robust code* while giving them full control to achieve a high performance.

At its heart, stdgpu offers the following GPU data structures and containers:

<table>
<tr align="center">
<td><a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1atomic.html"><code>atomic</code></a> &amp; <a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1atomic__ref.html"><code>atomic_ref</code></a><br>Atomic primitive types and references</td>
<td><a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1bitset.html"><code>bitset</code></a><br>Space-efficient bit array</td>
<td><a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1deque.html"><code>deque</code></a><br>Dynamically sized double-ended queue</td>
</tr>
<tr align="center">
<td><a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1queue.html"><code>queue</code></a> &amp; <a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1stack.html"><code>stack</code></a><br>Container adapters</td>
<td><a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1unordered__map.html"><code>unordered_map</code></a> &amp; <a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1unordered__set.html"><code>unordered_set</code></a><br>Hashed collection of unique keys and key-value pairs</td>
<td><a href="https://stotko.github.io/stdgpu/doxygen/classstdgpu_1_1vector.html"><code>vector</code></a><br>Dynamically sized contiguous array</td>
</tr>
</table>

In addition, stdgpu also provides commonly required functionality in [`algorithm`](https://stotko.github.io/stdgpu/doxygen/algorithm_8h.html), [`bit`](https://stotko.github.io/stdgpu/doxygen/bit_8h.html), [`contract`](https://stotko.github.io/stdgpu/doxygen/contract_8h.html), [`cstddef`](https://stotko.github.io/stdgpu/doxygen/cstddef_8h.html), [`functional`](https://stotko.github.io/stdgpu/doxygen/functional_8h.html), [`iterator`](https://stotko.github.io/stdgpu/doxygen/iterator_8h.html), [`limits`](https://stotko.github.io/stdgpu/doxygen/limits_8h.html), [`memory`](https://stotko.github.io/stdgpu/doxygen/memory_8h.html), [`mutex`](https://stotko.github.io/stdgpu/doxygen/mutex_8cuh.html), [`ranges`](https://stotko.github.io/stdgpu/doxygen/ranges_8h.html), [`utility`](https://stotko.github.io/stdgpu/doxygen/utility_8h.html) to complement the GPU data structures and to increase their usability and interoperability.


## Examples

In order to reliably perform complex tasks on the GPU, stdgpu offers flexible interfaces that can be used in both **agnostic code**, e.g. via the algorithms provided by thrust, as well as in **native code**, e.g. in custom CUDA kernels.

For instance, stdgpu is extensively used in [SLAMCast](https://www.researchgate.net/publication/331303359_SLAMCast_Large-Scale_Real-Time_3D_Reconstruction_and_Streaming_for_Immersive_Multi-Client_Live_Telepresence), a scalable live telepresence system, to implement real-time, large-scale 3D scene reconstruction as well as real-time 3D data streaming between a server and an arbitrary number of remote clients.

**Agnostic code**. In the context of [SLAMCast](https://www.researchgate.net/publication/331303359_SLAMCast_Large-Scale_Real-Time_3D_Reconstruction_and_Streaming_for_Immersive_Multi-Client_Live_Telepresence), a simple task is the integration of a range of updated blocks into the duplicate-free set of queued blocks for data streaming which can be expressed very conveniently:

```cpp
#include <stdgpu/cstddef.h>             // stdgpu::index_t
#include <stdgpu/iterator.h>            // stdgpu::make_device
#include <stdgpu/unordered_set.cuh>     // stdgpu::unordered_set

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

**Native code**. More complex operations such as the creation of the duplicate-free set of updated blocks or other algorithms can be implemented natively, e.g. in custom CUDA kernels with stdgpu's CUDA backend enabled:

```cpp
#include <stdgpu/cstddef.h>             // stdgpu::index_t
#include <stdgpu/unordered_map.cuh>     // stdgpu::unordered_map
#include <stdgpu/unordered_set.cuh>     // stdgpu::unordered_set

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

More examples can be found in the [`examples`](https://github.com/stotko/stdgpu/tree/master/examples) directory.


## Getting Started

stdgpu requires a **C++17 compiler** as well as minimal backend dependencies and can be easily built and integrated into your project via **CMake**:

- [Building From Source](https://stotko.github.io/stdgpu/getting_started/building_from_source.html)
- [Integrating Into Your Project](https://stotko.github.io/stdgpu/getting_started/integrating_into_your_project.html)

More guidelines as well as a comprehensive introduction into the design and API of stdgpu can be found in the [documentation](https://stotko.github.io/stdgpu).


## Contributing

For detailed information on how to contribute, see the [Contributing](https://stotko.github.io/stdgpu/development/contributing.html) section in the documentation.


## License

Distributed under the Apache 2.0 License. See [`LICENSE`](https://github.com/stotko/stdgpu/blob/master/LICENSE) for more information.

If you use stdgpu in one of your projects, please cite the following publications:

[**stdgpu: Efficient STL-like Data Structures on the GPU**](https://www.researchgate.net/publication/335233070_stdgpu_Efficient_STL-like_Data_Structures_on_the_GPU)

```bib
@UNPUBLISHED{stotko2019stdgpu,
    author = {Stotko, P.},
     title = {{stdgpu: Efficient STL-like Data Structures on the GPU}},
      year = {2019},
     month = aug,
      note = {arXiv:1908.05936},
       url = {https://arxiv.org/abs/1908.05936}
}
```

[**SLAMCast: Large-Scale, Real-Time 3D Reconstruction and Streaming for Immersive Multi-Client Live Telepresence**](https://www.researchgate.net/publication/331303359_SLAMCast_Large-Scale_Real-Time_3D_Reconstruction_and_Streaming_for_Immersive_Multi-Client_Live_Telepresence)

```bib
@article{stotko2019slamcast,
    author = {Stotko, P. and Krumpen, S. and Hullin, M. B. and Weinmann, M. and Klein, R.},
     title = {{SLAMCast: Large-Scale, Real-Time 3D Reconstruction and Streaming for Immersive Multi-Client Live Telepresence}},
   journal = {IEEE Transactions on Visualization and Computer Graphics},
    volume = {25},
    number = {5},
     pages = {2102--2112},
      year = {2019},
     month = may
}
```


## Contact

Patrick Stotko - [stotko@cs.uni-bonn.de](mailto:stotko@cs.uni-bonn.de)

<!-- end readme -->
