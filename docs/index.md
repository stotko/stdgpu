# stdgpu: Efficient STL-like Data Structures on the GPU

```{include} ../README.md
:relative-docs: docs/
:relative-images:
:start-after: <!-- start badges -->
:end-before: <!-- end badges -->
```

## Features

stdgpu is an open-source library providing **generic GPU data structures** for fast and reliable data management.

- Lightweight C++17 library with minimal dependencies
- **CUDA**, **OpenMP**, and **HIP (experimental)** backends
- Familiar STL-like GPU containers
- High-level, *agnostic* container functions like `insert(begin, end)`, to write shared C++ code
- Low-level, *native* container functions like `find(key)`, to write custom CUDA kernels, etc.
- Interoperability with [thrust](https://github.com/NVIDIA/thrust) GPU algorithms

Instead of providing yet another ecosystem, stdgpu is designed to be a *lightweight container library*. Previous libraries such as thrust, VexCL, ArrayFire or Boost.Compute focus on the fast and efficient implementation of various algorithms and only operate on contiguously stored data. stdgpu follows an *orthogonal approach* and focuses on *fast and reliable data management* to enable the rapid development of more general and flexible GPU algorithms just like their CPU counterparts.

At its heart, stdgpu offers the following GPU data structures and containers:


:::::{grid} 2 2 3 3
:gutter: 3 3 4 4

::::{grid-item-card}
:text-align: center

**{stdgpu}`stdgpu::atomic`** & **{stdgpu}`stdgpu::atomic_ref`**
^^^
Atomic primitive types and references

::::

::::{grid-item-card}
:text-align: center

**{stdgpu}`stdgpu::bitset`**
^^^
Space-efficient bit array

::::

::::{grid-item-card}
:text-align: center

**{stdgpu}`stdgpu::deque`**
^^^
Dynamically sized double-ended queue

::::

::::{grid-item-card}
:text-align: center

**{stdgpu}`stdgpu::queue`** & **{stdgpu}`stdgpu::stack`**
^^^
Container adapters

::::

::::{grid-item-card}
:text-align: center

**{stdgpu}`stdgpu::unordered_map`** & **{stdgpu}`stdgpu::unordered_set`**
^^^
Hashed collection of unique keys and key-value pairs

::::

::::{grid-item-card}
:text-align: center

**{stdgpu}`stdgpu::vector`**
^^^
Dynamically sized contiguous array

::::

:::::


In addition, stdgpu also provides further commonly used helper functionality in **{stdgpu}`algorithm`**, **{stdgpu}`bit`**, **{stdgpu}`contract`**, **{stdgpu}`cstddef`**, **{stdgpu}`execution`**, **{stdgpu}`functional`**, **{stdgpu}`iterator`**, **{stdgpu}`limits`**, **{stdgpu}`memory`**, **{stdgpu}`mutex`**, **{stdgpu}`numeric`**, **{stdgpu}`ranges`**, **{stdgpu}`type_traits`**, **{stdgpu}`utility`**.


## Examples

```{include} ../README.md
:relative-docs: docs/
:relative-images:
:start-after: <!-- start examples -->
:end-before: <!-- end examples -->
```


## Citation

```{include} ../README.md
:relative-docs: docs/
:relative-images:
:start-after: <!-- start citation -->
:end-before: <!-- end citation -->
```


```{toctree}
:hidden:

Overview <self>
```

```{toctree}
:hidden:
:caption: Getting Started

getting_started/building_from_source
getting_started/integrating_into_your_project
```

```{toctree}
:hidden:
:caption: API Reference

api/chapters
doxygen/modules
doxygen/annotated
doxygen/files
```

```{toctree}
:hidden:
:caption: Development

development/contributing
development/changelog
License <https://github.com/stotko/stdgpu/blob/master/LICENSE>
```
