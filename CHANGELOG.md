# Changelog

All notable changes to this project will be documented in this file. This project adheres to [Semantic Versioning](http://semver.org/).


## [stdgpu 1.1.0](https://github.com/stotko/stdgpu/releases/tag/1.1.0) (2019-11-22)

After a stabilization and cleanup phase, the next version of *stdgpu* is available.

**New Features & Enhancements**

- cmake: Improve compute capability detection [\#8](https://github.com/stotko/stdgpu/pull/8) [\#28](https://github.com/stotko/stdgpu/pull/28)
- cmake: Add option `STDGPU_BUILD_SHARED_LIBS` to build the project as a shared library [\#14](https://github.com/stotko/stdgpu/pull/14)
- unordered_map,unordered_set: Improve reliability [\#25](https://github.com/stotko/stdgpu/pull/25)
- platform: Add `STDGPU_DEVICE_ONLY` annotation macro [\#7](https://github.com/stotko/stdgpu/pull/7)
- test: Upgrade googletest to 1.10.0 [\#6](https://github.com/stotko/stdgpu/pull/6)
- Refactor internal code structure and move platform-specific code to a dedicated CUDA backend [\#1](https://github.com/stotko/stdgpu/pull/1) [\#2](https://github.com/stotko/stdgpu/pull/2) [\#4](https://github.com/stotko/stdgpu/pull/4) [\#5](https://github.com/stotko/stdgpu/pull/5) [\#9](https://github.com/stotko/stdgpu/pull/9) [\#10](https://github.com/stotko/stdgpu/pull/10) [\#11](https://github.com/stotko/stdgpu/pull/11) [\#16](https://github.com/stotko/stdgpu/pull/16) [\#19](https://github.com/stotko/stdgpu/pull/19) [\#23](https://github.com/stotko/stdgpu/pull/23) [\#24](https://github.com/stotko/stdgpu/pull/24) [\#27](https://github.com/stotko/stdgpu/pull/27)

**Bug Fixes**

- atomic: Fix missing template type names in the function definitions [\#30](https://github.com/stotko/stdgpu/pull/30)
- atomic: Fix bit shift in unit test [\#18](https://github.com/stotko/stdgpu/pull/18)
- cmake: Workaround unspecified CUDA directories on Windows [\#15](https://github.com/stotko/stdgpu/pull/15)
- cmake,src: Handle format warnings [\#29](https://github.com/stotko/stdgpu/pull/29)
- deque,unordered_map,unordered_set,vector: Fix missing typename [\#17](https://github.com/stotko/stdgpu/pull/17)
- deque,vector: Remove unreliable validity check in unit test [\#20](https://github.com/stotko/stdgpu/pull/20)
- memory: Workaround possible compilation failures [\#26](https://github.com/stotko/stdgpu/pull/26)
- mutex: Fix typo in test name [\#21](https://github.com/stotko/stdgpu/pull/21)
- unordered_map,unordered_set: Workaround while loop timeouts [\#3](https://github.com/stotko/stdgpu/pull/3)
- unordered_map,unordered_set: Fix corner case in bucket computation [\#22](https://github.com/stotko/stdgpu/pull/22)


## [stdgpu 1.0.0](https://github.com/stotko/stdgpu/releases/tag/1.0.0) (2019-08-19)

This is the first public version of *stdgpu*, an open-source C++ library providing generic STL-like GPU data structures for fast and reliable data management. The main components of the library are:

- **Core**: A collection of core features including configuration and platform management, a simple contract interface as well as a robust memory and iterator concept.
- **Container**: A set of robust containers for GPU programming with an STL-like design consisting of sequential and hash-based data structures.
- **Utilities**: A variety of utility functions supporting the container component and general GPU programming.
