# Changelog

All notable changes to this project will be documented in this file. This project adheres to [Semantic Versioning](http://semver.org/).


## [stdgpu 1.2.0](https://github.com/stotko/stdgpu/releases/tag/1.2.0) (2020-01-28)

This version of *stdgpu* introduces a lightweight backend system including CUDA and OpenMP backends, the integration of Azure Pipelines CI as well as codecov CI, support for the Clang compiler, removal of unnecessary requirements to the container's value types, as well as significant improvements to the test coverage and the documentation.

**New Features & Enhancements**

- General: Add backend system [\#31](https://github.com/stotko/stdgpu/pull/31)
- General: Add OpenMP backend [\#32](https://github.com/stotko/stdgpu/pull/32) [\#59](https://github.com/stotko/stdgpu/pull/59)
- General: Add Azure Pipelines CI [\#34](https://github.com/stotko/stdgpu/pull/34) [\#37](https://github.com/stotko/stdgpu/pull/37) [\#41](https://github.com/stotko/stdgpu/pull/41)
- General: Add code coverage report generation [\#65](https://github.com/stotko/stdgpu/pull/65)
- General: Add codecov CI task [\#72](https://github.com/stotko/stdgpu/pull/72)
- General: Add Clang support [\#40](https://github.com/stotko/stdgpu/pull/40)
- General: Add changelog file [\#48](https://github.com/stotko/stdgpu/pull/48)
- General: Add contributing file [\#49](https://github.com/stotko/stdgpu/pull/49)
- General: Add issue templates [\#81](https://github.com/stotko/stdgpu/pull/81)
- Container: Remove `DefaultConstructible` requirement from template type [\#58](https://github.com/stotko/stdgpu/pull/58)
- Container: Add `get_allocator()` function [\#56](https://github.com/stotko/stdgpu/pull/56)
- bitset: Add further missing member functions [\#53](https://github.com/stotko/stdgpu/pull/53)
- deque: Add `at()`, `shrink_to_fit()` and remove `CopyAssignable` requirement from type `T` [\#45](https://github.com/stotko/stdgpu/pull/45)
- memory: Add `safe_host_allocator` and deprecate `safe_pinned_host_allocator` [\#36](https://github.com/stotko/stdgpu/pull/36)
- memory: Add and use `destroy*` functions [\#60](https://github.com/stotko/stdgpu/pull/60)
- memory: Add `allocator_traits` and deprecate old specialized version [\#61](https://github.com/stotko/stdgpu/pull/61) [\#66](https://github.com/stotko/stdgpu/pull/66)
- mutex: Add `mutex_array::reference` class and deprecate `mutex_ref` [\#55](https://github.com/stotko/stdgpu/pull/55) [\#63](https://github.com/stotko/stdgpu/pull/63)
- unordered_map,unordered_set: Add single-parameter `createDeviceObject()` function [\#46](https://github.com/stotko/stdgpu/pull/46) [\#52](https://github.com/stotko/stdgpu/pull/52)
- vector: Add `at()`, `shrink_to_fit()` and remove `CopyAssignable` requirement from type `T` [\#44](https://github.com/stotko/stdgpu/pull/44)
- README: Improve consistency with doxygen version [\#42](https://github.com/stotko/stdgpu/pull/42)
- README: Add badges [\#35](https://github.com/stotko/stdgpu/pull/35) [\#79](https://github.com/stotko/stdgpu/pull/79) [\#85](https://github.com/stotko/stdgpu/pull/85) [\#86](https://github.com/stotko/stdgpu/pull/86)
- README,doc: Significantly improve description and readability [\#50](https://github.com/stotko/stdgpu/pull/50)
- doc: Include config.h and cleanup macro definitions [\#47](https://github.com/stotko/stdgpu/pull/47)
- scripts: Improve console output and internal structure [\#33](https://github.com/stotko/stdgpu/pull/33)
- scripts: Port install script to native CMake install command-line interface [\#82](https://github.com/stotko/stdgpu/pull/82)
- test: Adjust test array sizes and build flags [\#64](https://github.com/stotko/stdgpu/pull/64)
- test: Explicitly instantiate templates [\#70](https://github.com/stotko/stdgpu/pull/70)
- test: Also include deprecated functions into unit tests [\#80](https://github.com/stotko/stdgpu/pull/80)
- test: Improve coverage of several (member) functions [\#74](https://github.com/stotko/stdgpu/pull/74) [\#75](https://github.com/stotko/stdgpu/pull/75) [\#76](https://github.com/stotko/stdgpu/pull/76) [\#77](https://github.com/stotko/stdgpu/pull/77) [\#78](https://github.com/stotko/stdgpu/pull/78) [\#84](https://github.com/stotko/stdgpu/pull/84)

**Bug Fixes**

- README: Fix alignment of title [\#43](https://github.com/stotko/stdgpu/pull/43)
- atomic: Fix compare_exchange and add more operators as well as tests [\#83](https://github.com/stotko/stdgpu/pull/83)
- cmake: Fix minimum required version [\#71](https://github.com/stotko/stdgpu/pull/71)
- deque: Fix compilation error when calling `device_range()` [\#67](https://github.com/stotko/stdgpu/pull/67)
- unordered_base: Fix compilation errors with CUDA backend [\#69](https://github.com/stotko/stdgpu/pull/69)
- unordered_map,unordered_set: Fix delegate calls to unordered_base [\#68](https://github.com/stotko/stdgpu/pull/68)
- vector: Disallow non-defined bool specialization [\#57](https://github.com/stotko/stdgpu/pull/57)

**Deprecated Features**

- memory: `safe_pinned_host_allocator`, `default_allocator_traits`
- mutex: `mutex_ref`
- unordered_map,unordered_set: `createDeviceObject(index_t, index_t)`, `excess_count()`, `total_count()`


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
