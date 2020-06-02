# Changelog

All notable changes to this project will be documented in this file. This project adheres to [Semantic Versioning](http://semver.org/).


## [stdgpu 1.3.0](https://github.com/stotko/stdgpu/releases/tag/1.3.0) (2020-06-02)

This release of *stdgpu* introduces a new *experimental* HIP backend adding support for AMD GPUs, significant improvements to the API documentation as well as many new code examples, the integration of clang-tidy and cppcheck in the CI, as well as a tremendous amount of warning fixes to enable clean builds at very high warning levels.

**New Features & Enhancements**

- General: Add experimental HIP backend [\#121](https://github.com/stotko/stdgpu/pull/121) [\#143](https://github.com/stotko/stdgpu/pull/143)
- General: Add support for Compute Capability 3.0 in CUDA backend [\#153](https://github.com/stotko/stdgpu/pull/153)
- General: Add clang-tidy support [\#129](https://github.com/stotko/stdgpu/pull/129) [\#138](https://github.com/stotko/stdgpu/pull/138)
- General: Add cppcheck support [\#149](https://github.com/stotko/stdgpu/pull/149)
- General: Add CI job for documentation creation [\#109](https://github.com/stotko/stdgpu/pull/109)
- General: Deprecate misleading/obsolete cmake options [\#103](https://github.com/stotko/stdgpu/pull/103)
- atomic: Make all operations follow sequentially consistent ordering [\#176](https://github.com/stotko/stdgpu/pull/176)
- atomic: Add backend documentation of template parameter [\#177](https://github.com/stotko/stdgpu/pull/177)
- atomic: Cleanup backend-specific internals of CUDA backend [\#152](https://github.com/stotko/stdgpu/pull/152)
- bit: Add `ceil2` and `floor2` functions [\#105](https://github.com/stotko/stdgpu/pull/105)
- bit: Rename functions to match most recent draft of C++20 [\#110](https://github.com/stotko/stdgpu/pull/110)
- bitset: Remove dependency to cstdlib [\#145](https://github.com/stotko/stdgpu/pull/145)
- cstddef: Hide initializers for clearer documentation [\#166](https://github.com/stotko/stdgpu/pull/166)
- cstdlib: Deprecate `sizedivPow2` [\#161](https://github.com/stotko/stdgpu/pull/161)
- limits: Add implementation for non-specialized template and documentation for every type [\#167](https://github.com/stotko/stdgpu/pull/167)
- memory: Add `construct_at` function [\#95](https://github.com/stotko/stdgpu/pull/95)
- memory: Cleanup global variables and simplify `allocate`/`deallocate` logic [\#104](https://github.com/stotko/stdgpu/pull/104)
- memory: Improve `construct*` and `destroy*` unit tests [\#175](https://github.com/stotko/stdgpu/pull/175)
- platform: Add automatic dispatching of backend-specific definitions [\#119](https://github.com/stotko/stdgpu/pull/119)
- platform: Change detection of device code for OpenMP [\#174](https://github.com/stotko/stdgpu/pull/174)
- ranges: Add `size()` and `empty()` functions as well as additional constructors [\#122](https://github.com/stotko/stdgpu/pull/122)
- ranges: Add `index64_t` constructor and deprecate `index_t` version [\#102](https://github.com/stotko/stdgpu/pull/102)
- unordered_map,unordered_set: Improve robustness of Fibonacci Hashing [\#111](https://github.com/stotko/stdgpu/pull/111)
- README,doc: Significantly improve introduction, examples, and documentation [\#114](https://github.com/stotko/stdgpu/pull/114) [\#116](https://github.com/stotko/stdgpu/pull/116) [\#162](https://github.com/stotko/stdgpu/pull/162) [\#165](https://github.com/stotko/stdgpu/pull/165) [\#170](https://github.com/stotko/stdgpu/pull/170) [\#171](https://github.com/stotko/stdgpu/pull/171) [\#172](https://github.com/stotko/stdgpu/pull/172) [\#181](https://github.com/stotko/stdgpu/pull/181)
- doc: Group all class and function definitions into modules [\#169](https://github.com/stotko/stdgpu/pull/169)
- doc: Cleanup unnecessary documentation [\#168](https://github.com/stotko/stdgpu/pull/168)
- examples: Add many new examples and improve existing ones [\#173](https://github.com/stotko/stdgpu/pull/173)
- test: Disable unused GMock [\#160](https://github.com/stotko/stdgpu/pull/160)
- cmake: Make installable package relocatable [\#180](https://github.com/stotko/stdgpu/pull/180)
- cmake: Add option to treat warnings as errors [\#108](https://github.com/stotko/stdgpu/pull/108)
- cmake: Generate compile flags more robustly [\#128](https://github.com/stotko/stdgpu/pull/128)
- cmake: Simplify architecture flag generation in CUDA backend [\#154](https://github.com/stotko/stdgpu/pull/154)
- cmake: Install backend-specific find modules in subdirectories [\#117](https://github.com/stotko/stdgpu/pull/117)
- cmake: Update support for CMake 3.17+ [\#123](https://github.com/stotko/stdgpu/pull/123)

**Bug Fixes**

- General: Increase warning level and fix conversion and float-equal warnings [\#98](https://github.com/stotko/stdgpu/pull/98)
- General: Increase MSVC warning level and fix related warnings [\#107](https://github.com/stotko/stdgpu/pull/107) [\#156](https://github.com/stotko/stdgpu/pull/156)
- General: Fix Clang warnings [\#91](https://github.com/stotko/stdgpu/pull/91) [\#147](https://github.com/stotko/stdgpu/pull/147)
- General: Fix format warnings [\#101](https://github.com/stotko/stdgpu/pull/101)
- General: Fix sign-conversion warnings [\#100](https://github.com/stotko/stdgpu/pull/100)
- General: Fix shadow warnings [\#90](https://github.com/stotko/stdgpu/pull/90)
- General: Fix numerous clang-tidy warnings [\#130](https://github.com/stotko/stdgpu/pull/130) [\#131](https://github.com/stotko/stdgpu/pull/131) [\#132](https://github.com/stotko/stdgpu/pull/132) [\#133](https://github.com/stotko/stdgpu/pull/133) [\#134](https://github.com/stotko/stdgpu/pull/134) [\#135](https://github.com/stotko/stdgpu/pull/135) [\#136](https://github.com/stotko/stdgpu/pull/136) [\#137](https://github.com/stotko/stdgpu/pull/137) [\#140](https://github.com/stotko/stdgpu/pull/140) [\#141](https://github.com/stotko/stdgpu/pull/141)
- examples: Pass containers by reference for OpenMP backend [\#182](https://github.com/stotko/stdgpu/pull/182)
- src,test: Improve consistency and cleanup includes [\#118](https://github.com/stotko/stdgpu/pull/118)
- test: Fix missing namespace for `uint8_t` [\#142](https://github.com/stotko/stdgpu/pull/142)
- test: Pass containers by const reference to functors [\#158](https://github.com/stotko/stdgpu/pull/158)
- test: Fix double-promotion warnings in backend code [\#151](https://github.com/stotko/stdgpu/pull/151)
- test: Fix conversion warning and missing namespace [\#124](https://github.com/stotko/stdgpu/pull/124)
- test: Fix missing include in device_info cpp files [\#120](https://github.com/stotko/stdgpu/pull/120)
- bit: Fix potential negative bit shift in unit test [\#159](https://github.com/stotko/stdgpu/pull/159)
- bit,bitset: Fix missing post-conditions and remove unnecessary dependency [\#112](https://github.com/stotko/stdgpu/pull/112)
- bitset: Fix deprecated-copy warning [\#144](https://github.com/stotko/stdgpu/pull/144)
- compiler: Fix NVCC detection [\#155](https://github.com/stotko/stdgpu/pull/155)
- compiler,platform: Use unique numbers as internal macro definitions [\#139](https://github.com/stotko/stdgpu/pull/139)
- contract: Enforce user semicolon for all possible expansions [\#148](https://github.com/stotko/stdgpu/pull/148) [\#150](https://github.com/stotko/stdgpu/pull/150)
- limits: Suppress long double device code warning with MSVC [\#178](https://github.com/stotko/stdgpu/pull/178)
- platform: Move `STDGPU_HAS_CXX_17` to compiler [\#146](https://github.com/stotko/stdgpu/pull/146)
- ranges: Fix compilation with 64-bit index type [\#157](https://github.com/stotko/stdgpu/pull/157)
- ranges: Fix compilation error with select functor [\#125](https://github.com/stotko/stdgpu/pull/125)
- deque,vector: Fix overflow in test [\#99](https://github.com/stotko/stdgpu/pull/99)
- doc: Fix several minor documentation bugs [\#164](https://github.com/stotko/stdgpu/pull/164)
- scripts: Use released thrust version [\#126](https://github.com/stotko/stdgpu/pull/126)
- cmake: Fix error with unspecified build type [\#179](https://github.com/stotko/stdgpu/pull/179)
- cmake: Fix parsing of thrust version [\#163](https://github.com/stotko/stdgpu/pull/163)
- cmake: Workaround bug in imported rocthrust target name [\#127](https://github.com/stotko/stdgpu/pull/127)
- cmake: Properly handle CUDA toolkit dependency [\#96](https://github.com/stotko/stdgpu/pull/96)
- cmake: Add missing dependency checks in package config [\#94](https://github.com/stotko/stdgpu/pull/94)
- cmake: Fix selection of header files for installation [\#93](https://github.com/stotko/stdgpu/pull/93)
- cmake: Fix inconsistent thrust detection across the backends [\#92](https://github.com/stotko/stdgpu/pull/92)
- CI: Fix codecov task [\#113](https://github.com/stotko/stdgpu/pull/113)
- CI: Fix potentially missing OpenMP runtime package [\#106](https://github.com/stotko/stdgpu/pull/106)

**Deprecated Features**

- bit: `ispow2()`, `log2pow2()`, `mod2()`
- cstdlib: `sizedivPow2(std::size_t, std::size_t)`, `sizediv_t`
- memory: `safe_pinned_host_allocator`, `default_allocator_traits`
- mutex: `mutex_ref`
- ranges: `device_range(T*, index_t)`, `host_range(T*, index_t)`, non-const `begin()` and `end()` member functions
- unordered_map,unordered_set: `createDeviceObject(index_t, index_t)`, `excess_count()`, `total_count()`
- CMake Configuration Options: `STDGPU_ENABLE_AUXILIARY_ARRAY_WARNING`, `STDGPU_ENABLE_MANAGED_ARRAY_WARNING`, `STDGPU_USE_FAST_DESTROY`, `STDGPU_USE_FIBONACCI_HASHING`



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
