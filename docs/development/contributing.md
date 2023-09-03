# Contributing

Thank you for your interest into contributing to stdgpu! There are multiple ways for you to contribute and to help advancing the library:

- Report a bug or help triaging existing bugs
- Propose a new feature that is currently missing
- Submit a fix for a bug
- Submit a new feature or an improvement to an existing feature
- Write some documentation


## Reporting Bugs / Proposing New Features

Not all contributions require that you actually submit some code. Simply reporting a bug, which you have discovered, or proposing a feature, that is currently missing and you would like to see, is great way to help with the development of the library.

We track bugs and feature requests with [GitHub Issues](https://github.com/stotko/stdgpu/issues). Before opening a new issue, please check whether the problem has already been reported and respond there.

If your issue has not been reported yet, feel free to open a new issue and describe the problem. Please provide a clear summary of the problem, what behavior you have expected and what behavior you have actually observed. If possible, create a *Minimal Reproducable Example* that demonstrates the problem.


## Submitting Bug Fixes / Features / Documentation

We also highly welcome code contributions via [GitHub Pull Requests](https://github.com/stotko/stdgpu/pulls) which --- after acceptance --- will be offered under the **Apache 2.0** license, see the [LICENSE](https://github.com/stotko/stdgpu/blob/master/LICENSE).

To submit your changes, follow the standard *Fork & Pull Request Workflow*:

1. Fork the project and switch to a new suitably named branch.
3. Create one or more commits that reflect the changes you have made. Each commit should be self-contained, atomic and buildable. Thus, split multiple features into different commits and include fixups in the related commit instead of creating a new one.
4. If you add new functionality such as a new class or function, please also add respective documentation and tests for it.
5. Push the branch to your fork.
6. Open a new pull request with a brief motivation of the problem and how you addressed it in your changes. If there already exists a related [GitHub Issue](https://github.com/stotko/stdgpu/issues), please mention it there as well.

You can find more information for the development of your changes in the [building guide](../getting_started/building_from_source.md) as well as on the following pages:


:::::{grid} 2 2 3 3
:gutter: 3 3 4 4

::::{grid-item-card}
:link: contributing/coding_style
:link-type: doc

**Coding Style**
^^^
How to format the source code.

::::

::::{grid-item-card}
:link: contributing/documentation
:link-type: doc

**Documentation**
^^^
How to build the documentation.

::::

::::{grid-item-card}
:link: contributing/tests
:link-type: doc

**Tests**
^^^
How to run the unit tests.

::::

:::::


After you have submitted a pull request, your changes will be reviewed and you will receive some feedback:

1. **Automatic review**. Each pull request will be automatically tested using *Continuous Integration* tools. If a test fails, please take a look at the error and update your code accordingly.
2. **Manual human review**. In addition, your code will be manually reviewed by the project maintainers. They will assist you in improving your contribution, so please  incorporate their feedback.

After the review is complete and all tests pass, your pull request will be accepted and finally merged!


## Project Structure

stdgpu is structured according to common <a href="https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/spec/data/spec.bs">project layout conventions</a> which includes the following directories:

- `benchmarks/stdgpu/`: The benchmarks of the library. The actual code lies in the `*.inc` files which are then included and compiled in backend-specific source files.
- `cmake/`: Additional CMake scripts used for building and development. Backend-specific code lies in the respective subdirectories.
- `docs/`: Directory containing the sources of this documentation. In addition, the API documentation is specified directly in the header files of the actual library.
- `examples/`: Various example snippets, also split and compiled dependent on the chosen backend.
- `src/stdgpu/`: Source directory of the library.
- `tests/stdgpu/`: The unit tests of the library. Like for the benchmarks, the actual code lies in the `*.inc` files.
- `tools/`: Optional helper scripts for development.

In addition to this top-level structure, the library itself as well as the benchmarks, examples, and tests are further split into common and backend-specific code:

- **Common code**: Most of the functionality works the same for any backend, so their implementation is shared between them. Library implementations are split into files with a `_detail` suffix and put into an `impl/` subdirectory. Benchmark and test implementations are put into `*.inc` files.
- **Backend-specific code**: Some functions require calling functions of the backend, e.g. CUDA-only functions in case of the CUDA backend. These functions are put into backend directory, e.g. `src/stdgpu/cuda/`. Benchmark and test implementations are compiled in backend-specific files, e.g. `*.cu` in case of the CUDA backend.


```{toctree}
:hidden:

contributing/coding_style
contributing/documentation
contributing/tests
```
