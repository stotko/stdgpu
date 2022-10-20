# Style Guide

The following set of guidelines will help you to make your changes conformant with our coding style.


## Coding Style

We use **C++17** throughout the project. Functionality from more recent C++ standards may break compatibility with some of the supported compilers and will be rejected.

The source code is formatted according to a modified version of the Mozilla style guide that is specified in `.clang-format` and enforced by version **10** of `clang-format`. In order to automatically apply these rules to the source code, we provide the CMake targets `check_code_style` and `apply_code_style` as well as respective helper scripts:

- `scripts/utils/check_code_style.sh`
- `scripts/utils/apply_code_style.sh`

Note that other versions of `clang-format`, including more recent ones, may produce slightly different results which will also be considered non-conforming and, consequently, rejected.
