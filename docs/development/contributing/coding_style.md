# Coding Style

We use **C++17** throughout the project. This means that you can use any features from any C++ standard up to this particular version to implement your contribution. However, more recent features cannot be directly used and should be backported, if possible, to make them accessible in the library.

As one design principle of stdgpu is to closely follow the C++ standard library API, the code should obey the respective style:

- Use `snake_case` for classes, functions, and variables.
- Use `ALL_CAPS` with underscores for macros.
- Define all symbols in the `stdgpu` namespace, prefix macros with the `STDGPU_` "namespace".
- Move purely internal code, e.g. non-standard helper functions, into the `stdgpu::detail` namespace.

Furthermore, the source code is formatted according to a modified version of the Mozilla style guide that is specified in the `.clang-format` file and enforced by **clang-format 10**. Note that other versions of clang-format, including more recent ones, may produce slightly different results which however will be considered non-conforming by our CI and, consequently, rejected.

In order to check if the code is correctly formated, you can use the following command/script:


::::{tab-set}

:::{tab-item} Direct Command
:sync: direct

```sh
cmake --build build --target check_code_style
```

:::

:::{tab-item} Provided Script
:sync: script

```sh
bash tools/dev/check_code_style.sh
```

:::

::::


In case the code needs to be reformatted due to non-conforming style, you can use the following command/script:


::::{tab-set}

:::{tab-item} Direct Command
:sync: direct

```sh
cmake --build build --target apply_code_style
```

:::

:::{tab-item} Provided Script
:sync: script

```sh
bash tools/dev/apply_code_style.sh
```

:::

::::
