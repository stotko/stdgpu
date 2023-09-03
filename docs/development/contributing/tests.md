# Tests

We check our code by a variety of tests which complement the pre-conditions and post-conditions used throughout the library. In case you add a new class/function or you extend or provide a fix to an existing function, we recommend and encourage you to also include a corresponding unit test in the set of the existing tests.

For running the unit tests, you can use the following command/script:

::::{tab-set}

:::{tab-item} Direct Command
:sync: direct

```sh
cmake -E chdir build ctest -V -C Release
```

:::

:::{tab-item} Provided Script
:sync: script

```sh
bash tools/run_tests.sh Release
```

:::

::::


:::{note}
The `STDGPU_BUILD_TESTS` option must be enabled to also compile the tests, which is already the default if not manually altered, see [](../../getting_started/building_from_source.md#configuration-options).
:::
