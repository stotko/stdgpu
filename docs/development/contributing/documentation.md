# Documentation

We use [**Doxygen**](https://www.doxygen.org/index.html) for generating the C++ documentation and [**Sphinx**](https://www.sphinx-doc.org/) for the remaining parts. Thus, if you would like to contribute here, we recommend to do a local build of the documentation on your computer and to see how your changes will look like in the end.

For building, the following tools along with their versions are required:

- Doxygen **1.9.6**, which will be automatically built from source if that *exact* version is not found on your system (requires [Bison and Flex utilites](https://www.doxygen.nl/manual/install.html))
- Sphinx including some extensions, which can all be installed by
    ```sh
    pip install -r docs/requirements.txt
    ```

When these documenation dependencies are installed, you can build the documentation using the following command/script:


::::{tab-set}

:::{tab-item} Direct Command
:sync: direct

```sh
cmake --build build --target stdgpu_doc --parallel 8
```

:::

:::{tab-item} Provided Script
:sync: script

```sh
bash tools/dev/build_documentation.sh
```

:::

::::


:::{note}
The `STDGPU_BUILD_DOCUMENTATION` option must be enabled for this purpose, e.g. via `-D<option>=<value>`, see [](../../getting_started/building_from_source.md#configuration-options).
:::


Afterwards, you can view the generated documentation by opening the `build/docs/html/index.html` file in your browser.
