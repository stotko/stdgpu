## Examples

This directory contains several examples demonstrating how stdgpu can be used. In particular, the examples can be divided into two classes:

- **Host code with device support**. Examples that can be compiled and run by both the *host and device compiler* are put into this directory. This includes most of the functionality that complements the GPU data structures and containers.

- **Device only code**. Since all GPU data structures and containers can be used in *native* code to cover as many use cases as possible, e.g. in custom CUDA kernels, the respective examples must be compiled by the *device compiler*. This requires knowledge about the chosen backend, so they are put into backend-specific subdirectories.
