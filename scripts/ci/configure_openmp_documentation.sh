#!/bin/bash
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Configure project
sh scripts/utils/configure.sh Release -DSTDGPU_BACKEND=STDGPU_BACKEND_OPENMP -DSTDGPU_COMPILE_WARNING_AS_ERROR=ON -Dthrust_ROOT=external/thrust -DDoxygen_ROOT=external/doxygen/bin
