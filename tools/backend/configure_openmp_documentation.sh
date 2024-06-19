#!/bin/bash
set -e

# Create build directory
sh tools/backend/helper/create_empty_directory.sh build

# Configure project
sh tools/backend/helper/configure.sh Release -DSTDGPU_BACKEND=STDGPU_BACKEND_OPENMP -DSTDGPU_BUILD_DOCUMENTATION=ON -DSTDGPU_COMPILE_WARNING_AS_ERROR=ON -Dthrust_ROOT=external/cccl/thrust
