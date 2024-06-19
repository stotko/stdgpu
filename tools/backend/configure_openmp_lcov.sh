#!/bin/bash
set -e

# Create build directory
sh tools/backend/helper/create_empty_directory.sh build

# Configure project
sh tools/backend/helper/configure.sh Debug -DSTDGPU_BACKEND=STDGPU_BACKEND_OPENMP -DSTDGPU_BUILD_TEST_COVERAGE=ON -Dthrust_ROOT=external/cccl/thrust
