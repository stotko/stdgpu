#!/bin/bash
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Configure project
sh scripts/utils/configure.sh Debug -DSTDGPU_BACKEND=STDGPU_BACKEND_OPENMP -DSTDGPU_BUILD_TEST_COVERAGE=ON -Dthrust_ROOT=external/thrust
