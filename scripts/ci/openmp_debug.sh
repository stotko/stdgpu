#!/bin/sh
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Download external dependencies
sh scripts/utils/download_dependencies.sh

# Configure project
sh scripts/utils/configure_debug.sh -DSTDGPU_BACKEND=STDGPU_BACKEND_OPENMP -Dthrust_ROOT=external/thrust

# Build project
sh scripts/build_debug.sh
