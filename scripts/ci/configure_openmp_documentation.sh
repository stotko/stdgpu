#!/bin/sh
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Download external dependencies
sh scripts/utils/download_dependencies.sh

# Download doxygen
sh scripts/utils/download_doxygen.sh

# Configure project
sh scripts/utils/configure_release.sh -DSTDGPU_BACKEND=STDGPU_BACKEND_OPENMP -DSTDGPU_TREAT_WARNINGS_AS_ERRORS=ON -Dthrust_ROOT=external/thrust -DDoxygen_ROOT=external/doxygen/bin
