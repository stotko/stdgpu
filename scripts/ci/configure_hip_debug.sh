#!/bin/sh
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build


# Configure project
sh scripts/utils/configure_debug.sh -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP -DSTDGPU_TREAT_WARNINGS_AS_ERRORS=ON
