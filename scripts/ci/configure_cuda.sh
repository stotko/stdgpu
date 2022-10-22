#!/bin/bash
set -e

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
    shift
fi

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Configure project
sh scripts/utils/configure.sh $CONFIG -DSTDGPU_BACKEND=STDGPU_BACKEND_CUDA -DSTDGPU_COMPILE_WARNING_AS_ERROR=ON -DCMAKE_VERIFY_INTERFACE_HEADER_SETS=ON $@
