#!/bin/bash
set -e

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
    shift
fi

# Create build directory
sh tools/backend/helper/create_empty_directory.sh build

# Configure project
sh tools/backend/helper/configure.sh $CONFIG -DSTDGPU_BACKEND=STDGPU_BACKEND_OPENMP -DSTDGPU_COMPILE_WARNING_AS_ERROR=ON -Dthrust_ROOT=external/cccl/thrust -DCMAKE_VERIFY_INTERFACE_HEADER_SETS=ON $@
