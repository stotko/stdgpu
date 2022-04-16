#!/bin/bash
set -e

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
fi

# Remove first parameter
shift

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Configure project
sh scripts/utils/configure.sh $CONFIG -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP -DSTDGPU_TREAT_WARNINGS_AS_ERRORS=ON $@
