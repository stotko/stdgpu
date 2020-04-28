#!/bin/sh
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build


# Configure project
# NOTE: Set C++ compiler to HCC rather than HIPCC. HIPCC does not pass CMake's compiler check since it needs a --amdgpu-target=<arch> flag.
sh scripts/utils/configure_debug.sh -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP -DSTDGPU_TREAT_WARNINGS_AS_ERRORS=ON -DCMAKE_CXX_COMPILER=hcc
