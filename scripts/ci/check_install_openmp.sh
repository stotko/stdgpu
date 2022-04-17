#!/bin/bash
set -e

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
    shift
fi

# Create build directory
sh scripts/utils/create_empty_directory.sh build_install_test

# Compile dependent project
cmake -E cmake_echo_color --blue ">>>>> Check installation ($CONFIG)"
cmake -B build_install_test -S test/install_test -DCMAKE_BUILD_TYPE=$CONFIG -Dstdgpu_ROOT=bin -Dthrust_ROOT=external/thrust $@
cmake --build build_install_test --config ${CONFIG} --parallel 13
