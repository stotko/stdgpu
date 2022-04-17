#!/bin/bash
set -e

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
    shift
fi

# Configure project
cmake -E cmake_echo_color --blue ">>>>> Configure stdgpu project ($CONFIG)"
cmake -B build -S . -DCMAKE_BUILD_TYPE=$CONFIG -DCMAKE_INSTALL_PREFIX=bin $@
