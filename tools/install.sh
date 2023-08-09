#!/bin/bash
set -e

# Check number of input parameters
if [ "$#" -ne 0 ] && [ "$#" -ne 1 ]; then
    cmake -E echo "install: Expected 0 or 1 parameters, but received $# parameters"
    exit 1
fi

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
fi

# Install project
cmake -E cmake_echo_color --blue ">>>>> Install stdgpu project ($CONFIG)"
cmake --install build --config $CONFIG
