#!/bin/bash
set -e

# Check number of input parameters
if [ "$#" -ne 0 ] && [ "$#" -ne 1 ]; then
    cmake -E echo "build: Expected 0 or 1 parameters, but received $# parameters"
    exit 1
fi

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
fi

# Build project
cmake -E cmake_echo_color --blue ">>>>> Build stdgpu project ($CONFIG)"
cmake --build build --config ${CONFIG} --parallel 13
