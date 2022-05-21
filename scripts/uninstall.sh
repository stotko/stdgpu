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

# Uninstall project
cmake -E cmake_echo_color --blue ">>>>> Uninstall stdgpu project ($CONFIG)"
cmake --build build --config $CONFIG --target uninstall
