#!/bin/bash
set -e

# Check number of input parameters
if [ "$#" -ne 0 ] && [ "$#" -ne 1 ]; then
    cmake -E echo "run_tests: Expected 0 or 1 parameters, but received $# parameters"
    exit 1
fi

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
fi

# Run tests
cmake -E cmake_echo_color --blue ">>>>> Run tests ($CONFIG)"
cmake -E chdir build ctest -V -C $CONFIG
