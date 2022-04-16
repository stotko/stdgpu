#!/bin/bash
set -e

# Check number of input parameters
if [ "$#" -ne 0 ] && [ "$#" -ne 1 ]; then
    cmake -E echo "setup: Expected 0 or 1 parameters, but received $# parameters"
    exit 1
fi

if [ "$#" = 0 ]; then
    CONFIG="Release"
else
    CONFIG=$1
fi

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Configure project
sh scripts/utils/configure.sh $CONFIG

# Build project
sh scripts/build.sh $CONFIG

# Run tests
sh scripts/run_tests.sh $CONFIG
