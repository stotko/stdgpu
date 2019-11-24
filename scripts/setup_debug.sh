#!/bin/sh
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Configure project
sh scripts/utils/configure_debug.sh

# Build project
sh scripts/build_debug.sh

# Run tests
sh scripts/run_tests_debug.sh
