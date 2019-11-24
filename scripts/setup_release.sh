#!/bin/sh
set -e

# Create build directory
sh scripts/utils/create_empty_directory.sh build

# Configure project
sh scripts/utils/configure_release.sh

# Build project
sh scripts/build_release.sh

# Run tests
sh scripts/run_tests_release.sh
