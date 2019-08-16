#!/bin/sh

# Delete directories to allow a clean build
cmake -E cmake_echo_color --red ">>>>> Delete directories from old build"
cmake -E remove_directory build

# Create directories
cmake -E cmake_echo_color --green ">>>>> Create directories from current build"
cmake -E make_directory build

# Invoke CMake on project
cmake -E cmake_echo_color --blue ">>>>> Build stdgpu project"
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=bin


# Build project
sh scripts/build_debug.sh


# Run tests
sh scripts/run_tests_debug.sh
