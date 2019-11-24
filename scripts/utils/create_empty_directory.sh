#!/bin/sh
set -e

# Check number of input parameters
if [ "$#" -ne 1 ]; then
    cmake -E echo "create_empty_directory: Expected 1 parameter, but received $# parameters"
    exit 1
fi

# Delete old directory
cmake -E cmake_echo_color --red ">>>>> Delete directory \"$1\" from old build"
cmake -E remove_directory $1

# Create new directory
cmake -E cmake_echo_color --green ">>>>> Create directory \"$1\" for new build"
cmake -E make_directory $1
