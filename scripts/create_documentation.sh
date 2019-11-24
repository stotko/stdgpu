#!/bin/sh
set -e

# Create and install documentation
cmake -E cmake_echo_color --blue ">>>>> Create documentation"
cmake --build build --target stdgpu_doc
