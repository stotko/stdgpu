#!/bin/bash
set -e

# Build documentation
cmake -E cmake_echo_color --blue ">>>>> Build documentation"
cmake --build build --target stdgpu_doc
