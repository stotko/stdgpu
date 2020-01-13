#!/bin/sh
set -e

# Install project
cmake -E cmake_echo_color --blue ">>>>> Install stdgpu project"
cmake --install build --config Release --component stdgpu
