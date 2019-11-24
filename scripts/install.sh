#!/bin/sh
set -e

# Install project
cmake -E cmake_echo_color --blue ">>>>> Install stdgpu project"
cmake -DCOMPONENT="stdgpu" -P build/cmake_install.cmake
