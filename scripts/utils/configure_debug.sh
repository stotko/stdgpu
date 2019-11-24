#!/bin/sh
set -e

# Configure project
cmake -E cmake_echo_color --blue ">>>>> Configure stdgpu project"
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=bin $*
