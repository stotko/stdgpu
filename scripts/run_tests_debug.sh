#!/bin/sh
set -e

# Run tests
cmake -E cmake_echo_color --blue ">>>>> Run tests"
cmake -E chdir build ctest -V -C Debug
