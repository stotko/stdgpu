#!/bin/bash
set -e

# Check code style
cmake -E cmake_echo_color --blue ">>>>> Check code style"
cmake --build build --target check_code_style
